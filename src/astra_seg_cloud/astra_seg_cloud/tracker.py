"""轻量 IoU 跟踪器：给每一帧的检测分配跨帧稳定的 ID。

12fps 下物体位移不大，用「上一帧框 + 速度预测」做 IoU 贪心匹配就足够稳，
且不引入 scipy / Kalman 等额外依赖。丢帧超过 max_age 才回收 ID。
"""

import numpy as np


def iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = (a[:, i][:, None] for i in range(4))
    bx1, by1, bx2, by2 = (b[:, i][None, :] for i in range(4))
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    return (inter / (area_a + area_b - inter + 1e-9)).astype(np.float32)


class _Track:
    __slots__ = ('id', 'box', 'label', 'score', 'vel', 'misses', 'hits')

    def __init__(self, box, label, score):
        self.id = -1
        self.box = box
        self.label = label
        self.score = score
        self.vel = np.zeros(2, dtype=np.float32)
        self.misses = 0
        self.hits = 1


class IouTracker:
    def __init__(self, iou_thres=0.3, max_age=15, velocity_smooth=0.5):
        self.iou_thres = iou_thres
        self.max_age = max_age
        self.alpha = velocity_smooth
        self.tracks = []
        self._next_id = 1
        self.new_ids = 0
        self.total_ids = 0

    @staticmethod
    def _predict(track):
        box = track.box.copy()
        if track.misses > 0:
            shift = track.vel * (track.misses + 1)
            box[0] += shift[0]
            box[1] += shift[1]
            box[2] += shift[0]
            box[3] += shift[1]
        return box

    def update(self, dets):
        """dets: list[Detection]；返回与 dets 等长的 id 列表。"""
        if not dets:
            for t in self.tracks:
                t.misses += 1
            self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
            return []

        det_boxes = np.stack([d.box for d in dets]).astype(np.float32)

        if self.tracks:
            pred = np.stack([self._predict(t) for t in self.tracks])
            iou = iou_matrix(pred, det_boxes)

            # 贪心匹配：IoU 从高到低，先到先得
            ii, jj = np.nonzero(iou >= self.iou_thres)
            pairs = sorted(zip(ii, jj), key=lambda p: iou[p[0], p[1]], reverse=True)
            matched_track, matched_det = {}, {}
            for i, j in pairs:
                if i in matched_track or j in matched_det:
                    continue
                matched_track[i] = j
                matched_det[j] = i
        else:
            matched_track, matched_det = {}, {}

        ids = [0] * len(dets)

        # 命中：更新轨迹
        for ti, di in matched_track.items():
            track = self.tracks[ti]
            det = dets[di]
            center_prev = (track.box[:2] + track.box[2:]) * 0.5
            center_now = (det.box[:2] + det.box[2:]) * 0.5
            self.tracks[ti].vel = (self.alpha * (center_now - center_prev)
                                   + (1.0 - self.alpha) * track.vel)
            track.box = det.box.copy()
            track.label = det.label
            track.score = det.score
            track.misses = 0
            track.hits += 1
            ids[di] = track.id

        # 未命中的轨迹：老化
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track:
                track.misses += 1

        # 新目标
        for di, det in enumerate(dets):
            if di in matched_det:
                continue
            track = _Track(det.box.copy(), det.label, det.score)
            track.id = self._next_id
            self._next_id += 1
            self.new_ids += 1
            self.total_ids += 1
            self.tracks.append(track)
            ids[di] = track.id

        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return ids
