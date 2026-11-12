"""YOLOv8-seg ONNX inference.

导出约定（ultralytics 8.x, imgsz=640, opset=12）：
    input  "images" : float32[1, 3, 640, 640]，RGB，0~1，letterbox 填充值 114
    output0         : float32[1, 4 + nc + 32, 8400]   框 + 类别分数 + mask 系数
    output1         : float32[1, 32, 160, 160]        mask 原型

这里不依赖 ultralytics 运行，只用 onnxruntime + opencv。
"""

import cv2
import numpy as np

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]


class Detection:
    """一个实例：原图坐标下的框、类别、置信度、二值 mask。"""

    __slots__ = ('box', 'score', 'class_id', 'mask')

    def __init__(self, box, score, class_id, mask):
        self.box = box          # np.float32[4]  (x1, y1, x2, y2)
        self.score = score      # float
        self.class_id = class_id
        self.mask = mask        # np.bool_[H, W]

    @property
    def label(self):
        if 0 <= self.class_id < len(COCO_CLASSES):
            return COCO_CLASSES[self.class_id]
        return f'class_{self.class_id}'


def letterbox(img, new_size=640, color=(114, 114, 114)):
    """等比缩放 + 居中填充，返回 (图, 缩放比, (left, top, new_w, new_h))。"""
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_size - new_w) / 2.0, (new_size - new_h) / 2.0
    left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
    right, bottom = int(round(dw + 0.1)), int(round(dh + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top, new_w, new_h)


def _nms(boxes_xywh, scores, iou_thres):
    if len(boxes_xywh) == 0:
        return np.empty(0, dtype=np.int64)
    idx = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), scores.tolist(), 0.0, iou_thres)
    if idx is None or len(idx) == 0:
        return np.empty(0, dtype=np.int64)
    return np.asarray(idx).reshape(-1).astype(np.int64)


def resolve_class_ids(names):
    """把类别名（或数字 id）解析成 COCO 类别 id 集合。"""
    ids = set()
    unknown = []
    for raw in names:
        name = str(raw).strip().lower()
        if not name:
            continue
        if name.isdigit():
            ids.add(int(name))
        elif name in COCO_CLASSES:
            ids.add(COCO_CLASSES.index(name))
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f'未知类别: {unknown}；COCO 可用类别见 yolo_seg.COCO_CLASSES')
    return ids


class YoloSegOnnx:
    def __init__(self, model_path, imgsz=640, conf_thres=0.25, iou_thres=0.45,
                 allowed_class_ids=None, providers=None, logger=None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        if providers is None:
            avail = ort.get_available_providers()
            providers = [p for p in ('CUDAExecutionProvider',
                                     'CPUExecutionProvider') if p in avail]
        try:
            self.session = ort.InferenceSession(model_path, so, providers=providers)
        except Exception:
            self.session = ort.InferenceSession(
                model_path, so, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.allowed_class_ids = (set(allowed_class_ids)
                                  if allowed_class_ids else None)
        self.provider = self.session.get_providers()[0]
        self._log = logger or (lambda m: None)
        self._log(f'YOLO-seg 模型已加载: {model_path} (provider={self.provider})')

    # ------------------------------------------------------------------ 前处理
    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]
        padded, r, letter = letterbox(img_bgr, self.imgsz)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]
        return np.ascontiguousarray(blob), r, letter, h, w

    # ------------------------------------------------------------------ 推理
    def warmup(self, iters=2):
        """预热 CUDA / 图优化，避免第一帧卡 3 秒把队列灌满。"""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        import time
        t0 = time.perf_counter()
        for _ in range(iters):
            self.infer(dummy)
        self._log(f'模型预热完成 ({(time.perf_counter() - t0) * 1000:.0f} ms)')

    def infer(self, img_bgr):
        """返回 list[Detection]，mask 为原图尺寸的 bool 数组。"""
        blob, r, (left, top, new_w, new_h), h, w = self._preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._decode(outputs, r, left, top, new_w, new_h, h, w)

    def _decode(self, outputs, r, left, top, new_w, new_h, h, w):
        preds = None
        protos = None
        for out in outputs:
            if out.ndim == 4:
                protos = out[0]           # (32, 160, 160)
            elif out.ndim == 3:
                preds = out[0]
        if preds is None or protos is None:
            raise RuntimeError(
                f'UNEXPECTED OUTPUT SHAPES: {[o.shape for o in outputs]}')

        # (116, 8400) -> (8400, 116)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
        n_masks = protos.shape[0]
        n_cls = preds.shape[1] - 4 - n_masks
        if n_cls <= 0:
            raise RuntimeError('解析不出类别数，模型可能不是 yolov8-seg')

        boxes_cxcywh = preds[:, :4]
        cls_scores = preds[:, 4:4 + n_cls]
        mask_coeffs = preds[:, 4 + n_cls:]

        class_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(len(class_ids)), class_ids]

        keep = scores >= self.conf_thres
        if self.allowed_class_ids is not None:
            keep &= np.isin(class_ids, list(self.allowed_class_ids))
        if not np.any(keep):
            return []
        boxes_cxcywh = boxes_cxcywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        mask_coeffs = mask_coeffs[keep]

        # cx,cy,w,h -> x,y,w,h
        xywh = boxes_cxcywh.copy()
        xywh[:, 0] -= xywh[:, 2] / 2.0
        xywh[:, 1] -= xywh[:, 3] / 2.0

        kept = _nms(xywh, scores, self.iou_thres)
        if len(kept) == 0:
            return []

        xywh = xywh[kept]
        scores = scores[kept]
        class_ids = class_ids[kept]
        mask_coeffs = mask_coeffs[kept]

        # 640 空间的框，用来把 prototype mask 裁回框内
        xywh640 = xywh.copy()

        # letterbox 逆变换：640 坐标 -> 原图坐标
        xywh[:, 0] = (xywh[:, 0] - left) / r
        xywh[:, 1] = (xywh[:, 1] - top) / r
        xywh[:, 2] /= r
        xywh[:, 3] /= r

        # mask = coeffs @ protos -> (N, 160*160) -> sigmoid
        flat = protos.reshape(n_masks, -1).astype(np.float32)
        masks = mask_coeffs.astype(np.float32) @ flat
        masks = 1.0 / (1.0 + np.exp(-masks))
        mh, mw = protos.shape[1], protos.shape[2]
        masks = masks.reshape(-1, mh, mw)
        sx, sy = mw / self.imgsz, mh / self.imgsz

        dets = []
        for i in range(len(kept)):
            x1 = float(np.clip(xywh[i, 0], 0, w - 1))
            y1 = float(np.clip(xywh[i, 1], 0, h - 1))
            x2 = float(np.clip(xywh[i, 0] + xywh[i, 2], 0, w - 1))
            y2 = float(np.clip(xywh[i, 1] + xywh[i, 3], 0, h - 1))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            # 关键：mask 必须在框内取，否则 prototype 在框外也会 >0.5，
            # 导致 mask 溢出到整幅图、点云混入其他物体。
            gx1 = int(np.clip(np.floor(xywh640[i, 0] * sx), 0, mw - 1))
            gy1 = int(np.clip(np.floor(xywh640[i, 1] * sy), 0, mh - 1))
            gx2 = int(np.clip(np.ceil((xywh640[i, 0] + xywh640[i, 2]) * sx), gx1 + 1, mw))
            gy2 = int(np.clip(np.ceil((xywh640[i, 1] + xywh640[i, 3]) * sy), gy1 + 1, mh))

            crop = masks[i][gy1:gy2, gx1:gx2]
            if crop.size == 0:
                continue

            bw = max(1, int(round(x2 - x1)))
            bh = max(1, int(round(y2 - y1)))
            crop = cv2.resize(crop, (bw, bh), interpolation=cv2.INTER_LINEAR)

            full = np.zeros((h, w), dtype=np.float32)
            ox1, oy1 = int(round(x1)), int(round(y1))
            ox2, oy2 = min(w, ox1 + bw), min(h, oy1 + bh)
            full[oy1:oy2, ox1:ox2] = crop[:oy2 - oy1, :ox2 - ox1]
            mask = full > 0.5
            if not mask.any():
                continue

            dets.append(Detection(
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
                score=float(scores[i]),
                class_id=int(class_ids[i]),
                mask=mask))

        return dets
