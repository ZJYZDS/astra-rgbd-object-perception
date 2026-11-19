#!/usr/bin/env python3
"""生成可直接送印的 ArUco marker（默认 10 cm，A4 一页一个，按 100% 打印）。

附 100 mm 校验条：打印后量一下，不是 100 mm 就说明打印店缩放过了。

用法:
    python3 scripts/make_aruco_pdf.py --out ~/gt_aruco --size-cm 10 --ids 0,1,2,3,4,5
输出:
    ~/gt_aruco/aruco_10cm.pdf       送印文件（A4，每页一个 marker）
    ~/gt_aruco/marker_10cm_ID*.png  单张 PNG（600 dpi）
    ~/gt_aruco/README.txt
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

A4_MM = (210.0, 297.0)
DPI = 300


def mm2px(v):
    return int(round(v / 25.4 * DPI))


def marker_image(dict_id, marker_id, size_cm, margin_cm=1.0):
    """白底 + marker + 静区白边，返回 (PIL 图, marker 边长 px, 静区 px)。"""
    dic = cv2.aruco.getPredefinedDictionary(dict_id)
    px = int(round(size_cm / 2.54 * 600))          # marker 本体按 600dpi 生成，质量更好
    if hasattr(cv2.aruco, 'generateImageMarker'):
        m = cv2.aruco.generateImageMarker(dic, int(marker_id), px)
    else:
        m = cv2.aruco.drawMarker(dic, int(marker_id), px)
    pad = int(round(margin_cm / 2.54 * 600))
    page = np.full((px + 2 * pad, px + 2 * pad), 255, np.uint8)
    page[pad:pad + px, pad:pad + px] = m
    return Image.fromarray(page)


def font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def build_page(dict_name, marker_id, size_mm, margin_mm=10):
    W, H = mm2px(A4_MM[0]), mm2px(A4_MM[1])
    page = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(page)

    mk = marker_image(getattr(cv2.aruco, dict_name), marker_id, size_mm / 10.0,
                      margin_mm / 10.0)
    mk_px = mm2px(size_mm)
    total_px = mk_px + 2 * mm2px(margin_mm)
    mk = mk.resize((total_px, total_px), Image.NEAREST)

    x0 = (W - total_px) // 2
    y0 = mm2px(80)                                  # 顶部留 80 mm 写说明
    page.paste(mk, (x0, y0))

    # 细边框（便于裁剪，在静区外）
    d.rectangle([x0 - 2, y0 - 2, x0 + total_px + 1, y0 + total_px + 1],
                outline=(220, 220, 220), width=2)

    f_big, f_mid, f_small = font(46), font(34), font(26)
    d.text((W // 2, mm2px(28)), f'{dict_name}   ID {marker_id}',
           fill='black', font=f_big, anchor='mm')
    d.text((W // 2, mm2px(50)), f'黑白边长 {size_mm:g} mm（不含白边）',
           fill='black', font=f_mid, anchor='mm')
    d.text((W // 2, mm2px(64)),
           '打印：100% / 实际大小，不要“适应页面”',
           fill=(90, 90, 90), font=f_small, anchor='mm')

    # 100 mm 校验条
    bx, by = (W - mm2px(100)) // 2, H - mm2px(40)
    d.rectangle([bx, by, bx + mm2px(100), by + mm2px(4)], fill='black')
    d.text((W // 2, by - mm2px(6)), '100 mm 校验条：打印后量这个长度',
           fill=(60, 60, 60), font=f_small, anchor='mb')
    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.expanduser('~/gt_aruco'))
    ap.add_argument('--size-cm', type=float, default=10.0)
    ap.add_argument('--ids', default='0,1,2,3,4,5')
    ap.add_argument('--dict', default='DICT_4X4_50')
    ap.add_argument('--margin-mm', type=float, default=20.0,
                    help='静区白边（毫米）。建议 >= 1 个模块 = 黑边长/6')
    ap.add_argument('--scale', type=float, default=1.0,
                    help='打印补偿系数：打印机缩了 2%% 就填 1.02（黑边长会乘它）')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ids = [int(x) for x in args.ids.split(',') if x.strip()]
    side_mm = args.size_cm * 10 * args.scale
    pages = [build_page(args.dict, mid, side_mm, args.margin_mm) for mid in ids]
    pdf_path = os.path.join(args.out, f'aruco_{int(args.size_cm)}cm.pdf')
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:],
                  resolution=DPI, title=f'ArUco {args.dict}')

    for mid, pg in zip(ids, pages):
        pg.save(os.path.join(args.out, f'page_{int(args.size_cm)}cm_ID{mid}.png'))
        # 单独的 marker（无文字，方便自己拼版）
        mk = marker_image(getattr(cv2.aruco, args.dict), mid, side_mm / 10.0,
                          margin_cm=args.margin_mm / 10.0)
        mk.save(os.path.join(args.out, f'marker_{int(args.size_cm)}cm_ID{mid}.png'))

    with open(os.path.join(args.out, 'README.txt'), 'w') as f:
        f.write(f"""ArUco 打印说明（方案 B：物体实时位姿 GT）
    静区白边    : {args.margin_mm:g} mm（贴在深色物体上不要剪掉）
==========================================
文件        : aruco_{int(args.size_cm)}cm.pdf （A4，每页一个 marker，共 {len(ids)} 页）
字典        : {args.dict}
黑白边长    : {args.size_cm:g} cm（不含静区白边）
IDS         : {ids}

打印要求
--------
1. 缩放必须选 **100% / 实际大小**，不要“适应页面 / 缩放至纸张”；
2. 纸张：普通 A4 白纸，**哑光**（亮面相纸会反光，影响检测）；
3. 墨要黑（灰了解码失败）；
4. 打印后用尺量页面底部 100 mm 校验条，必须 100 mm，否则重打。

使用要求
--------
- marker 必须**平整**贴在物体上（弯曲会让位姿误差变大）；
- 四周 10 mm 白边（静区）必须保留，不能被胶带/物体遮住；
- 物体太小放不下 10 cm marker 时：贴在一块 15x15 cm 硬板上，再把物体固定在板上，
  板与物体的固定变换量一次写进 config/gt_markers.yaml；
- marker 的 ID 与物体类别的对应关系写在 config/gt_markers.yaml。
""")
    print(f'PDF  -> {pdf_path}')
    print(f'PNG  -> {args.out}/marker_{int(args.size_cm)}cm_ID*.png')
    print(f'说明 -> {args.out}/README.txt')


if __name__ == '__main__':
    main()
