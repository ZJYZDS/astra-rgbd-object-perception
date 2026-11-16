#!/usr/bin/env python3
"""修正自动标签的类别 id / 删除错误实例。

用法示例：
    # 把某张图里所有实例改成 remote（id=4）
    python3 scripts/fix_labels.py --labels ~/grasp_ds/labels --file img_0012 --set-class 4

    # 只改第 2 个实例
    python3 scripts/fix_labels.py --labels ~/grasp_ds/labels --file img_0012 --set-class 4 --index 1

    # 删掉某张图的标签（整张图不要了）
    python3 scripts/fix_labels.py --labels ~/grasp_ds/labels --file img_0012 --clear

    # 批量：按清单把某些文件设为某类
    python3 scripts/fix_labels.py --labels ~/grasp_ds/labels --list wrong.txt --set-class 4
"""
import argparse
import os


def patch(path, set_class=None, index=None, clear=False):
    if not os.path.exists(path):
        print(f'跳过（不存在）: {path}')
        return
    if clear:
        open(path, 'w').close()
        print(f'清空 {os.path.basename(path)}')
        return
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if set_class is not None:
        if index is None:
            lines = [f'{set_class} ' + ln.split(' ', 1)[1] for ln in lines]
        else:
            if 0 <= index < len(lines):
                lines[index] = f'{set_class} ' + lines[index].split(' ', 1)[1]
            else:
                print(f'  第 {index} 个实例不存在，跳过')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'{os.path.basename(path)}: {len(lines)} 个实例 -> {[ln.split()[0] for ln in lines]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--file', default='', help='不带扩展名的文件名')
    ap.add_argument('--list', default='', help='文件列表（每行一个名字）')
    ap.add_argument('--set-class', type=int, default=None)
    ap.add_argument('--index', type=int, default=None)
    ap.add_argument('--clear', action='store_true')
    args = ap.parse_args()

    names = []
    if args.file:
        names = [args.file]
    elif args.list:
        with open(args.list) as f:
            names = [ln.strip() for ln in f if ln.strip()]
    for n in names:
        patch(os.path.join(args.labels, n + '.txt'), args.set_class,
              args.index, args.clear)


if __name__ == '__main__':
    main()
