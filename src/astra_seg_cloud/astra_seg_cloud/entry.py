"""启动引导。

这台机器上 ~/.local 里装了 NumPy 2.x，会把 ROS Humble（按 NumPy 1.x 编译）的
cv_bridge 撑坏。这里在导入任何科学计算库之前做两件事：

1. 从 sys.path 里摘掉 user site（~/.local），避免 numpy 2 覆盖系统 numpy 1.x；
2. 把项目自带的隔离依赖目录（默认 ~/Astra_ws/.venv）加到最前面。

这样 `ros2 run` / `ros2 launch` 都能直接跑，不需要用户手工改环境变量。
"""

import os
import sys


def _strip_user_site():
    try:
        import site
        paths = site.getusersitepackages()
    except Exception:
        return
    if isinstance(paths, str):
        paths = [paths]
    drop = {os.path.realpath(p) for p in paths if p}
    sys.path[:] = [p for p in sys.path if os.path.realpath(p) not in drop]


def _prepend_venv():
    venv = os.environ.get('ASTRA_SEG_VENV', os.path.expanduser('~/Astra_ws/.venv'))
    ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
    site_pkgs = os.path.join(venv, 'lib', ver, 'site-packages')
    if os.path.isdir(site_pkgs):
        real = os.path.realpath(site_pkgs)
        if real not in {os.path.realpath(p) for p in sys.path}:
            sys.path.insert(0, real)
        return site_pkgs
    return None


def main(args=None):
    _strip_user_site()
    _prepend_venv()

    # 延迟导入：必须在前面的 path 调整之后
    from astra_seg_cloud.pipeline_node import main as _main
    return _main(args)


if __name__ == '__main__':
    sys.exit(main())
