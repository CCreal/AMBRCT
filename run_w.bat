@echo off
REM 1. 切换到目标目录
cd /d "E:\lcc\project_space\UAV-tracking-ftz\cranfield-drone-tracking-gym-main"

REM 2. 初始化 Conda（加载 activate 命令）
CALL "D:\miniconda\miniconda\Scripts\activate.bat"

REM 3. 激活目标环境
CALL conda activate blender

REM 4. 运行 Python 脚本
python test_model_step_w_mkl.py

REM 可选：退出环境（不影响其他窗口）
CALL conda deactivate
pause