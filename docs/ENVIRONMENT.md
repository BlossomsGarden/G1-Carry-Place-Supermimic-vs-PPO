# 🧰 已实测环境与安装边界

| 层 | 已实测版本 / 设备 | 说明 |
|---|---|---|
| 操作系统 | Ubuntu 20.04.1 LTS | 服务器上的实测平台；本仓库未在 Windows 物理训练 |
| GPU | NVIDIA RTX 4090 24 GiB；另有 48 GiB 型号 | 本页四组 SuperMimic 在 24 GiB 单卡；正式 PPO 四张卡 |
| 宿主驱动 | NVIDIA 550.144.03 | **须安装在宿主机**；conda/pip 只提供用户态库，不安装显卡内核驱动 |
| Python | `wan` 中实测 **3.10.13**；`pyproject.toml` 仍声明 `>=3.12,<3.14` | 历史训练与推理直接运行源码，`PYTHONPATH=src`；不能把源码可运行误称为此 Python 版本符合可编辑安装的元数据声明 |
| 深度学习 | PyTorch `2.6.0+cu124`、torchvision `0.21.0`、torchaudio `2.6.0` | CUDA 12.4 对应的 wheel；`nvidia-smi` 中 CUDA 版本不等于 toolkit 安装状态 |
| 仿真/任务 | MuJoCo `3.10.0`、mjlab `1.4.0`、mujoco-warp `3.8.0.3`、warp-lang `1.14.0` | 以**服务器运行时组合**为准；来源仓库的 uv 解析锁曾选出更新的 torch/warp，故不把那份锁当成本实验环境 |
| 算法 | 仓库内 `rsl_rl` 5.2.0 + `src/mjlab_g1` | 主模型源码、评估和任务配置均在本仓库 |

历史运行证明上述 **驱动 + 依赖组合**可在该服务器上训练和物理评估；不等于宣称普通 `pip install -e .` 会自动解析出同一组版本，也不保证其他驱动/GPU 相同表现。网络不通时，在可访问网络的机器下载 wheel 和 mjlab v1.4.0 源码再复制到服务器；无需在服务器实时联网。

**原样复现实验首选已有 `wan` 环境（Python 3.10.13）**：检查包与驱动后，按 README 的 `PYTHONPATH=src` 启动脚本，不对 `wan` 执行 `pip install -e .`（项目元数据目前要求 Python ≥3.12）。下面的 Python 3.12 指令只展示**从零新建环境**的方向，没有验证该新环境与历史 `wan` 的二进制组合逐项相同；先安装对应 PyTorch CUDA wheel：

~~~bash
conda create -n g1-carry python=3.12 -y
conda activate g1-carry
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# 备齐 mjlab v1.4.0 及其必要依赖后，再固定本实验仿真层；必要时从本地 wheelhouse 安装
python -m pip install mujoco==3.10.0 mujoco-warp==3.8.0.3 warp-lang==1.14.0 onnxruntime scipy pynput tensordict GitPython onnx onnxscript
python -m pip install --no-deps -e /path/to/mjlab-v1.4.0
python -m pip install --no-deps -e ./rsl_rl
python -m pip install --no-deps -e .
~~~

`--no-deps` 是为了保持验证过的 CUDA/PyTorch 组合；新环境仍需要 mjlab 其余依赖齐备。用 `python -m pip check` 检查缺失与声明冲突，按实际报错补包并通过下方 smoke 验证；遇到声明的 torch 下界与 2.6 冲突时不要强称 pip 解析“官方支持”，本项目报告的是**这台机器上已执行通过**的运行时组合。源项目的 `uv.lock` 曾解析为 torch 2.13.0、warp-lang 1.15.0，本仓库故意未携带该锁；不要以无参数 `uv sync` 覆盖已验证环境。

~~~bash
nvidia-smi
python -c "import torch, mujoco, importlib.metadata as m; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), mujoco.__version__, m.version('mjlab'))"
export PYTHONPATH=src MUJOCO_GL=disable
python -m unittest discover -s tests -p 'test_carryplace_eval.py' -v
CUDA_VISIBLE_DEVICES=0 python src/mjlab_g1/scripts/train_track.py G1-SMTrack-CarryPlace --learner ppo --smoke --no-compile
~~~

烟测只检查接线、仿真、反向传播和写 checkpoint，不是“训练完成”。原实验机还会加载站点自己的 ROS2 环境脚本；如果你的主机需要它设置 CUDA/库路径，可先运行 `source /path/to/ROS2/env.sh`，或设置 `CARRY_ENV_SH=/path/to/ROS2/env.sh` 交给启动脚本加载。并行训练还需多卡之间能通信、足够的宿主内存与磁盘空间。端口/本机名问题优先用单机 `127.0.0.1` rendezvous，见 [可复现命令](REPRODUCE.md)。
