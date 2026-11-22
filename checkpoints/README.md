# 已训练权重与复现入口

这四份文件是原实验运行目录的**完整 PyTorch checkpoint**（包括优化器状态），其旁边的 `params/agent.json` 是相应训练 run 的推理配置。`eval_track.py` 默认在 checkpoint 所在目录读取这个配置，缺少它可能按不同网络结构实例化。只加载可信来源的 PyTorch 模型。

| 路径 | 学习器 / 迭代 | 文件字节 | SHA256 |
|---|---|---:|---|
| `supermimic/model_1000.pt` | 单卡 H32 / 朝向 std 1.2，1,000 轮 | 223,742,486 | `2c326307c91deff24787320d0bde5d4b3882675314ca5255754c16720b825022` |
| `supermimic_continued/model_1300.pt` | 从上述单卡权重续训 300 轮；终点到位复测 128/128 | 223,742,998 | `e0fc6f63b478a5dde6b4d981a766b6aa9e8c7bfc66e368efcacbbf45bc6ae6e2` |
| `supermimic_4gpu/model_1500.pt` | 四卡从零训练，每卡 2,048 环境；两种子到位复测 128/128，第三种子 64/64 | 223,743,321 | `f6781e2d562fde9ddd8abe7f28608623a875219687097be1c7cec19ab0c893bf` |
| `ppo/model_3999.pt` | 四卡 PPO-Sim，3,999 轮存档 | 113,770,290 | `1ecfde005be0f227d5f7f6302bfd4de4c58f2a2e0d7affd4b0155b2fe3457198` |

四份权重放在 [Hugging Face 模型仓库](https://huggingface.co/BlossomsGarden/G1-Carry-Place-Supermimic-vs-PPO)，GitHub 仅保留各自的 `params/agent.json`。从 GitHub 仓库根目录下载后应检查文件大小及 SHA256。`eval_track.py` 会读取 checkpoint 所在目录的 `params/agent.json`；请只加载可信来源的 PyTorch 模型。

~~~bash
hf download BlossomsGarden/G1-Carry-Place-Supermimic-vs-PPO --include 'checkpoints/*/model_*.pt' --local-dir .
sha256sum checkpoints/supermimic/model_1000.pt checkpoints/ppo/model_3999.pt
sha256sum checkpoints/supermimic_continued/model_1300.pt
sha256sum checkpoints/supermimic_4gpu/model_1500.pt

export CARRY_PYTHON=/path/to/wan/bin/python  # 可省略：默认使用 PATH 上的 python
export CARRY_ENV_SH=/path/to/site/env.sh  # 仅本站点需要时设置
bash scripts/eval_checkpoints.sh supermimic 0 2027 64
bash scripts/eval_checkpoints.sh supermimic 0 2027 64 checkpoints/supermimic_continued/model_1300.pt
bash scripts/eval_checkpoints.sh supermimic 0 2027 64 checkpoints/supermimic_4gpu/model_1500.pt
bash scripts/eval_checkpoints.sh ppo 0 2027 64
~~~

脚本还可传入第 5 个参数指定外部 checkpoint 完整路径。终点三维到位、间歇持物、连续双手三个结果写在 `runs/evaluations/<算法>_<种子>_n64_metrics.json`；上面的两次运行产生**新的一批**随机化物理回放，不能逐集对号先前归档的数据。

原实验服务器离线时，在联网电脑下载完整权重，再连同 GitHub 已跟踪的 `checkpoints/*/params/agent.json` 复制到服务器本项目对应路径；也可直接用推理脚本的第 5 个参数指定仍保存在服务器历史 run 中的 checkpoint。
