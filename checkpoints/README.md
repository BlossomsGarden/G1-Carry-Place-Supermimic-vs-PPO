# 已训练权重与复现入口

这两份文件是原实验运行目录的**完整 PyTorch checkpoint**（包括优化器状态），其旁边的 `params/agent.json` 是相应训练 run 的推理配置。`eval_track.py` 默认在 checkpoint 所在目录读取这个配置，缺少它可能按不同网络结构实例化。只加载可信来源的 PyTorch 模型。

| 路径 | 学习器 / 迭代 | 文件字节 | SHA256 |
|---|---|---:|---|
| `supermimic/model_1000.pt` | 单卡 H32 / 朝向 std 1.2，1,000 轮 | 223,742,486 | `2c326307c91deff24787320d0bde5d4b3882675314ca5255754c16720b825022` |
| `ppo/model_3999.pt` | 四卡 PPO-Sim，3,999 轮存档 | 113,770,290 | `1ecfde005be0f227d5f7f6302bfd4de4c58f2a2e0d7affd4b0155b2fe3457198` |

权重由 **Git LFS** 管理，克隆仓库后运行 `git lfs pull` 并检查文件字节数；只有几十/几百字节的文件是 LFS 指针，不是模型。可在可信的 Linux GPU 环境中直接评测：

~~~bash
git lfs pull
sha256sum checkpoints/supermimic/model_1000.pt checkpoints/ppo/model_3999.pt

export CARRY_PYTHON=/path/to/wan/bin/python  # 可省略：默认使用 PATH 上的 python
export CARRY_ENV_SH=/path/to/site/env.sh  # 仅本站点需要时设置
bash scripts/eval_checkpoints.sh supermimic 0 2027 64
bash scripts/eval_checkpoints.sh ppo 0 2027 64
~~~

脚本还可传入第 5 个参数指定外部 checkpoint 完整路径。终点三维到位、间歇持物、连续双手三个结果写在 `runs/evaluations/<算法>_<种子>_n64_metrics.json`；上面的两次运行产生**新的一批**随机化物理回放，不能逐集对号先前归档的数据。

原实验服务器离线时，在可访问 GitHub 的电脑先执行 `git lfs pull`，再将完整的 `checkpoints/` 目录复制到服务器本项目路径；也可直接用推理脚本的第 5 个参数指定仍保存在服务器历史 run 中的 checkpoint。
