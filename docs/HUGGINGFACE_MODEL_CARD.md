---
language: zh
tags:
- robotics
- humanoid
- mujoco
- reinforcement-learning
- imitation-learning
---

# G1 Carry & Place · SuperMimic vs PPO

在 mjlab + MuJoCo 中用一条 SUGAR CarryBox 参考（500 帧，10 秒），训练 G1 人形机器人抓箱、搬运和放置的四份 PyTorch checkpoint。**推理代码、任务资产、评测记录及许可证边界**见 [GitHub 项目](https://github.com/BlossomsGarden/G1-Carry-Place-Supermimic-vs-PPO)。这些权重不是 SuperMimic 论文官方发布的模型。

| 模型文件 | 训练设置 | 两个种子的终点到位率 | 间歇持物 | 连续双手 |
|---|---|---:|---:|---:|
| `checkpoints/supermimic/model_1000.pt` | 单卡，4,096 环境，1,000 轮 | 124/128（原始权重的新一批复评） | 107/128 | 0/128 |
| `checkpoints/supermimic_continued/model_1300.pt` | 单卡续训至 1,300 轮 | 128/128（同卡复测 128/128） | 102/128 | 0/128 |
| `checkpoints/supermimic_4gpu/model_1500.pt` | 四卡，从零训至 1,500 轮 | 128/128（同卡复测 128/128） | 127/128 | 7/128 |
| `checkpoints/ppo/model_3999.pt` | 四卡 PPO-Sim，4,000 轮 | 127/128 | 127/128 | 53/128 |

主指标为**物理回放最后一帧箱中心到目标中心的三维距离 ≤ 0.12 m**；不要求持续抓握、物体静止或部署到真实机器人。以上数字来自不同归档批次，历史权重的新一批 256 集复评与新续训权重的选权重评测不可逐集拼接。单次 128/128 不代表普遍成功率为 100%。逐集物理证据及训练耗时见 [GitHub 报告](https://github.com/BlossomsGarden/G1-Carry-Place-Supermimic-vs-PPO/blob/main/docs/RESULTS.md)。

## 下载并评测

先克隆 GitHub 代码仓库，在仓库根目录执行以下命令。Hugging Face 同时保存每个模型对应的 `params/agent.json`；GitHub 仓库也跟踪这些小配置文件。每个权重必须与自己的配置处在同一个模型目录中。

~~~bash
hf download BlossomsGarden/G1-Carry-Place-Supermimic-vs-PPO --include 'checkpoints/*/model_*.pt' --local-dir .
sha256sum checkpoints/*/model_*.pt  # 对照 GitHub checkpoints/README.md 列出的哈希
bash scripts/eval_checkpoints.sh supermimic 0 2027 64 checkpoints/supermimic_4gpu/model_1500.pt
bash scripts/eval_checkpoints.sh ppo 0 2027 64
~~~

复现实测运行环境：Ubuntu 20.04.1、RTX 4090、NVIDIA 驱动 550.144.03、Python 3.10.13、PyTorch 2.6.0+cu124、mjlab 1.4.0、MuJoCo 3.10.0。详见 [环境说明](https://github.com/BlossomsGarden/G1-Carry-Place-Supermimic-vs-PPO/blob/main/docs/ENVIRONMENT.md)。模型使用仿真中特权机器人及箱体状态，没有相机、力传感器或真机验证。PyTorch checkpoint 包含可执行反序列化对象，只从可信来源加载；源数据与 G1 资产的使用条件见项目的 THIRD_PARTY.md。
