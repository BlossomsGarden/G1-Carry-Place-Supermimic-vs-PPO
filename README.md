# 🤖 G1 Carry & Place：SuperMimic vs PPO

**一条 10 秒的抱箱示范，能否教会 G1 走过去、双手抱起、搬到另一处并放稳？** 我把 SUGAR CarryBox 的一条轨迹接进 **mjlab + MuJoCo**，在相同任务上对照“真实物理采样 + PPO-Sim”与“世界模型 + 可微分想象更新”的 SuperMimic 实现。代码、单条示范、四组单卡消融、四卡 PPO 基线和真实物理评测证据都在这里。

| 🎞️ 示范：希望机器人怎样动 | ⚡ SuperMimic：看似搬放成功 | 🦾 PPO：连续双手抱箱成功 |
|:---:|:---:|:---:|
| <img src="assets/demo_reference.gif" width="280" alt="SUGAR 运动学参考轨迹，非训练策略"> | <img src="assets/demo_sm_intermit.gif" width="280" alt="SuperMimic 真物理策略，间歇持物成功，严格双手失败"> | <img src="assets/demo_ppo_strict.gif" width="280" alt="PPO 真物理策略，连续双手持物成功"> |
| [完整 MP4](assets/carry_place_reference_098.mp4) · **仅运动学指定姿态** | [完整 MP4](assets/sm4096_h32_ori12_iter1000_seed161803_episode0_physical.mp4) · **间歇持物通过 / 连续双手失败** | [完整 MP4](assets/ppo3999_seed314159_episode59_physical.mp4) · **严格连续双手通过** |

> **别被视频角标误导：**SuperMimic 视频标着 `STRICT FAILURE`，指原评测器要求“整个搬运段双手一刻不能松”的指标不通过；该片段**确实**举起并离地净搬运 1.526 m、到位放下。取搬放结果与持续稳定夹持是两个问题。[四个可播放案例与逐帧解释](docs/DEMOS.md) · [地面假接触核查](assets/sm4096_h32_ori12_iter1000_seed161803_episode0_floor_audit.png)

## ✨ 30 秒看结论

| 训练器 / 选中权重 | GPU / 同时仿真环境 | 训练至权重 / 完整训练 | 真实交互总量 | 稳态真实环境步/秒 | 训练峰值显存 | 同种子 2027+314159：间歇持物 | 同种子：连续双手 |
|---|---|---:|---:|---:|---|---:|---:|
| **SuperMimic H32 / 朝向 std 1.2** | 1 × RTX 4090 / 4,096 | **3.75 h / 3.76 h** | 1,037 万 | 776 | 11,754 MiB | **110/128 = 85.9%** | **0/128** |
| **PPO-Sim 终点** | 4 × RTX 4090 / 8,192（每卡 2,048） | **2.58 h / 2.58 h** | 7.864 亿 | 86,216 | 6,288 / 4,922 / 4,922 / 4,922 MiB | **127/128 = 99.2%** | **53/128 = 41.4%** |

此处两方法用**相同两种子、相同几何校正后的物理判据**；SuperMimic 另外的第三种子有 58/64，三种子合计 **168/192 = 87.5% 间歇持物**。PPO 原始未校正的严格成绩为 51/128；修正不可能的触地记录后为 53/128。两方法的训练 GPU 数、总环境数、网络与真实数据预算都不同，这张表是**工程实测**，不是“SuperMimic 在同等成功率下快 1000×”的证据。2 秒采样显存是**整卡已用量峰值**，评测开销未计入训练。[口径、日志、原始 64 回合证据与失败分析 →](docs/RESULTS.md)

## 🧪 四个 SuperMimic 配方：单卡 4,096 环境

相同训练种子 0、相同一条 500 帧示范、相同 1,000 轮上限；**先用种子 0 的 16 条开发集选择检查点**，再用种子 2027 / 314159 / 161803 各 64 条评测。H 是**每个想象起点向未来展开的步数**，朝向 std 是箱体**朝向奖励尺度**（rad），不是仿真环境数或动作噪声。所有策略想象都分 16 块 × 256 起点累积梯度。

| H / 朝向 std | 单卡 GPU | 选中第几轮；达到时间 | 完整 1,000 轮 | 稳态真实环境步/秒 | 峰值显存 | 间歇持物：三个种子各 64 条 | 连续双手 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32 / 0.4 | 4 | 1,000；3.83 h | 3.84 h | 778 | 11,754 MiB | 51 / 51 / 51；**79.7%** | 0/192 |
| 16 / 0.4 | 5 | 800；1.91 h | 2.36 h | 1,278 | 10,904 MiB | 32 / 34 / 36；**53.1%** | 0/192 |
| **32 / 1.2** | **6** | **1,000；3.75 h** | **3.76 h** | **776** | **11,754 MiB** | **58 / 52 / 58；87.5%** | **0/192** |
| 16 / 1.2 | 7 | 800；1.94 h | 2.39 h | 1,266 | 10,904 MiB | 22 / 35 / 36；**48.4%** | 0/192 |

**关键取舍：**H16 快约 1.6×，但“更新更快”不必然等于“最终抱箱更稳”。H16/1.2 第 800 轮被选中，第 1,000 轮任务结果下降；单看世界模型损失或训练 reward 会漏掉退化。H32/1.2 可以在一张 4090 上约 3.75 小时训出明显可见的取搬放，但持续双手握持仍是**0/192**。每组真实交互量 **10,371,072**（含 replay 预填充），想象 rollout 是模型计算，不算真实 MuJoCo 步数。[更细的时序与失败原因 →](docs/RESULTS.md)

## 🧠 方法：同任务，梯度路径不同

~~~mermaid
flowchart LR
    D["SUGAR data_098<br/>1 条 × 500 帧"] --> R["G1 + 箱体参考<br/>足底/手箱几何修正"]
    R --> E["mjlab + MuJoCo<br/>真实接触、同观测与域随机化"]
    E --> P["PPO-Sim<br/>物理 rollout + 优势估计"]
    E --> M["SuperMimic<br/>真实 replay → 世界模型"]
    M --> I["4,096 个想象起点<br/>H16 / H32 可微展开"]
    P --> V["统一物理复评<br/>间歇持物 + 严格双手"]
    I --> V
~~~

- **PPO-Sim** 从仿真批量采集实际交互，经 PPO 优势估计/更新策略；训练步速高，消耗更多**真实仿真**转换。这里的 `--learner ppo` 不是仓库别处的 AMP-PPO Lift 实验。
- **SuperMimic** 从真实仿真数据拟合状态转移模型，在模型里用 4,096 个起点做短时域想象，将跟踪奖励梯度穿过世界模型更新策略；模型误差、接触非光滑和箱体终点姿态都可能影响操作。`16×256` 是同一次 4,096 起点更新的**显存分块**，不是四个 epoch，也不是多开 4,096 个真实仿真环境。
- **观察与局限**：世界模型接收机器人+箱体的状态与随机化信息（含 84 维物理状态），本实验没有摄像头闭环、力觉或真机验证；仿真中特权状态可得不代表真机可直接部署。[模型输入与实现](src/mjlab_g1/rl/supermimic/features.py) · [策略更新](src/mjlab_g1/rl/supermimic/algorithm.py)。

## 📦 Dataset 与任务

- **只训练一条参考**：SUGAR CarryBox `data_098`，10 秒、50 Hz、500 帧；同时含 G1 29 关节、身体轨迹、箱体 6D 位姿/速度和接触标签。源 processed 数据虽有 100 条、49,043 帧（87 条可转换），当前策略**只读一条修正后的 v2**。原轨迹→接触 v1→足底+手箱 v2 的文件和 [SHA/字段说明](dataset/README.md) 随仓库提供。
- **物理任务**：29 DoF G1，0.5 kg 箱体（约 0.400×0.546×0.536 m），从地面取、至少离地搬运 0.5 m、到目标区域松手并站稳；控制 50 Hz。示范的箱体位姿是追踪目标，**不是强制绑定到机器人手上**。[任务定义](src/mjlab_g1/tasks/smtrack/env_cfgs.py) · [真实物理评分](src/mjlab_g1/tasks/smtrack/carryplace_eval.py)。
- **为何修正参考**：原始转换的持物时手箱穿透 p95 达 9.51 cm，接近阶段仅 23.0% 的帧有脚部接地；v2 几何审计分别改善为 0.21 cm 和 98.7%。几何修正并不保证动力学可实现。[数据审计](dataset/README.md)。

## 🛠️ 环境与复现入口

实测：**Ubuntu 20.04.1 LTS · RTX 4090 · 驱动 550.144.03 · Python 3.12 · PyTorch 2.6.0+cu124 · mjlab 1.4.0 · MuJoCo 3.10.0 · mujoco-warp 3.8.0.3 · warp-lang 1.14.0**。GPU 驱动安装于**宿主机**，仅有 conda 环境不足以运行 CUDA 仿真；从零搭环境、安装 PyTorch wheel 和已验证环境检查见 [ENVIRONMENT.md](docs/ENVIRONMENT.md)。本仓库不使用上游 `uv.lock` 的更新版 torch 解析结果覆盖 2.6.0。

~~~bash
# 仓库根目录，已进入具备上述依赖的 Linux conda 环境
python scripts/verify_report.py
export PYTHONPATH=src MUJOCO_GL=disable
python -m unittest discover -s tests -p 'test_carryplace_eval.py' -v

# 历史 PPO 配方：4 卡 × 2,048 环境 = 全局 8,192
CARRY_MASTER_PORT=29571 CARRY_SAVE_INTERVAL=100 bash scripts/run_carry_experiment.sh \
  ppo 0,1,4,5 8192 4000 cp_ppo_v2_s0_4g

# 最佳 SuperMimic 配方：cuda:6 单卡、4,096 环境，H32 / std1.2
CARRY_SAVE_INTERVAL=100 bash scripts/run_carry_experiment.sh \
  supermimic 6 4096 1000 sm4096_h32_ori12_s0 \
  --imag-microbatches 16 --lr-cosine-iters 250000 --imagination-horizon 32 --box-ori-std 1.2
~~~

两条训练会各自耗时数小时；分别运行，或换不重叠 GPU 同时启动。**不含权重原件**：首版提交附代码/演示/逐回合结果，原历史权重仍在实验服务器；重训、64 回合随机化评测、单条 NPZ 重新评分和四组完整命令见 [REPRODUCE.md](docs/REPRODUCE.md)。`scripts/verify_report.py` **只核对已归档数据**，不伪装成重新跑过 GPU 实验。

## 🔎 踩坑与下一步

1. **视觉与判据分层**：箱子完成取搬放可以 87.5%，同时不间断双手仍 0%；[一个“看似成功”的严格失败片段](assets/demo_sm_contact_failure.gif)与[接触时序](assets/sm2000_seed314159_episode47_contact_timeline.png)可以直观解释差异。
2. **物理状态反证评分器**：箱中心 0.924 m 时仍标“触地”不可能；网格最大顶点半径 0.29545 m。只过滤中心高度 >0.33 m 的假触地后重评，保留低位拖行和真正失败；几何变更时阈值必须重推。[修正、回归与回放证据](docs/RESULTS.md)。
3. **下一项真正值得做的实验**：给双手相对箱体的接触/滑动设置独立指标和训练约束，再用**多个训练种子**、相同 GPU·h/仿真步预算、统一严格物理判据复评；箱重 3 kg、不同箱型与真实传感器闭环尚未在本仓库证明。

代码来源与资产许可边界见 [THIRD_PARTY.md](THIRD_PARTY.md)。这是**单任务工程复现**，没有声称实现 SUGAR 论文原 tracker、复现未开源视频作者的私有代码或验证 SuperMimic 论文的 1000× 数字。
