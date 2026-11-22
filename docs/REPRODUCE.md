# 🚀 复现实验与核对指标

以下命令在仓库根目录、Linux shell 中运行。先按 [环境说明](ENVIRONMENT.md) 配置**宿主驱动 + Python 包**。主任务固定为 `G1-SMTrack-CarryPlace`；训练默认读取随仓库提供的 [v2 参考](../dataset/tracking/carry_place_refs/carry_place_sugar_098_refined_v2.npz) 和 [凸箱体网格](../assets/sugar_small_box_convex.obj)，两方法始终共享场景、50 Hz 控制、0.005 s 物理步长、随机化和训练示范。只有 1 条、500 帧训练参考。

## 先验收数据、评测器和脚本

~~~bash
python scripts/verify_report.py            # 只读归档：验证四组逐 episode / 日志 / PPO 历史
python scripts/rescore_terminal_placement.py # 重算旧物理回放的终点三维到位率（无 GPU）
export PYTHONPATH=src MUJOCO_GL=disable
python -m unittest discover -s tests -p 'test_carryplace_eval.py' -v
CUDA_VISIBLE_DEVICES=0 python src/mjlab_g1/scripts/train_track.py \
  G1-SMTrack-CarryPlace --learner ppo --smoke --no-compile
~~~

`--smoke` 只用缩小网络、8 环境、3 轮检查是否能写入 checkpoint；不能用于成功率/吞吐结论。

## 历史配方：四组单卡 SuperMimic

在**四个 shell 中并发**运行，每个进程独占指定一张物理 GPU；GPU 编号是 `nvidia-smi` 的物理编号。运行时进程只看见自己的卡，内部会重编号为 `cuda:0`。`CARRY_SAVE_INTERVAL=100` 每百轮保存一次，`--lr-cosine-iters 250000` 保持学习率日程尺度，避免因为只做 1,000 轮短试就提前把学习率降至终点。每轮真实样本和想象样本分别计账。

~~~bash
CARRY_SAVE_INTERVAL=100 bash scripts/run_carry_experiment.sh supermimic 4 4096 1000 sm4096_h32_ori04_s0 --imag-microbatches 16 --lr-cosine-iters 250000 --imagination-horizon 32 --box-ori-std 0.4 &
CARRY_SAVE_INTERVAL=100 bash scripts/run_carry_experiment.sh supermimic 5 4096 1000 sm4096_h16_ori04_s0 --imag-microbatches 16 --lr-cosine-iters 250000 --imagination-horizon 16 --box-ori-std 0.4 &
CARRY_SAVE_INTERVAL=100 bash scripts/run_carry_experiment.sh supermimic 6 4096 1000 sm4096_h32_ori12_s0 --imag-microbatches 16 --lr-cosine-iters 250000 --imagination-horizon 32 --box-ori-std 1.2 &
CARRY_SAVE_INTERVAL=100 bash scripts/run_carry_experiment.sh supermimic 7 4096 1000 sm4096_h16_ori12_s0 --imag-microbatches 16 --lr-cosine-iters 250000 --imagination-horizon 16 --box-ori-std 1.2 &
wait
~~~

`wait` 结束后，逐个确认四组 `runs/<run_name>_exit_code.txt` 均为 `0`。

这是每张卡各自独立一个 4,096 环境的实验，**不是**一次 4 卡并行训练。`--imag-microbatches 16` 将 4,096 个想象起点拆为 16×256 做梯度累积，不增加策略更新次数、不改变环境数。训练脚本输出到 `runs/<run_name>.log` 和 `runs/<run_name>_gpu_samples.csv`；实际 checkpoint 在 `logs/smtrack/G1-SMTrack-CarryPlace/supermimic/<timestamp>_<run_name>/model_*.pt`。本仓库脚本把旧服务器专用硬编码路径替换为仓库相对路径，保留上述历史实验的模型参数/关键训练 flags；**现有归档来自历史运行，不是声称新的便携脚本已经重新跑满四组**。

## PPO 四卡 baseline

使用**4 张卡各 2,048 = 全局 8,192 环境**；单机分布式的 `--master_addr=127.0.0.1` 避免机器名解析导致 rendezvous 失败，多个同时训练的分布式进程要给不同 `CARRY_MASTER_PORT`。

~~~bash
CARRY_MASTER_PORT=29571 CARRY_SAVE_INTERVAL=100 bash scripts/run_carry_experiment.sh \
  ppo 0,1,4,5 8192 4000 cp_ppo_v2_s0_4g
~~~

从零到第 4,000 次更新的存档文件名是 `model_3999.pt`，不要把 800 轮或更早 checkpoint 当成正式终点。PPO 在本实验是**同参考任务上的 PPO-Sim**，并非仓库另外的 AMP-PPO 双臂 Lift 任务；不应混用两个任务的成功口径。

## 2026-11-22：单卡续训与四卡 SuperMimic

以下为正式实验配方：单卡已在第 1,300 轮复测达标并受控停训，四卡从零训练并每 500 轮评测；结果须使用相同的物理回放逐检查点核对，不能把训练 reward 当成终点到位率。单卡从原 `model_1000.pt` 恢复（恢复时重新预填充 replay）；四卡和上面的 PPO 使用相同的**全局 8,192 环境，每卡 2,048**。两路各使用一个进程组和独立日志、端口；并发时的墙钟及整卡显存峰值包含共享服务器资源竞争。

~~~bash
# 单卡：cuda:6，保存第 1100、1200、…、3000 轮；续训不从零计数
CARRY_SAVE_INTERVAL=100 bash scripts/run_carry_experiment.sh \
  supermimic 6 4096 3000 smcont_h32_ori12_from1000_s0_20261122b \
  --resume-checkpoint /path/to/original/model_1000.pt \
  --imag-microbatches 16 --lr-cosine-iters 250000 \
  --imagination-horizon 32 --box-ori-std 1.2

# 四卡：cuda:1,4,5,7，每卡 2,048 环境；最多 4000 轮、每 500 轮存档并检查
CARRY_MASTER_PORT=29641 CARRY_SAVE_INTERVAL=500 bash scripts/run_carry_experiment.sh \
  supermimic 1,4,5,7 8192 4000 sm4g8192_h32_ori12_mb4_s0_20261122 \
  --imag-microbatches 4 --lr-cosine-iters 250000 \
  --imagination-horizon 32 --box-ori-std 1.2
~~~

实际四卡运行在第 500 / 1,000 / 1,500 轮各评估两种子 128 集，到位分别为 **42 / 124 / 128**；第 1,500 轮同卡复测 **128/128**、第三种子 **64/64**，触发监控器 SIGINT。四张训练卡确认回到空闲，启动器退出码 **1** 是有意中断分布式训练的结果，**不是**训满 4,000 轮。选中权重与配置见 [checkpoint 目录](../checkpoints/README.md)，训练耗时及 2 秒采样显存见 [运行账本](metrics/learning_curve/run_costs.json)。

四卡策略更新仍是**全局 4,096 个想象起点**：每卡 1,024 起点、每卡 4 块 × 256，梯度按同一全局总数归一化再同步；不是每卡 4,096 起点。先前每卡 16 块 × 64 的调速短试未保存正式检查点，不纳入曲线。评测每点随机种子 2027 / 314159 各 64 集；单卡每 100 轮、四卡每 500 轮评测。首次到达单卡 ≥127/128 或四卡 128/128 须在相同评测卡重测两种子并补测 161803，再决定是否停止；有限样本 128/128 不代表真实成功率 100%。

监督脚本 [monitor_sm_continuation.py](../scripts/monitor_sm_continuation.py) 独立于训练进程：接收训练 PID、运行目录、评测 GPU、首末轮次和保存间隔；每次物理回放保存 64 集原始评测、评分 JSON 与监控状态。单卡监督参数为 **--first 1100 --last 3000 --interval 100 --original-iteration 1000 --eval-gpu 3 --threshold-total 127**；四卡为 **--first 500 --last 4000 --interval 500 --original-iteration 0 --eval-gpu 2 --threshold-total 128**。两路评测使用文件锁错开固定输出文件名，达到阈值后同卡复测并给对应训练进程组发 SIGINT；四卡另有 [退出守护](../scripts/ensure_training_stops.py) 在复核达标后等 120 秒，若训练仍在则仅对核实过的训练进程组发 SIGTERM，并保留退出时间上界与真实退出码是否缺失的记录。历史 PPO 九个检查点另在 GPU 2 上重新跑了同种子、同集数。**首个单卡续训检查点第 1,100 轮到位 108/128，低于原权重重测的 125/128**，因此不能用训练 reward 选“最新”权重。

训练服务器上用 [archive_learning_curve_server.py](../scripts/archive_learning_curve_server.py) 仅打包两个运行的评分 JSON、评测/训练日志、GPU 两秒采样、参数、权重文件大小和时间戳，不将数百 MB 的每个权重塞进数据归档。本地拿到归档并解压到 docs/metrics/learning_curve/ 后，以下命令逐集核对每个标记，重建图和 GPU 成本；第三面板只绘制实际完成物理回放的第 500 / 1,000 / 1,500 轮，不会补插值。

~~~bash
python scripts/build_terminal_success_curve_data.py \
  --archive docs/metrics/learning_curve --output docs/metrics/learning_curves_20261122.json
python scripts/plot_terminal_learning_curves.py \
  --input docs/metrics/learning_curves_20261122.json --output assets/terminal_success_learning_curves.svg
python scripts/summarize_carry_run_costs.py \
  --archive docs/metrics/learning_curve --output docs/metrics/learning_curve/run_costs.json
~~~

分布式 SuperMimic 日志由 **rank 0** 打印真实步数；四卡总真实步数须乘 4，不能把单卡日志字段直接与 PPO 全局步数比较。训练 GPU·h 是训练墙钟 × 卡数；评测卡使用另计、显存为训练卡 nvidia-smi 的整卡已用量两秒采样最大值。续训 replay 重新预填充，不能把累计轮次误当成不间断的单次训练。

若要单独排查同卡同种子的重复性，在**原实验服务器**执行两遍完全相同的物理回放，并核对初始随机化、初始状态、首动作和后续物理状态（固定输出文件名前会先获取同一评测锁）：

~~~bash
python scripts/probe_eval_repeatability.py \
  --repo "$PWD" --checkpoint /path/to/resumed/model_1200.pt \
  --output runs/repeatability_1200 --gpu 3 --seed 2027 --episodes 64
~~~

[实际重复性证据与审慎解释](RESULTS.md#2026-11-22-单卡续训不能因一次-127128-就停)已归档。探针保存物理状态和随机化参数的大型 NPZ 于服务器诊断目录；仓库仅保留其小型汇总 JSON 与两份逐集评分。

## 同一物理评测 + 保存逐帧状态

[Hugging Face 模型仓库](https://huggingface.co/BlossomsGarden/G1-Carry-Place-Supermimic-vs-PPO) 提供四份权重（原单卡、续训单卡、四卡 SuperMimic、四卡 PPO），GitHub 的 [checkpoints/](../checkpoints/README.md) 保留各自的 `params/agent.json`；将权重下载到相应路径后即可评测；两个主要评测种子是 2027/314159，161803 还用于新权重的第三种子检查。以下会生成**新**随机化物理回放，和首页归档批次分开计数。

~~~bash
hf download BlossomsGarden/G1-Carry-Place-Supermimic-vs-PPO --include 'checkpoints/*/model_*.pt' --local-dir .  # 在联网机器下载；原始实验服务器离线，可传入历史权重作为第 5 个参数
export CARRY_PYTHON=/data/wlh/miniconda3/envs/wan/bin/python # 原服务器路径；其他主机可不设
for seed in 2027 314159; do
  bash scripts/eval_checkpoints.sh supermimic 6 "$seed" 64
  # SuperMimic 的固定输出名会被下一次同种子评测覆盖，先保存原权重的整批状态与评分。
  for ext in .npz .json _metrics.json; do
    cp "runs/evaluations/supermimic_${seed}_n64$ext" "runs/evaluations/sm_original_${seed}_n64$ext"
  done
  bash scripts/eval_checkpoints.sh supermimic 6 "$seed" 64 checkpoints/supermimic_4gpu/model_1500.pt
  bash scripts/eval_checkpoints.sh ppo 6 "$seed" 64
done
~~~

批量评测在空闲卡跑，每条从参考第 0 帧起步，打开 DR/观测/重置噪声并用确定性策略动作。`eval_track.py` 保存原始物理轨迹；`score_carry_object_outcome.py` 对同一轨迹输出终点三维到位、间歇持物、几何校正的连续双手三个指标。结果为 `runs/evaluations/{learner}_{seed}_n64_metrics.json`；不同 GPU 的物理接触可能略有波动，评测成本**不计入训练表**。要复核历史首页数字而不新跑 GPU，运行 `python scripts/rescore_terminal_placement.py`，其逐集输入与源哈希见 [归档](metrics/terminal_goal_3d_12cm_rescore.json)。

原实验服务器不能连接 Hugging Face 时，先在联网本机下载四份完整权重，再连同 GitHub 仓库里的 `checkpoints/*/params/agent.json` 复制到服务器本项目相应路径；若模型仍在原历史运行目录，可把 `bash scripts/eval_checkpoints.sh` 的**第 5 个参数**设为历史 checkpoint 路径，脚本从该目录的 `params/agent.json` 恢复相应网络配置。

只复核首页的一个独立物理片段，不需要训练 checkpoint：

~~~bash
PYTHONPATH=src python scripts/score_carry_object_outcome.py \
  --rollout assets/sm4096_h32_ori12_iter1000_seed161803_episode0_physical.npz \
  --output runs/evaluations/sm_single_rescored.json
~~~

该片段输出应为 `intermittent_carry_rate=1`、`strict_success_after_floor_guard_rate=0`、连续离地净搬运约 1.526 m。[视频、GIF、接触时序、如何重新导出](DEMOS.md)。
