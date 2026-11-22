# 🚀 复现实验与核对指标

以下命令在仓库根目录、Linux shell 中运行。先按 [环境说明](ENVIRONMENT.md) 配置**宿主驱动 + Python 包**。主任务固定为 `G1-SMTrack-CarryPlace`；训练默认读取随仓库提供的 [v2 参考](../dataset/tracking/carry_place_refs/carry_place_sugar_098_refined_v2.npz) 和 [凸箱体网格](../assets/sugar_small_box_convex.obj)，两方法始终共享场景、50 Hz 控制、0.005 s 物理步长、随机化和训练示范。只有 1 条、500 帧训练参考。

## 先验收数据、评测器和脚本

~~~bash
python scripts/verify_report.py            # 只读归档：验证四组逐 episode / 日志 / PPO 历史
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

## 同一物理评测 + 保存逐帧状态

本初版 Git 仓库存代码、1 条训练参考、实验日志/评分 JSON 和演示素材；**训练好的 checkpoint 原件仍在实验机的历史 run 目录，不在初始提交中**。训练完找到实际 run 名，再按校正后评分器分别评估。三个验证种子中 2027/314159 用于两方法交集比较；161803 只在 SuperMimic 四组最终表中出现。

~~~bash
SM_RUN=$(find logs/smtrack/G1-SMTrack-CarryPlace/supermimic -maxdepth 1 -type d -name '*sm4096_h32_ori12_s0' | sort | tail -1)
PPO_RUN=$(find logs/smtrack/G1-SMTrack-CarryPlace/ppo -maxdepth 1 -type d -name '*cp_ppo_v2_s0_4g' | sort | tail -1)
test -f "$SM_RUN/model_1000.pt" && test -f "$PPO_RUN/model_3999.pt"
mkdir -p runs/evaluations
for seed in 2027 314159; do
  CUDA_VISIBLE_DEVICES=6 python src/mjlab_g1/scripts/eval_track.py \
    --task G1-SMTrack-CarryPlace --learner supermimic --checkpoint "$SM_RUN/model_1000.pt" \
    --episodes 64 --seed "$seed" --randomize \
    --rollout-all-npz "runs/evaluations/sm_$seed.npz" --output "runs/evaluations/sm_$seed.json"
  python scripts/score_carry_object_outcome.py --rollout "runs/evaluations/sm_$seed.npz" \
    --eval-json "runs/evaluations/sm_$seed.json" \
    --motion dataset/tracking/carry_place_refs/carry_place_sugar_098_refined_v2.npz \
    --output "runs/evaluations/sm_${seed}_object.json"
  CUDA_VISIBLE_DEVICES=6 python src/mjlab_g1/scripts/eval_track.py \
    --task G1-SMTrack-CarryPlace --learner ppo --checkpoint "$PPO_RUN/model_3999.pt" \
    --episodes 64 --seed "$seed" --randomize \
    --rollout-all-npz "runs/evaluations/ppo_$seed.npz" --output "runs/evaluations/ppo_$seed.json"
  python scripts/score_carry_object_outcome.py --rollout "runs/evaluations/ppo_$seed.npz" \
    --eval-json "runs/evaluations/ppo_$seed.json" \
    --motion dataset/tracking/carry_place_refs/carry_place_sugar_098_refined_v2.npz \
    --output "runs/evaluations/ppo_${seed}_object.json"
done
~~~

批量评测在一张独立空闲卡上跑，每条起点仍是第 0 帧，打开 DR/观测/重置噪声并用确定性动作。`eval_track.py` 报告严格成功并保存原始物理轨迹；`score_carry_object_outcome.py` 把相同轨迹再按物体完成/间歇持物与几何修正严格指标计分。历史 [逐回合样本与完整数据](metrics/) 使用一次次真实回放；不同 GPU 重跑可能出现少量不同接触轨迹，不要求逐字节一致。评测耗时、显存和保存大规模回放的磁盘成本**不计入训练表**。

只复核首页的一个独立物理片段，不需要训练 checkpoint：

~~~bash
PYTHONPATH=src python scripts/score_carry_object_outcome.py \
  --rollout assets/sm4096_h32_ori12_iter1000_seed161803_episode0_physical.npz \
  --output runs/evaluations/sm_single_rescored.json
~~~

该片段输出应为 `intermittent_carry_rate=1`、`strict_success_after_floor_guard_rate=0`、连续离地净搬运约 1.526 m。[视频、GIF、接触时序、如何重新导出](DEMOS.md)。
