# 一条演示，两个训练器

本实验实际训练只用 SUGAR CarryBox 的 `data_098`：**1 条、500 帧、50 Hz、10 秒**。机器人是 29 自由度 G1；约第 152 帧进入持箱，约第 459 帧放手，箱体参考位移约 1.53 m。原数据包有 100 条、49,043 帧；87 条、43,159 帧被成功转换，**其余 86 条转换结果没有进入本文训练**。不要把 4,096 个并行环境或几亿次仿真交互误称为演示条数。

| 文件 | 角色 |
|---|---|
| [原始转换](tracking/carry_place_refs/carry_place_sugar_098.npz) | SUGAR 机器人/物体/标注重排到 mjlab G1；保留原转换便于审计 |
| [手箱接触 v1](tracking/carry_place_refs/carry_place_sugar_098_refined_v1.npz) | 对双臂做有限位 signed-distance IK 的历史中间产物 |
| [训练参考 v2](tracking/carry_place_refs/carry_place_sugar_098_refined_v2.npz) | 小幅平滑盆骨高度以贴合地面，重新优化双臂、重算速度；两训练器均使用 |

`npz` 字段包括 `joint_pos/vel [500,29]`、`body_pos_w [500,30,3]`、`body_quat_w [500,30,4]`、`box_pos_w [500,3]`、`box_quat_w [500,4]`、物体速度、逐帧 `contact_label [500]`、`grasp_frame`、`release_frame` 与目标箱体位姿。姿态四元数为 wxyz；单位为米、秒、弧度。文件是**运动和箱体的参考轨迹**，不包含相机图像、力传感器数据，也不代表真正受力可执行的演示动作。

碰撞几何审计（指定姿态硬摆，不是策略 rollout）：原始参考接近阶段“至少一只脚接地”23.0%、双手在持物区间均与箱体相交 86.0%、手箱穿透 p95 为 9.51 cm；v2 分别为 **98.7%、97.4%、0.21 cm**。修复运动学几何后，仍须在真实仿真中训练并检验接触和稳定性。[审计输入与结果](../docs/metrics/carry098_refined_v2.json)。

可用 `src/mjlab_g1/scripts/convert_sugar_carry.py` 将官方 processed `data_098` 的 `robot_50hz.npz`、`obj_motion_global_50hz.pkl`、`contact_labels_50hz.npy` 转成第一份 NPZ；随后以 `refine_carry_contact.py` 从原始转换生成 v2。原始 100 条 processed 数据未镜像到此仓库，SUGAR 原始数据请从其项目渠道取得并遵守其发布条件；这里保存的是**本实验用的单条转换参考**。碰撞网格 [sugar_small_box_convex.obj](../assets/sugar_small_box_convex.obj) 从相应物体 USD 提取，参数见[网格元数据](../assets/sugar_small_box_convex.json)。

训练中的随机化物理状态、世界模型 replay、模型想象起点和 PPO rollout 都是**训练时生成的仿真数据**；SuperMimic 的 4,096 个 imagined starts/轮不等于新收集的 4,096 帧示范。
