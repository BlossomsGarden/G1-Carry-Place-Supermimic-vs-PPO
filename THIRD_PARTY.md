# 来源与边界

- 任务和学习器代码以开源 [humanoid-mimic](https://github.com/shuayyy/humanoid-mimic) 为基础，在本项目里加入 SUGAR→mjlab CarryPlace 参考、接触/足底修正、逐阶段真实物理评测、四组单卡探索与审计脚本。`src/mjlab_g1` 是这一实现路线，**并非 SuperMimic 论文作者官方实现的完全复刻**；本仓库 `LICENSE` 为 Apache-2.0，相关代码请同时遵循上游声明。
- `rsl_rl/` 是来源项目携带的 RSL-RL 5.2.0 实现，`rsl_rl/pyproject.toml` 声明 BSD-3-Clause ([full license](rsl_rl/LICENSE))。G1 STL/XML 资产源自原项目的 Unitree G1 资产；第三方资产的授权以各自上游发布条件为准，不应因为本仓库有 Apache-2.0 就视作全部资产自动同一许可。
- 示范源自 [SUGAR](https://github.com/tianshuwu/SUGAR) processed CarryBox，**本仓库只包含所选 data_098 的转换参考**与衍生碰撞网格；其余源数据没有在此再发布。请遵循原项目的数据使用条款。
- 论文参考：[SuperMimic](https://arxiv.org/abs/2610.28480)。本文是**单任务、单条示范、单训练种子的工程对照**，没有复现论文跨数百小时运动数据集的训练设置、基线硬件或 1000× GPU-hours 结论，也没有移植评论区视频作者未开源的 tracker/refiner 实现。
