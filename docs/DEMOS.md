# 🎬 Demo 导览：每段画面有什么、没有什么

| 片段 | 来源及分母 | 应该看见什么 | 与评测的关系 |
|---|---|---|---|
| [轨迹参考 GIF](../assets/demo_reference.gif) · [MP4](../assets/carry_place_reference_098.mp4) · [关键帧](../assets/carry_place_reference_098_frames.png) | SUGAR `data_098`，500 帧固定参考 | 走近、取箱、携箱移动、放到绿色目标旁 | **运动学逐帧指定的机器人/箱体位姿**，不是训练策略，也不是动力学成功证据 |
| [PPO 成功 GIF](../assets/demo_ppo_strict.gif) · [MP4](../assets/ppo3999_seed314159_episode59_physical.mp4) · [接触时序](../assets/ppo3999_seed314159_episode59_contact_timeline.png) | PPO 终点、314159 号种子**重录批次的 episode 59** | 先双手抓牢，再连续举着走，放开后箱体落地 | 该条严格成功；重录批次严格成功 30/64，不能并入首页另两批 128 集。[该批 64 集](metrics/ppo3999_seed314159_record64.json) |
| [SuperMimic 搬放 GIF](../assets/demo_sm_intermit.gif) · [MP4](../assets/sm4096_h32_ori12_iter1000_seed161803_episode0_physical.mp4) · [地面异常时序](../assets/sm4096_h32_ori12_iter1000_seed161803_episode0_floor_audit.png) | H32 / 1.2、种子 161803 的 episode 0 | 实际抱起、空中移动约 1.526 m、抵达并落地 | **间歇持物成功、严格连续双手失败**；画面上 `STRICT FAILURE` 角标是后一个指标 |
| [四卡 SuperMimic 严格成功 GIF](../assets/demo_sm4gpu_strict.gif) · [MP4](../assets/sm4gpu1500_seed161803_episode10_physical.mp4) · [六帧关键画面](../assets/sm4gpu1500_seed161803_episode10_physical.png) | 第 1,500 轮权重，第三种子 161803 的 episode 10 | 双手举起、空中净搬运约 1.540 m、到位放下，终点距离 0.027 m | **该集严格成功**；主双种子严格 7/128，第三种子严格 2/64。可从[单集逐帧状态](../assets/sm4gpu1500_seed161803_episode10_physical.npz)和[同批原始 64 集评分](metrics/learning_curve/four/iter1500_seed161803_heldout_eval.json)核对 |
| [SuperMimic 失败 GIF](../assets/demo_sm_contact_failure.gif) · [MP4](../assets/sm2000_seed314159_episode47_physical.mp4) · [接触时序](../assets/sm2000_seed314159_episode47_contact_timeline.png) | 旧两卡终点、种子 314159 的 episode 47 | 箱子看似完成搬运，但双手接触期间断 | 离地净位移 1.513 m；连续双手搬运仅 0.151 m，**严格失败** |

所有策略视频都从**真实 MuJoCo 策略控制生成的逐帧状态**渲染；运动学参考单列、标题明确标注，绝不把“照着示范直接摆姿态”称为策略成功。GIF 以 8 fps、320 px 宽从对应 MP4 创建，供 GitHub 首页直接循环预览；想看接触细节请打开 MP4/时序图。单条示例不是批量成功率估计，[完整训练与物理结果](RESULTS.md) 按分母单独报告。

GIF 的生成配方（从仓库的 MP4 重新生成，依赖 ffmpeg）：

~~~bash
ffmpeg -i assets/ppo3999_seed314159_episode59_physical.mp4 \
  -filter_complex '[0:v]fps=8,scale=320:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3' \
  -y assets/demo_ppo_strict.gif
~~~

已归档的四卡严格成功片段可直接从本仓库的**同一批真实物理状态**重渲染，不需要重新训练或重新评测：

~~~bash
MUJOCO_GL=egl python src/mjlab_g1/scripts/render_carry_policy_rollout.py \
  --rollout-npz assets/sm4gpu1500_seed161803_episode10_physical.npz \
  --eval-json docs/metrics/learning_curve/four/iter1500_seed161803_heldout_eval.json \
  --g1-xml /path/to/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml \
  --box-mesh assets/sugar_small_box_convex.obj \
  --output assets/sm4gpu1500_seed161803_episode10_physical.mp4 \
  --contact-sheet assets/sm4gpu1500_seed161803_episode10_physical.png
~~~

服务器的 conda 环境没有系统 `ffmpeg` 命令；渲染脚本自动使用 `imageio_ffmpeg` 附带的二进制，GIF 则调用相同二进制和上面的滤镜。重渲染录制的物理状态只核对视频来源，不等于再跑一次策略成功率。

产生**新的**物理策略视频时，请先用 [评测命令](REPRODUCE.md) 输出成对的 `--output eval.json` 和 `--rollout-npz one_episode.npz`；之后执行 `src/mjlab_g1/scripts/render_carry_policy_rollout.py`，它核对权重/种子/episode/帧数/原始成功标记后渲染。`--g1-xml` 指向安装的 mjlab `asset_zoo/robots/unitree_g1/xmls/g1.xml`，`--box-mesh` 指向本仓库箱体网格。**不能**拿不同次随机回放的 JSON 去配另一次 NPZ，接触动力学会放大数值细微差异。
