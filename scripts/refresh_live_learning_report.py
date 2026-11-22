"""Keep an honestly provisional report while long GPU runs continue.

Only real 128-episode monitor results enter the README's four-GPU row. The
other two historical comparison rows are left intact. Run on an archive
snapshot after validating it with build_terminal_success_curve_data.py.
"""

import argparse
import datetime as dt
import json
import re
from pathlib import Path


def fmt_count(n):
    return f"{n}/128 = {n / 128:.1%}"


def status_label(status):
    if status == "running":
        return "训练中"
    if status.startswith("target corroborated at iteration"):
        return "复测达标，已停训"
    return status


def single_gpu_row(archive, costs):
    status = json.loads((archive / "single/monitor_state.json").read_text(encoding="utf-8"))
    eligible = [point for point in status["results"]
                if point.get("confirm_terminal_128", 0) >= 127]
    if not eligible:
        return None
    best = max(eligible, key=lambda x: (x["confirm_terminal_128"],
                                        x["intermittent_128"], x["iteration"]))
    run = costs["runs"]["single"]
    manifest = json.loads((archive / "single/checkpoint_manifest.json").read_text(encoding="utf-8"))
    checkpoint = next(x for x in manifest if x["iteration"] == best["iteration"])
    start = dt.datetime.fromisoformat(run["start_utc"]).timestamp()
    selected_h = 13503 / 3600 + (checkpoint["mtime_unix"] - start) / 3600
    full = (f"{'≤' if run['wall_elapsed_is_upper_bound'] else ''}"
            f"{13521 / 3600 + run['wall_elapsed_s'] / 3600:.2f} h"
            if run["training_terminal"] else "续训中")
    log = (archive / "single" / (run["run_name"] + ".log")).read_text(errors="replace")
    match = re.search(r"\[SuperMimic\] it " + str(best["iteration"]) +
                      r" \| env_steps (\d+)", log)
    if not match:
        raise ValueError("Selected single-GPU checkpoint is missing its environment steps")
    steps = int(match.group(1))
    speed = run["steady_real_env_steps_per_s_last_100plus_iters"]
    memory = max(11754, run["gpu_peak_used_mib"]["6"])
    return (f"| **SuperMimic 单卡续训 H32 / std 1.2，第 {best['iteration']:,} 轮（复测通过）** "
            f"| 1 × RTX 4090 / 4,096 "
            f"| **{selected_h:.2f} h / {full}** "
            f"| {steps / 10000:,.1f} 万 "
            f"| {speed:,.0f} "
            f"| 6：{memory:,} MiB "
            f"| **{fmt_count(best['terminal_128'])}**（同卡复测 {fmt_count(best['confirm_terminal_128'])}） "
            f"| {fmt_count(best['intermittent_128'])} "
            f"| {fmt_count(best['strict_128'])} |")


def four_gpu_row(archive, costs):
    status = json.loads((archive / "four/monitor_state.json").read_text(encoding="utf-8"))
    points = status["results"]
    if not points:
        return None
    best = max(points, key=lambda x: (x["terminal_128"], x["intermittent_128"],
                                      x["strict_128"], x["iteration"]))
    run = costs["runs"]["four"]
    manifest = json.loads((archive / "four/checkpoint_manifest.json").read_text(encoding="utf-8"))
    checkpoint = next(x for x in manifest if x["iteration"] == best["iteration"])
    started = dt.datetime.fromisoformat(run["start_utc"]).timestamp()
    selected_h = (checkpoint["mtime_unix"] - started) / 3600
    full = (f"{'≤' if run['wall_elapsed_is_upper_bound'] else ''}"
            f"{run['wall_elapsed_s'] / 3600:.2f} h"
            if run["training_terminal"] else "训练中")
    log = (archive / "four" / (run["run_name"] + ".log")).read_text(errors="replace")
    match = re.search(r"\[SuperMimic\] it " + str(best["iteration"]) +
                      r" \| env_steps (\d+)", log)
    if not match:
        raise ValueError("Selected four-GPU checkpoint is missing its rank-zero environment steps")
    global_steps = int(match.group(1)) * 4
    speed = run["steady_real_env_steps_per_s_last_100plus_iters"]
    memories = run["gpu_peak_used_mib"]
    gpu_text = "；".join(f"{device}：{memories[str(device)]:,}" for device in (1, 4, 5, 7)) + " MiB"
    tag = status_label(status["status"])
    outcome = f"**{fmt_count(best['terminal_128'])}**"
    if best.get("confirm_terminal_128") is not None:
        outcome += f"（同卡复测 {fmt_count(best['confirm_terminal_128'])}）"
    row = (f"| **SuperMimic 四卡 H32 / std 1.2，第 {best['iteration']:,} 轮（{tag}）** "
           f"| 4 × RTX 4090 / 8,192（每卡 2,048） "
           f"| **{selected_h:.2f} h / {full}** "
           f"| {global_steps / 10000:,.1f} 万 "
           f"| {speed:,.0f} "
           f"| {gpu_text} "
           f"| {outcome} "
           f"| {fmt_count(best['intermittent_128'])} "
           f"| {fmt_count(best['strict_128'])} |")
    return row, best, selected_h, global_steps


def refresh(readme, archive, costs):
    original = readme.read_text(encoding="utf-8")
    single = single_gpu_row(archive, costs)
    if single is not None:
        pattern = (r"(?m)^\| \*\*(?:原四组配方最佳 SuperMimic|"
                   r"SuperMimic 单卡续训 H32 / std 1\.2)[^\n]*\n")
        edited, count = re.subn(pattern, single + "\n", original)
        if count != 1:
            raise ValueError(f"Expected one single-GPU row, found {count}")
        original = edited
    current = four_gpu_row(archive, costs)
    if current is not None:
        row, *_ = current
        pattern = r"(?m)^\| \*\*SuperMimic 四卡 H32 / std 1\.2[^\n]*\n"
        edited, count = re.subn(pattern, row + "\n", original)
        if count != 1:
            raise ValueError(f"Expected one four-GPU row, found {count}")
        original = edited
    if original != readme.read_text(encoding="utf-8"):
        readme.write_text(original, encoding="utf-8")

    ppo = json.loads((archive / "ppo/curve.json").read_text(encoding="utf-8"))
    single = json.loads((archive / "single/monitor_state.json").read_text(encoding="utf-8"))
    four = json.loads((archive / "four/monitor_state.json").read_text(encoding="utf-8"))
    snapshot_note = ("运行中快照" if four["status"] == "running" else "归档快照")
    out = ["# 学习曲线评测账本", "", f"> 此页是{snapshot_note}；README 的历史行和续训新评测属于不同回放批次。",
           "> 首测达到阈值还要同卡复测；128/128 不证明总体成功率 100%。", "",
           "| 方法 | 当前状态 | 最新实测检查点 | 最高首测（附复测） | 训练用时至当前 | 当前训练峰值显存（各卡 MiB） |",
           "|---|---|---|---|---:|---|"]
    for name, state, points in (("PPO 四卡", "已结束", ppo["points"]),
                                ("SuperMimic 单卡续训", single["status"], single["results"]),
                                ("SuperMimic 四卡", four["status"], four["results"])):
        run = costs["runs"].get("four" if "四卡" in name and "SuperMimic" in name else "single") if "SuperMimic" in name else None
        latest = f"第 {points[-1]['iteration']:,} 轮 {fmt_count(points[-1]['terminal_128'])}" if points else "尚无新检查点"
        candidates = list(points)
        if name == "SuperMimic 单卡续训":
            candidates.append(costs["runs"]["single"]["original_checkpoint_baseline"])
        best = max(candidates, key=lambda x: (x["terminal_128"], x["intermittent_128"],
                                                x["strict_128"], x["iteration"]), default=None)
        best_str = f"第 {best['iteration']:,} 轮 {fmt_count(best['terminal_128'])}" if best else "尚无实测点"
        if best is not None and best.get("confirm_terminal_128") is not None:
            best_str += f"；同卡复测 {fmt_count(best['confirm_terminal_128'])}"
        elapsed = (f"原 3.76 h + 续训 {run['wall_elapsed_s'] / 3600:.2f} h"
                   if name == "SuperMimic 单卡续训" else
                   f"{run['wall_elapsed_s'] / 3600:.2f} h" if run else "2.58 h")
        peak = ("；".join(f"{k}: {v:,}" for k,v in run["gpu_peak_used_mib"].items())
                if run else "0: 6,288；1/4/5: 4,922")
        label = status_label(state)
        out.append(f"| {name} | {label} | {latest} | {best_str} | {elapsed} | {peak} |")
    out += ["", "PPO 末轮本次独立物理复评：" + fmt_count(ppo["points"][-1]["terminal_128"]) + "。",
            "单卡原始第 1,000 轮在评测 GPU 3 上重测：125/128；续训监控不改变这个原始权重。",
            "四卡从第 500 轮起每 500 轮评测两种子各 64 集，达到 128/128 后同卡复测并检查第三种子。", "",
            "| 方法 | 更新轮次 | 首测到位 /128 | 同卡复测到位 /128 | 第三种子到位 /64 | 间歇持物 /128 | 连续双手 /128 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name, points in (("PPO", ppo["points"]), ("单卡 SuperMimic", single["results"]),
                         ("四卡 SuperMimic", four["results"])):
        for p in points:
            corroborated = p.get("confirm_terminal_128", "—")
            heldout = p.get("heldout", {}).get("terminal", "—")
            out.append(f"| {name} | {p['iteration']:,} | {p['terminal_128']} "
                       f"| {corroborated} | {heldout} | {p['intermittent_128']} | {p['strict_128']} |")
    out += ["", "[逐集评测和训练日志](./) · [三曲线底层数据](../learning_curves_20261122.json) · "
            "[训练成本两秒采样](run_costs.json)", ""]
    output = archive / "LIVE_REPORT.md"
    output.write_text("\n".join(out), encoding="utf-8")
    print("live_report", output, "four_row", "updated" if current else "waiting for first physical test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args()
    costs = json.loads((args.archive / "run_costs.json").read_text(encoding="utf-8"))
    refresh(args.readme, args.archive, costs)


if __name__ == "__main__":
    main()
