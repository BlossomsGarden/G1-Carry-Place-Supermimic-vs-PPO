"""Plot measured terminal-placement success across saved policy iterations.

The input JSON contains three series of independently simulated 128-episode
scores; every marker is an actual checkpoint, not a smoothed estimate.
"""

import argparse
import html
import json
from pathlib import Path


COLORS = {"ppo_4gpu": "#2864C5", "supermimic_1gpu": "#DC5480",
          "supermimic_4gpu": "#00A48A"}


def escape(value):
    return html.escape(str(value), quote=True)


def render(data):
    width, height = 1050, 1010
    left, right = 85, 955
    panel_height = 227
    panel_tops = (148, 405, 662)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" viewBox="0 0 {width} {height}" '
             'role="img" aria-labelledby="title desc">',
             '<title id="title">终点箱体到位率与训练轮次</title>',
             '<desc id="desc">PPO 四卡、SuperMimic 单卡续训、SuperMimic 四卡，'
             '每个检查点对两个随机种子各 64 条真实物理回放打分。</desc>',
             '<rect width="100%" height="100%" fill="#FCFDFE"/>',
             '<text x="58" y="51" fill="#192536" font-size="27" font-weight="700">'
             '终点箱体到位率 × 训练轮次</text>',
             '<text x="58" y="81" fill="#58657A" font-size="15">'
             '每点：随机种子 2027 + 314159，各 64 集；最后一帧，目标三维距离 ≤ 0.12 m'
             '</text>',
             '<text x="58" y="107" fill="#58657A" font-size="14">'
             '迭代轮数不是数据集 epoch；不同训练器每轮的真实环境交互和 GPU 成本不同。'
             '</text>']
    assert data.get("denominator") == 128
    if len(data["series"]) != 3:
        raise ValueError("Expected PPO, one-GPU continuation, and four-GPU SuperMimic")

    def px(x):
        return left + (right - left) * x / 4000

    for i, series in enumerate(data["series"]):
        top = panel_tops[i]
        graph_top, graph_bottom = top + 29, top + panel_height - 27
        color = COLORS[series["id"]]
        points = sorted(series["points"], key=lambda row: row["iteration"])
        if any(not 0 <= p["terminal_128"] <= 128 for p in points):
            raise ValueError("Invalid count in " + series["id"])

        def py(n):
            return graph_bottom - (graph_bottom - graph_top) * n / 128

        parts.append(f'<rect x="47" y="{top - 17}" width="956" height="{panel_height + 12}" '
                     'rx="15" fill="#FFFFFF" stroke="#DCE4EC"/>')
        parts.append(f'<circle cx="64" cy="{top + 1}" r="6" fill="{color}"/>')
        parts.append(f'<text x="80" y="{top + 7}" fill="#1E293B" '
                     f'font-size="17" font-weight="700">{escape(series["label"])}</text>')
        if not points:
            if series.get("status") != "running":
                raise ValueError("No measured points for " + series["id"])
            parts.append(f'<text x="520" y="{top + 106}" text-anchor="middle" '
                         'fill="#697586" font-size="18">首个第 500 轮物理评测尚未完成；不绘制虚构数据点</text>')
            continue
        last = points[-1]
        summary = f'最新 #{last["iteration"]}: {last["terminal_128"]}/128 = {last["terminal_128"] / 128:.1%}'
        parts.append(f'<text x="{right}" y="{top + 7}" text-anchor="end" '
                     f'fill="{color}" font-size="15" font-weight="700">{escape(summary)}</text>')
        for value in (0, 32, 64, 96, 128):
            y = py(value)
            parts.append(f'<path d="M {left} {y:.1f} H {right}" stroke="#E8EDF3" '
                         'stroke-width="1"/>')
            parts.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" '
                         f'fill="#6A7688" font-size="12">{value * 100 // 128}%</text>')
        threshold_y = py(127)
        parts.append(f'<path d="M {left} {threshold_y:.1f} H {right}" '
                     'stroke="#DD9E30" stroke-width="1.4" stroke-dasharray="4 5"/>')
        for tick in range(0, 4001, 500):
            x = px(tick)
            parts.append(f'<path d="M {x:.1f} {graph_top} V {graph_bottom}" '
                         'stroke="#EEF1F4" stroke-width="1"/>')
            parts.append(f'<text x="{x:.1f}" y="{graph_bottom + 20:.1f}" '
                         f'text-anchor="middle" fill="#6A7688" font-size="11">{tick:,}</text>')
        path = " ".join(("M" if j == 0 else "L") + f' {px(p["iteration"]):.1f} '
                        f'{py(p["terminal_128"]):.1f}' for j, p in enumerate(points))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.8" '
                     'stroke-linejoin="round" stroke-linecap="round"/>')
        for p in points:
            x, y = px(p["iteration"]), py(p["terminal_128"])
            tooltip = (f'第 {p["iteration"]} 轮首测：{p["terminal_128"]}/128 = '
                       f'{p["terminal_128"] / 128:.1%}')
            if p.get("same_gpu_confirm_128") is not None:
                tooltip += f'；同卡复测：{p["same_gpu_confirm_128"]}/128'
                tooltip += f'；第三种子：{p["third_seed_64"]}/64'
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#FFFFFF" '
                         f'stroke="{color}" stroke-width="2.4"><title>'
                         f'{escape(tooltip)}</title></circle>')
    parts.extend(['<text x="520" y="951" text-anchor="middle" fill="#334155" '
                  'font-size="16" font-weight="600">训练迭代轮次（iteration）</text>',
                  '<text x="520" y="978" text-anchor="middle" fill="#758296" '
                  'font-size="12">虚线 = PPO 历史表格门槛 127/128；样本 128/128 不代表总体真实成功率 100%</text>',
                  '</svg>'])
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
