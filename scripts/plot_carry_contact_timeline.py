"""Plot a real policy rollout's box lift and measured left/right/floor contacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rollout-npz", type=Path, required=True)
    p.add_argument("--eval-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--begin", type=int, default=140)
    p.add_argument("--end", type=int, default=230)
    args = p.parse_args()
    report = json.loads(args.eval_json.read_text(encoding="utf-8"))
    with np.load(args.rollout_npz, allow_pickle=False) as source:
        d = {key: source[key].copy() for key in source.files}
    i = int(d["episode_index"])
    trial = report["per_episode"][i]
    if (str(d["checkpoint"]) != report["checkpoint"]
            or int(d["seed"]) != report["seed"]
            or len(d["state"]) != trial["episode_steps"] + 1
            or args.begin < 0 or not args.begin < args.end < len(d["state"])):
        raise ValueError("rollout and physical evaluation do not match")
    both = np.flatnonzero(d["both_hands_contact"])
    first_both = int(both[0]) if len(both) else None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np.arange(args.begin, args.end + 1)
    time = frames / 50
    rise = d["state"][:, 73] - d["state"][0, 73]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1], "hspace": .06})
    top.plot(time, rise[frames], color="#2b83ba", linewidth=2, label="Actual box rise")
    top.axhline(.15, color="#666666", linestyle="--", linewidth=1,
                label="First lift threshold: 0.15 m")
    first_lift = trial.get("first_lift_frame")
    if first_lift is None:
        ids = np.flatnonzero(rise > .15)
        first_lift = int(ids[0]) if len(ids) else None
    if first_lift is not None:
        top.axvline(first_lift / 50, color="#d7191c", linestyle=":", linewidth=1.3)
        top.text(first_lift / 50 + .03, .18, f"first lift: {first_lift / 50:.2f}s", color="#b2182b")
    top.set_ylabel("Box rise above start (m)")
    top.legend(loc="upper left")
    top.set_title(f"{report['learner'].upper()} {Path(report['checkpoint']).stem} | "
                  f"randomized trial {i} | measured contacts")
    traces = (("right_hand_contact", "Right hand", 2, "#fdae61"),
              ("left_hand_contact", "Left hand", 1, "#2c7bb6"),
              ("floor_contact", "Box-floor", 0, "#4d4d4d"))
    for key, name, row, color in traces:
        yes = np.asarray(d[key][frames], dtype=bool)
        bottom.scatter(time[yes], np.full(yes.sum(), row), marker="s", s=17,
                       color=color, linewidth=0, label=name)
    bottom.set_yticks([0, 1, 2], ["Box-floor", "Left hand", "Right hand"])
    bottom.set_ylim(-.5, 2.5)
    bottom.set_xlim(args.begin / 50, args.end / 50)
    bottom.set_xlabel("Actual simulated time (s)")
    bottom.grid(axis="x", alpha=.25)
    if first_lift is not None and first_both is not None and first_both > first_lift:
        caption = "Bilateral contact starts after first lift; pre-lift grasp criterion fails."
    elif trial["grasp"] and not trial["carry"]:
        caption = "Bilateral grasp precedes lift, but contact interrupts during transport; strict carry fails."
    else:
        caption = "Actual box motion and MuJoCo contacts; final stage gates use the physical evaluation."
    fig.text(.5, .045, caption, ha="center", fontsize=9)
    fig.subplots_adjust(left=.14, right=.97, top=.89, bottom=.24, hspace=.07)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(f"Wrote {args.output}: first lift frame {first_lift}, "
          f"first bilateral contact {first_both}")


if __name__ == "__main__":
    main()
