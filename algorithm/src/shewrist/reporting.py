"""Optional Matplotlib report for one auditable offline session."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


_ACTIVITY_COLORS = {
    "background_or_rejected": "#D7DEE5",
    "extension": "#2F6F9F",
    "flexion": "#318B82",
    "radial_deviation": "#C98624",
    "ulnar_deviation": "#6D5A9B",
}


def plot_session_report(
    rows: Sequence[Mapping[str, object]],
    analysis: Mapping[str, object],
    output_stem: str | Path,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required for session report charts") from exc
    plt.rcParams.update(
        {
            "font.sans-serif": ["Hiragino Sans GB", "PingFang HK", "Arial Unicode MS", "Heiti TC", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    if not rows:
        raise ValueError("timeline rows are empty")
    output = Path(output_stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    t = np.array([float(row["timestamp_ms"]) for row in rows]) / 1000.0
    t = t - t[0]
    fe = np.array([float(row["theta_FE"]) for row in rows])
    rud = np.array([float(row["theta_RUD"]) for row in rows])
    quality = np.array([float(row["quality"]) for row in rows])
    alerts = np.array([bool(row["alert"]) for row in rows])
    stops = np.array([bool(row["safety_stop"]) for row in rows])
    activities = np.array([str(row["activity_shadow"]) for row in rows], dtype=object)
    metrics = analysis["deterministic_control"]["metrics"]
    replay = analysis.get("replay", {})
    explanation = analysis.get("explanation", {})
    fig = plt.figure(figsize=(15, 10), dpi=180, facecolor="#F5F7FA")
    grid = fig.add_gridspec(5, 1, height_ratios=(2.0, 1.2, 0.9, 0.9, 1.1), hspace=0.32)
    axes = [fig.add_subplot(grid[index, 0]) for index in range(4)]
    summary_ax = fig.add_subplot(grid[4, 0])
    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(axis="y", color="#E7EBEF", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(float(t[0]), float(t[-1]))
    axes[0].plot(t, fe, color="#2F6F9F", linewidth=0.9, label="FE")
    axes[0].plot(t, rud, color="#318B82", linewidth=0.9, label="RUD")
    axes[0].axhline(15.0, color="#C98624", linestyle="--", linewidth=0.8)
    axes[0].axhline(-15.0, color="#C98624", linestyle="--", linewidth=0.8)
    axes[0].axhline(30.0, color="#B9504B", linestyle=":", linewidth=0.8)
    axes[0].axhline(-30.0, color="#B9504B", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("角度 (°)")
    axes[0].set_title("腕角时间轴（阈值仅为工程分层）", loc="left", fontweight="bold")
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[1].plot(t, quality, color="#33865A", linewidth=0.9)
    axes[1].fill_between(t, 0.0, quality, color="#E8F4EC")
    axes[1].axhline(0.2, color="#B9504B", linestyle="--", linewidth=0.9, label="确定性有效线")
    axes[1].axhline(0.5, color="#6D5A9B", linestyle=":", linewidth=0.9, label="ML 窗口门槛")
    axes[1].set_ylim(-0.03, 1.05)
    axes[1].set_ylabel("quality")
    axes[1].set_title("质量与拒识", loc="left", fontweight="bold")
    axes[1].legend(loc="lower right", frameon=False, ncol=2, fontsize=8)
    categories = list(dict.fromkeys(activities.tolist()))
    for row_index, activity in enumerate(categories):
        mask = activities == activity
        axes[2].fill_between(t, row_index - 0.35, row_index + 0.35, where=mask, step="mid", color=_ACTIVITY_COLORS.get(activity, "#657684"), alpha=0.95)
    axes[2].set_yticks(range(len(categories)), [value.replace("_", " ") for value in categories])
    axes[2].set_ylim(-0.7, max(0.7, len(categories) - 0.3))
    axes[2].set_title("CNN-HMM 影子动作（无控制权限）", loc="left", fontweight="bold")
    axes[3].scatter(t[alerts], np.ones(np.count_nonzero(alerts)), color="#C98624", marker="^", s=28, label="提醒")
    axes[3].scatter(t[stops], np.full(np.count_nonzero(stops), 0.45), color="#B9504B", marker="s", s=24, label="安全停止状态")
    axes[3].set_ylim(0.0, 1.35)
    axes[3].set_yticks([])
    axes[3].set_xlabel("会话时间 (s)")
    axes[3].set_title("确定性提醒与压力停止", loc="left", fontweight="bold")
    axes[3].legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    summary_ax.axis("off")
    provider = explanation.get("provider", "disabled") if isinstance(explanation, Mapping) else "disabled"
    api_called = explanation.get("api_called", False) if isinstance(explanation, Mapping) else False
    summary = (
        f"会话 {analysis.get('session_id')}   证据 {analysis.get('evidence_type')}   "
        f"有效样本 {float(metrics.get('valid_sample_pct', 0.0)):.1f}%   "
        f"P_high {float(metrics.get('P_high_pct') or 0.0):.1f}%   "
        f"提醒 {int(metrics.get('alert_count', 0))}   安全停止 {int(metrics.get('safety_stop_count', 0))}\n"
        f"分块 {replay.get('chunk_count', '?')} × {replay.get('chunk_size_samples', '?')} samples   "
        f"批/流状态一致 {replay.get('deterministic_state_equal')}   最终输出一致 {replay.get('final_analysis_equal')}   "
        f"解释器 {provider}   外部 API 调用 {api_called}\n"
        "结论边界：公开/模拟回放只证明软件链路；未完成目标硬件、角度真值或人体有效性验证。"
    )
    summary_ax.text(
        0.01,
        0.92,
        summary,
        va="top",
        fontsize=10,
        linespacing=1.6,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "white", "edgecolor": "#CAD4DC"},
    )
    fig.suptitle("SheWrist 纯离线算法 v0.8 会话报告", fontsize=18, fontweight="bold", x=0.06, ha="left")
    paths = []
    for suffix in ("png", "svg"):
        path = output.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths.append(path)
    plt.close(fig)
    return paths