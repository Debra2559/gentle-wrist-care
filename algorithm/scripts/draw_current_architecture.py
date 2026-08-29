#!/usr/bin/env python3
"""Draw the implemented SheWrist architecture in an academic three-stage style."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, RegularPolygon
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "project_overview"

C = {
    "ink": "#172B3A",
    "muted": "#617382",
    "line": "#AEBCC7",
    "white": "#FFFFFF",
    "stage1": "#E8A900",
    "stage1_fill": "#FFF9E8",
    "stage2": "#E77E31",
    "stage2_fill": "#FFF8F2",
    "stage3": "#3B9A78",
    "stage3_fill": "#F3FBF7",
    "blue": "#3977B8",
    "blue_fill": "#EAF3FB",
    "green": "#4A9A57",
    "green_fill": "#EAF6E9",
    "purple": "#7562A8",
    "purple_fill": "#F0ECF8",
    "red": "#C75750",
    "red_fill": "#FDEDEC",
    "gray": "#6E7478",
    "gray_fill": "#F0F2F3",
    "amber_fill": "#FFF0CF",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Hiragino Sans GB",
                "PingFang HK",
                "Arial Unicode MS",
                "Heiti TC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "text.color": C["ink"],
        }
    )


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    face: str = "white",
    edge: str = "line",
    title_size: float = 10.5,
    body_size: float = 7.8,
    linewidth: float = 1.5,
    radius: float = 0.9,
    center: bool = False,
    zorder: int = 3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.38,rounding_size={radius}",
        facecolor=C.get(face, face),
        edgecolor=C.get(edge, edge),
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if center:
        ax.text(
            x + w / 2,
            y + h * (0.62 if body else 0.5),
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            zorder=zorder + 1,
        )
        if body:
            ax.text(
                x + w / 2,
                y + h * 0.28,
                body,
                ha="center",
                va="center",
                fontsize=body_size,
                color=C["muted"],
                linespacing=1.25,
                zorder=zorder + 1,
            )
    else:
        ax.text(
            x + 1.1,
            y + h - 1.2,
            title,
            ha="left",
            va="top",
            fontsize=title_size,
            fontweight="bold",
            zorder=zorder + 1,
        )
        if body:
            ax.text(
                x + 1.1,
                y + h - 4.0,
                body,
                ha="left",
                va="top",
                fontsize=body_size,
                color=C["muted"],
                linespacing=1.3,
                zorder=zorder + 1,
            )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "ink",
    width: float = 1.5,
    style: str = "-|>",
    dashed: bool = False,
    curve: float = 0.0,
    zorder: int = 2,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=width,
        linestyle="--" if dashed else "-",
        color=C.get(color, color),
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def stage_badge(ax, x: float, y: float, number: str, color: str) -> None:
    badge = RegularPolygon(
        (x, y),
        numVertices=8,
        radius=3.4,
        orientation=np.pi / 8,
        facecolor=C.get(color, color),
        edgecolor=C["gray"],
        linewidth=1.5,
        zorder=8,
    )
    ax.add_patch(badge)
    ax.text(x, y, number, color="white", fontsize=17, ha="center", va="center", fontweight="bold", zorder=9)


def state_node(ax, x: float, y: float, label: str, face: str, edge: str) -> None:
    node = Circle((x, y), 2.15, facecolor=C[face], edgecolor=C[edge], linewidth=1.35, zorder=5)
    ax.add_patch(node)
    ax.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold", zorder=6)


def main() -> None:
    setup_style()
    config = load_json(ROOT / "config" / "ml_activity.json")
    summary = load_json(ROOT / "outputs" / "ml" / "summary.json")
    model_path = ROOT / summary["model_path"]
    model = np.load(model_path, allow_pickle=False)
    trainable = sum(model[name].size for name in ("conv_weight", "conv_bias", "dense_weight", "dense_bias"))
    model_kib = model_path.stat().st_size / 1024

    fig, ax = plt.subplots(figsize=(19, 12), dpi=190, facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.subplots_adjust(left=0.018, right=0.982, top=0.972, bottom=0.035)

    ax.text(3, 97.2, "SheWrist 当前本地算法架构", fontsize=23, fontweight="bold", va="top")
    ax.text(
        3,
        93.5,
        "当前已实现版本 · 公开腕部动作分类基线 · CNN-HMM 仅以 shadow 模式运行",
        fontsize=10.5,
        color=C["muted"],
        va="top",
    )
    ax.text(97, 97.2, "2026-08-28", fontsize=9, color=C["muted"], ha="right", va="top")

    stage1 = FancyBboxPatch(
        (3, 14),
        24.5,
        76,
        boxstyle="round,pad=0.7,rounding_size=3",
        facecolor=C["stage1_fill"],
        edgecolor=C["stage1"],
        linewidth=2.0,
        zorder=0,
    )
    ax.add_patch(stage1)
    stage_badge(ax, 4.8, 89.2, "1", "gray")
    ax.text(15.2, 87.7, "传感、预处理与特征工程", fontsize=13.5, fontweight="bold", ha="center")
    ax.text(15.2, 84.7, "Sensor & Feature Engineering", fontsize=8.5, color=C["muted"], ha="center")

    box(ax, 5.7, 73.5, 8.5, 7.5, "前臂背侧\nforearm IMU", "Ax Ay Az\nGx Gy Gz", face="blue_fill", edge="blue", center=True, title_size=9.5, body_size=7.2)
    box(ax, 16.1, 73.5, 8.5, 7.5, "第3掌骨手背\nhand IMU", "Ax Ay Az\nGx Gy Gz", face="blue_fill", edge="blue", center=True, title_size=8.8, body_size=7.2)
    arrow(ax, (10, 73.4), (13, 69.5), "blue", 1.3)
    arrow(ax, (20.4, 73.4), (17.4, 69.5), "blue", 1.3)
    box(ax, 6.2, 60.5, 18, 8.5, "时间同步与质量", "100 Hz → 50 Hz · 零偏 · 缺包/异常检查", face="white", edge="stage1", center=True, title_size=10, body_size=7.2)
    arrow(ax, (15.2, 60.4), (15.2, 57.6), "stage1")
    box(ax, 6.2, 49, 18, 8.5, "双 IMU 相对姿态", "6轴 Madgwick · q_rel = inv(qF) × qH", face="white", edge="stage1", center=True, title_size=10, body_size=7.1)
    arrow(ax, (15.2, 48.9), (15.2, 46.1), "stage1")
    box(ax, 6.2, 37.5, 18, 8.5, "中立位与功能轴", "归零 · FE/RUD 轴 · 角度与角速度", face="white", edge="stage1", center=True, title_size=10, body_size=7.1)
    arrow(ax, (15.2, 37.4), (15.2, 34.6), "stage1")
    box(
        ax,
        5.4,
        21,
        19.6,
        13.5,
        "滑动窗口 X[t] = 75 x 6",
        "1.5 s 窗 · 0.5 s 步长\nθFE, θRUD, dθFE, dθRUD, |ω|, quality",
        face="amber_fill",
        edge="stage1",
        center=True,
        title_size=10.5,
        body_size=7.2,
    )
    ax.text(15.2, 17.4, "公开数据：11 人 · set2 腕部动作", ha="center", fontsize=8.2, fontweight="bold", color=C["stage1"])

    stage2 = FancyBboxPatch(
        (31, 42),
        66,
        48,
        boxstyle="round,pad=0.7,rounding_size=3",
        facecolor=C["stage2_fill"],
        edgecolor=C["stage2"],
        linewidth=2.0,
        zorder=0,
    )
    ax.add_patch(stage2)
    stage_badge(ax, 32.8, 89.2, "2", "gray")
    ax.text(64, 87.8, "已训练的 CNN-HMM 动作影子模型", fontsize=14, fontweight="bold", ha="center")
    ax.text(64, 84.9, "Trained activity classifier + order-agnostic temporal smoothing", fontsize=8.7, color=C["muted"], ha="center")

    cnn_y = 70.2
    cnn_nodes = [
        (34, 8.3, "标准化", "75 × 6"),
        (44.6, 10.3, "Conv1D", "12 filters · k=7"),
        (57.2, 7.1, "ReLU", "69 × 12"),
        (66.6, 10.4, "时间均值池化", "12"),
        (79.3, 14.4, "Dense + Softmax", "5 类概率 p[t]"),
    ]
    for index, (x, w, title, body) in enumerate(cnn_nodes):
        face = "green_fill" if index == 1 else "white"
        edge = "green" if index == 1 else "stage2"
        box(ax, x, cnn_y, w, 9.3, title, body, face=face, edge=edge, center=True, title_size=9.5, body_size=7.2)
        if index:
            previous_x, previous_w, _, _ = cnn_nodes[index - 1]
            arrow(ax, (previous_x + previous_w, cnn_y + 4.65), (x, cnn_y + 4.65), "ink", 1.4)
    ax.text(38.1, 81.2, "输入", fontsize=8, color=C["muted"], ha="center")
    ax.text(49.8, 81.2, "特征提取", fontsize=8, color=C["green"], ha="center")
    ax.text(86.5, 81.2, "窗口分类", fontsize=8, color=C["stage2"], ha="center")
    arrow(ax, (27.6, 28), (34, 74.8), "stage1", 2.0, curve=-0.12)

    ax.text(34, 65.7, "整段离线 HMM + Viterbi", fontsize=10.5, fontweight="bold")
    ax.text(34, 63.3, "CNN 概率作为 emission；前景类别共享转移先验，避免记住实验顺序", fontsize=7.5, color=C["muted"])
    state_positions = {
        "B": (43, 54),
        "E": (51, 59),
        "F": (51, 49),
        "R": (60, 59),
        "U": (60, 49),
    }
    for label, (x, y) in state_positions.items():
        face = "gray_fill" if label == "B" else "blue_fill"
        edge = "gray" if label == "B" else "blue"
        state_node(ax, x, y, label, face, edge)
    for label in ("E", "F", "R", "U"):
        arrow(ax, state_positions["B"], state_positions[label], "gray", 0.9, dashed=True, zorder=3)
        arrow(ax, state_positions[label], state_positions["B"], "gray", 0.8, dashed=True, curve=0.12, zorder=3)
    ax.text(41.8, 46.1, "B 背景", fontsize=6.9, color=C["muted"])
    ax.text(48.1, 46.1, "E伸 / F屈 / R桡 / U尺", fontsize=6.9, color=C["muted"])

    box(ax, 65.3, 48, 11.3, 12, "Viterbi", "整段最可能\n状态序列", face="purple_fill", edge="purple", center=True, title_size=10.5, body_size=7.5)
    arrow(ax, (62.2, 54), (65.3, 54), "purple", 1.6)
    box(ax, 80, 48, 13.6, 12, "质量与置信度门", "quality ≥ 0.50\nconfidence ≥ 0.55", face="red_fill", edge="red", center=True, title_size=9.5, body_size=7.2)
    arrow(ax, (76.6, 54), (80, 54), "purple", 1.6)
    ax.text(94.1, 57.8, "通过 → 五类动作", fontsize=7.1, color=C["green"], fontweight="bold")
    ax.text(94.1, 51.5, "失败 → unknown", fontsize=7.1, color=C["red"], fontweight="bold")

    metrics = [
        "11 折 LOSO：每折 9 训练 + 1 验证 + 1 测试",
        f"30 epochs · {trainable} trainable params · {model_kib:.1f} KiB",
        f"HMM macro-F1 {summary['aggregate']['hmm_macro_f1']['mean']:.3f} · coverage {summary['aggregate']['coverage']['mean']:.3f}",
        f"完整训练与导出 {summary['elapsed_seconds']:.2f} s · 已保存 NPZ",
    ]
    ax.text(64, 44.4, "    |    ".join(metrics), fontsize=7.25, ha="center", color=C["muted"])

    stage3 = FancyBboxPatch(
        (31, 8),
        66,
        29,
        boxstyle="round,pad=0.7,rounding_size=3",
        facecolor=C["stage3_fill"],
        edgecolor=C["stage3"],
        linewidth=2.0,
        zorder=0,
    )
    ax.add_patch(stage3)
    stage_badge(ax, 32.8, 36.2, "3", "gray")
    ax.text(64, 34.8, "确定性提示与解释严格隔离", fontsize=13.5, fontweight="bold", ha="center")

    ax.text(34, 29.5, "确定性安全通道", fontsize=9.3, fontweight="bold", color=C["green"])
    box(ax, 34, 20.4, 13.5, 7.1, "角度 + 安全症状", "已标定压力可选", face="green_fill", edge="green", center=True, title_size=8.8, body_size=6.7)
    box(ax, 51.1, 20.4, 13.5, 7.1, "阈值与状态机", "持续时间 · 冷却 · 旁路", face="green_fill", edge="green", center=True, title_size=8.8, body_size=6.6)
    box(ax, 68.2, 20.4, 14.6, 7.1, "释放 / 停止提示", "非执行器控制", face="green_fill", edge="green", center=True, title_size=8.8, body_size=6.7)
    for start, end in (((47.5, 24), (51.1, 24)), ((64.6, 24), (68.2, 24))):
        arrow(ax, start, end, "green", 1.6)
    ax.text(85, 23.8, "唯一产生安全提示", fontsize=8.2, color=C["green"], fontweight="bold")

    ax.text(34, 17.3, "ML 解释通道", fontsize=9.3, fontweight="bold", color=C["purple"])
    box(ax, 34, 10.1, 13.5, 5.8, "动作 + 角度 + 质量", face="purple_fill", edge="purple", center=True, title_size=8.4)
    box(ax, 51.1, 10.1, 13.5, 5.8, "惯性 Token", "replay / shadow", face="purple_fill", edge="purple", center=True, title_size=8.8, body_size=6.4)
    box(ax, 68.2, 10.1, 14.6, 5.8, "模板 / 未来 LLM", "只改写结构化事实", face="purple_fill", edge="purple", center=True, title_size=8.5, body_size=6.4)
    for start, end in (((47.5, 13), (51.1, 13)), ((64.6, 13), (68.2, 13))):
        arrow(ax, start, end, "purple", 1.5, dashed=True)
    box(ax, 85, 10.1, 9, 5.8, "NO CONTROL", "安全效力 = none", face="red_fill", edge="red", center=True, title_size=8.2, body_size=6.3)
    arrow(ax, (82.8, 13), (85, 13), "red", 1.3, dashed=True)

    ax.text(
        50,
        3.7,
        "与参考图的关键差异：当前没有 coping behavior 真实标签，因此没有训练 coping 二阶段模型；现有模型只识别五类公开腕部动作。",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=C["red"],
        bbox={"boxstyle": "round,pad=0.45", "facecolor": C["red_fill"], "edgecolor": C["red"], "linewidth": 1.1},
    )
    ax.text(
        50,
        0.8,
        "图：SheWrist 当前已实现架构。实线绿色为确定性安全链；虚线紫色为无控制权限的 ML 影子链。",
        ha="center",
        fontsize=8.5,
        color=C["muted"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / "05_current_model_architecture.png"
    svg = OUTPUT_DIR / "05_current_model_architecture.svg"
    fig.savefig(png, dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(
        json.dumps(
            {
                "png": str(png.relative_to(ROOT)),
                "svg": str(svg.relative_to(ROOT)),
                "dataset_subjects": summary["dataset"]["subjects"],
                "windows": summary["dataset"]["windows"],
                "trainable_parameters": int(trainable),
                "model_size_bytes": model_path.stat().st_size,
                "trained": True,
                "operating_mode": summary["operating_mode"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()