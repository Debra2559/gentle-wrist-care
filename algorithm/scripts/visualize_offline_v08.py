#!/usr/bin/env python3
"""Draw the implemented offline-v0.8 architecture and delivery status."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "offline_v08"
COLORS = {
    "bg": "#F5F7FA",
    "ink": "#172B3A",
    "muted": "#657684",
    "line": "#CAD4DC",
    "blue": "#2F6F9F",
    "blue_light": "#E8F2F8",
    "green": "#33865A",
    "green_light": "#E8F4EC",
    "purple": "#6D5A9B",
    "purple_light": "#F0ECF8",
    "amber": "#C98624",
    "amber_light": "#FFF2D9",
    "red": "#B9504B",
    "red_light": "#FBE9E7",
    "white": "#FFFFFF",
}


def box(ax, x, y, w, h, title, body, color, badge=""):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2", facecolor=COLORS[f"{color}_light"], edgecolor=COLORS[color], linewidth=1.3)
    ax.add_patch(patch)
    ax.text(x + 1.2, y + h - 1.5, title, va="top", fontsize=10.5, fontweight="bold", color=COLORS["ink"])
    ax.text(x + 1.2, y + h - 5.2, body, va="top", fontsize=8.2, linespacing=1.45, color=COLORS["muted"])
    if badge:
        ax.text(x + w - 1.0, y + h - 1.2, badge, ha="right", va="top", fontsize=7, fontweight="bold", color=COLORS[color], bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": COLORS[color]})


def arrow(ax, x1, y1, x2, y2, color="line", style="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.4, linestyle=style, color=COLORS[color]))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg"):
        path = OUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def draw_architecture():
    fig, ax = plt.subplots(figsize=(19, 11), dpi=180, facecolor=COLORS["bg"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
    ax.text(2, 97, "SheWrist 纯离线算法 v0.8", fontsize=22, fontweight="bold", va="top", color=COLORS["ink"])
    ax.text(2, 92, "当前已实现的数据、算法、回放、报告与可替换解释 API 架构", fontsize=10.5, color=COLORS["muted"])
    box(ax, 2, 72, 15, 13, "输入契约", "公开双 IMU 原始流\n或 joint_state.csv\n可选机械通道", "blue", "已实现")
    box(ax, 21, 72, 15, 13, "输入审计", "字段/时间戳检查\nSHA-256 清单\n证据类型", "blue", "已实现")
    box(ax, 40, 72, 16, 13, "同步与校准", "100 Hz 重采样\n中立位/功能轴\n失败显式退出", "blue", "已实现")
    box(ax, 60, 72, 16, 13, "腕部状态", "FE / RUD / 角速度\nquality\ncalibration_id", "blue", "已实现")
    box(ax, 80, 72, 18, 13, "分块回放", "128 samples/chunk\n跨块状态保持\n批/流一致校验", "green", "已实现")
    for left, right in ((17, 21), (36, 40), (56, 60), (76, 80)):
        arrow(ax, left, 78.5, right, 78.5, "blue")
    box(ax, 4, 43, 22, 18, "确定性安全链", "角度/持续时间\n压力与不适旁路\n提醒/停止/人工建议", "green", "唯一控制权")
    box(ax, 31, 43, 22, 18, "ML 影子链", "1.5 s 窗口\n1D-CNN + HMM\n质量/置信度拒识", "purple", "control=none")
    box(ax, 58, 43, 18, 18, "结构化 Token", "动作/区间/置信度\n质量/模型版本\nsafety_effect=none", "purple", "稳定契约")
    box(ax, 81, 43, 17, 18, "解释适配器", "默认本地模板\nmodel 字段占位\n可换生产 API", "amber", "API 默认关闭")
    arrow(ax, 89, 72, 15, 61, "green")
    arrow(ax, 89, 72, 42, 61, "purple")
    arrow(ax, 53, 52, 58, 52, "purple")
    arrow(ax, 76, 52, 81, 52, "amber")
    box(ax, 4, 16, 25, 16, "自动报告", "analysis.json / tokens.json\ntimeline.csv / joint_state.csv\nPNG + SVG 时间轴", "green", "一条命令")
    box(ax, 37, 16, 25, 16, "故障矩阵", "丢包/乱序/时移/静默\n饱和/零偏/旋转/滑移\n检测与证据边界分开", "amber", "9 场景")
    box(ax, 70, 16, 28, 16, "未来替换点", "SHEWRIST_LLM_ENDPOINT\nSHEWRIST_LLM_API_KEY\nSHEWRIST_LLM_MODEL", "purple", "OpenAI-compatible")
    arrow(ax, 15, 43, 16, 32, "green")
    arrow(ax, 42, 43, 49, 32, "purple")
    arrow(ax, 89, 43, 84, 32, "amber")
    ax.text(50, 7, "硬边界：LLM/ML 只能解释结构化事实，不能创建、取消或修改提醒、压力停止与机械动作。", ha="center", fontsize=11.5, fontweight="bold", color=COLORS["red"], bbox={"boxstyle": "round,pad=0.5", "facecolor": COLORS["red_light"], "edgecolor": COLORS["red"]})
    return save(fig, "06_offline_v08_architecture")


def draw_status():
    analysis = json.loads((ROOT / "outputs/offline_session/analysis.json").read_text(encoding="utf-8"))
    fault = json.loads((ROOT / "outputs/fault_suite/fault_report.json").read_text(encoding="utf-8"))
    metrics = analysis["deterministic_control"]["metrics"]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180, facecolor=COLORS["bg"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.subplots_adjust(left=0.025, right=0.975, top=0.95, bottom=0.05)
    ax.text(2, 96, "离线 v0.8 实际验收结果", fontsize=21, fontweight="bold", va="top", color=COLORS["ink"])
    ax.text(2, 90, "subject01 原始双 IMU → 校准 → 腕角 → 双分支 → Token → 报告", fontsize=10, color=COLORS["muted"])
    cards = [
        ("有效样本", f"{metrics['valid_sample_pct']:.2f}%", "green"),
        ("分块数", str(analysis["replay"]["chunk_count"]), "blue"),
        ("影子 Token", str(len(analysis["ml_shadow"]["tokens"])), "purple"),
        ("故障场景", str(len(fault["results"])), "amber"),
        ("外部 API", "未调用", "green"),
    ]
    for i, (label, value, color) in enumerate(cards):
        x = 2 + i * 19.4
        box(ax, x, 72, 17.5, 13, label, "", color)
        ax.text(x + 1.2, 76.4, value, fontsize=18, fontweight="bold", color=COLORS["ink"])
    box(ax, 2, 37, 46, 27, "已通过", "原始双 IMU 单命令闭环\n确定性状态跨分块一致\n最终 JSON 指纹一致\n低质量窗口拒识\nLLM/API 默认关闭\nML/LLM 控制权限均为 none", "green", "software acceptance")
    box(ax, 52, 37, 46, 27, "仍未证明", "目标硬件时间同步与漂移\n独立角度真值 MAE\nFSR/张力/快速释放性能\n滑移与零偏的可靠在线检测\n人体有效性、舒适度或疾病风险", "red", "evidence boundary")
    box(ax, 2, 12, 30, 17, "现在的数据", "公开双 IMU：算法基线\n合成数据：软件安全链\nOpto 样例：方法参考", "blue")
    box(ax, 35, 12, 30, 17, "现在的模型", "CNN-HMM 已训练\n严格 shadow + 拒识\n有限参数选择另行锁测", "purple")
    box(ax, 68, 12, 30, 17, "下一阶段", "继续完善离线 demo\n生产 API 只换适配器\n硬件仍暂缓接入", "amber")
    return save(fig, "07_offline_v08_acceptance")


def main():
    plt.rcParams.update({"font.sans-serif": ["Hiragino Sans GB", "PingFang HK", "Arial Unicode MS", "Heiti TC", "DejaVu Sans"], "axes.unicode_minus": False})
    paths = draw_architecture() + draw_status()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps({"generated": paths}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generated": paths}, ensure_ascii=False))


if __name__ == "__main__":
    main()
