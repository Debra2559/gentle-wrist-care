#!/usr/bin/env python3
"""Generate a visual snapshot of the current SheWrist project, data, and roadmap."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "project_overview"
TODAY = date(2026, 8, 28)

COLORS = {
    "ink": "#172B3A",
    "muted": "#657684",
    "line": "#CAD4DC",
    "bg": "#F5F7FA",
    "panel": "#FFFFFF",
    "blue": "#2F6F9F",
    "blue_light": "#E8F2F8",
    "teal": "#318B82",
    "teal_light": "#E5F4F1",
    "green": "#33865A",
    "green_light": "#E8F4EC",
    "amber": "#C98624",
    "amber_light": "#FFF2D9",
    "red": "#B9504B",
    "red_light": "#FBE9E7",
    "purple": "#6D5A9B",
    "purple_light": "#F0ECF8",
    "gray_light": "#EEF1F4",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def configure_style() -> None:
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
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["bg"],
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
        }
    )


def canvas(width: float, height: float, title: str, subtitle: str = ""):
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.subplots_adjust(left=0.025, right=0.975, top=0.96, bottom=0.04)
    ax.text(2, 97, title, fontsize=22, fontweight="bold", va="top", color=COLORS["ink"])
    if subtitle:
        ax.text(2, 92.5, subtitle, fontsize=10.5, va="top", color=COLORS["muted"])
    ax.text(98, 97, TODAY.isoformat(), fontsize=9, va="top", ha="right", color=COLORS["muted"])
    return fig, ax


def rounded_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str = "",
    face: str = "panel",
    edge: str = "line",
    title_color: str = "ink",
    body_color: str = "muted",
    title_size: float = 11,
    body_size: float = 8.5,
    linewidth: float = 1.2,
    badge: str | None = None,
    badge_color: str = "blue",
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.55,rounding_size=1.2",
        facecolor=COLORS.get(face, face),
        edgecolor=COLORS.get(edge, edge),
        linewidth=linewidth,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 1.4,
        y + height - 1.5,
        title,
        fontsize=title_size,
        fontweight="bold",
        va="top",
        color=COLORS.get(title_color, title_color),
        zorder=3,
    )
    if badge:
        ax.text(
            x + width - 1.2,
            y + height - 1.35,
            badge,
            fontsize=7.2,
            fontweight="bold",
            va="top",
            ha="right",
            color=COLORS.get(badge_color, badge_color),
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": COLORS["panel"],
                "edgecolor": COLORS.get(badge_color, badge_color),
                "linewidth": 0.8,
            },
            zorder=4,
        )
    if body:
        ax.text(
            x + 1.4,
            y + height - 5.1,
            body,
            fontsize=body_size,
            va="top",
            linespacing=1.45,
            color=COLORS.get(body_color, body_color),
            zorder=3,
        )
    return patch


def arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "line", width: float = 1.5):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=width,
        color=COLORS.get(color, color),
        connectionstyle="arc3,rad=0.0",
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def save(fig, stem: str) -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg"):
        path = OUTPUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def current_test_count() -> tuple[int, int]:
    text = (ROOT / "docs" / "validation_report.md").read_text(encoding="utf-8")
    matches = re.findall(r"`(\d+)/(\d+)`[^\n]*测试通过", text)
    return tuple(map(int, matches[-1])) if matches else (0, 0)


def draw_dashboard(summary: dict[str, Any]) -> list[str]:
    tests_passed, tests_total = current_test_count()
    aggregate = summary["aggregate"]
    fig, ax = canvas(
        18,
        10,
        "SheWrist 项目现状总览",
        "工程算法闭环已经跑通；真正缺口是目标硬件与独立标签，不是训练算力。",
    )
    cards = [
        ("公开参与者", str(summary["dataset"]["subjects"]), "Upper-body movements"),
        ("训练窗口", f"{summary['dataset']['windows']:,}", "75 × 6 输入"),
        ("回归测试", f"{tests_passed}/{tests_total}", "全部通过"),
        ("完整训练", f"{summary['elapsed_seconds']:.1f} s", "11 折 + 容错 + 最终模型"),
        ("HMM macro-F1", f"{aggregate['hmm_macro_f1']['mean']:.3f}", "最差折 0.188"),
        ("运行权限", "SHADOW", "ML 控制权限 = none"),
    ]
    for index, (label, value, note) in enumerate(cards):
        x = 2 + index * 16.1
        face = "purple_light" if index == 5 else "panel"
        edge = "purple" if index == 5 else "line"
        rounded_box(ax, x, 77, 14.5, 11.5, label, "", face=face, edge=edge, title_size=8.5)
        ax.text(x + 1.4, 82.8, value, fontsize=18, fontweight="bold", va="center", color=COLORS["ink"])
        ax.text(x + 1.4, 79.2, note, fontsize=7.3, va="center", color=COLORS["muted"])

    rounded_box(
        ax,
        2,
        43,
        47,
        29,
        "当前已经完成",
        "双 IMU 同步、融合、相对腕角与质量门控\n"
        "角度/已标定压力/安全症状状态机与 A/B/C 分析\n"
        "公开数据 11 折 CNN-HMM 基线与有限锁测\n"
        "分块回放、9 场景故障、Token 与双分支审计\n"
        f"可替换解释 API、一键报告与 {tests_passed}/{tests_total} 回归测试",
        face="green_light",
        edge="green",
        badge="已实现",
        badge_color="green",
        body_size=10,
    )
    rounded_box(
        ax,
        51,
        43,
        47,
        29,
        "当前不能声称",
        "公开数据没有独立腕角真值、压力或疼痛标签\n"
        "CNN-HMM 不能预测劳损容量或疾病风险\n"
        "合成 A/B/C 不能证明人体效果\n"
        "光学样例是源工具箱基线，不是当前算法精度\n"
        "未经目标硬件与人体数据，不得进入自动控制",
        face="red_light",
        edge="red",
        badge="证据边界",
        badge_color="red",
        body_size=10,
    )
    rounded_box(
        ax,
        2,
        12,
        30,
        25,
        "数据现在怎么用",
        "公开双 IMU：保留为动作与方向基线\n"
        "Opto 样例：保留为误差方法参考\n"
        "合成 A/B/C：只做软件与故障闭环\n"
        "不继续用公开数据硬凑产品标签",
        face="blue_light",
        edge="blue",
        badge="冻结基线",
        badge_color="blue",
        body_size=9.2,
    )
    rounded_box(
        ax,
        35,
        12,
        30,
        25,
        "当前下一步",
        "1. 评审离线 v0.8 单命令演示\n"
        "2. 核对故障降级与证据边界\n"
        "3. 冻结数据、模型与 API 契约\n"
        "4. 硬件接入继续暂缓",
        face="amber_light",
        edge="amber",
        badge="离线优先",
        badge_color="amber",
        body_size=9.2,
    )
    rounded_box(
        ax,
        68,
        12,
        30,
        25,
        "之后才做",
        "合规后采集目标硬件多人、多会话、重戴数据\n"
        "独立视频标签与长时背景任务\n"
        "重新做按人验证，仍先保持 shadow\n"
        "人体 A/B/C 不作为当前四天关键路径",
        face="purple_light",
        edge="purple",
        badge="需新数据",
        badge_color="purple",
        body_size=9.2,
    )
    ax.text(
        50,
        5.5,
        "当前成熟度：工程原型可运行；产品人体 Go/No-Go 尚未建立。",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )
    return save(fig, "00_project_dashboard")


def draw_flow() -> list[str]:
    fig, ax = canvas(
        20,
        11,
        "当前系统流程",
        "同一份腕部状态并行进入确定性安全链与 ML 影子链；只有确定性链拥有提醒与释放权限。",
    )
    rounded_box(
        ax,
        2,
        71,
        16,
        14,
        "输入",
        "前臂背侧 forearm IMU\n第3掌骨手背 hand IMU\nRFP-602 / 评分 / 安全症状",
        face="blue_light",
        edge="blue",
        badge="目标硬件待物理验证",
        badge_color="amber",
        body_size=8.4,
    )
    rounded_box(ax, 22, 71, 15, 14, "同步与质量", "时间检查\n重采样\n零偏与质量评分", face="panel", edge="blue")
    rounded_box(ax, 41, 71, 15, 14, "姿态融合", "6 轴 Madgwick\nq_rel\n中立位归零", face="panel", edge="blue")
    rounded_box(ax, 60, 71, 17, 14, "功能轴与运动学", "功能动作标定\nFE / RUD 角度\n角速度", face="panel", edge="blue")
    rounded_box(ax, 81, 71, 17, 14, "统一腕部状态", "joint_state\nquality\n机械通道", face="teal_light", edge="teal", badge="已实现", badge_color="green")
    for left, right in ((18, 22), (37, 41), (56, 60), (77, 81)):
        arrow(ax, (left, 78), (right, 78), "blue")

    arrow(ax, (89.5, 71), (72, 60), "green", 2.0)
    arrow(ax, (89.5, 71), (72, 35), "purple", 2.0)

    rounded_box(
        ax,
        2,
        43,
        20,
        18,
        "确定性规则",
        "角度阈值与持续时间\n已标定压力 > 4.4 kPa\n安全症状独立旁路",
        face="green_light",
        edge="green",
        badge="控制链",
        badge_color="green",
    )
    rounded_box(ax, 27, 43, 20, 18, "暴露与状态机", "P_high / D_FE / D_RUD\n冷却与去抖\nA/B/C 指标", face="green_light", edge="green")
    rounded_box(
        ax,
        52,
        43,
        20,
        18,
        "安全输出",
        "角度提醒\n释放 / 停止提示\n非执行器控制",
        face="green_light",
        edge="green",
        badge="有权限",
        badge_color="green",
    )
    rounded_box(ax, 77, 43, 21, 18, "工程报告", "暴露指标\n报警记录\n三态 Go/No-Go", face="panel", edge="green")
    for left, right in ((22, 27), (47, 52), (72, 77)):
        arrow(ax, (left, 52), (right, 52), "green")

    rounded_box(ax, 2, 17, 17, 18, "滑动窗口", "50 Hz\n1.5 s 窗\n0.5 s 步长", face="purple_light", edge="purple")
    rounded_box(ax, 23, 17, 17, 18, "1D-CNN", "五类公开动作\nSoftmax\n温度校准", face="purple_light", edge="purple")
    rounded_box(ax, 44, 17, 17, 18, "HMM", "整段离线解码\n背景/动作持续\n不记忆实验顺序", face="purple_light", edge="purple")
    rounded_box(ax, 65, 17, 16, 18, "拒识", "质量 < 0.5\n置信度 < 0.55\n输出 unknown", face="purple_light", edge="purple")
    rounded_box(
        ax,
        85,
        17,
        13,
        18,
        "影子输出",
        "动作 Token\n模板解释\n控制效力 = none",
        face="purple_light",
        edge="purple",
        badge="无权限",
        badge_color="red",
        body_size=8,
    )
    for left, right in ((19, 23), (40, 44), (61, 65), (81, 85)):
        arrow(ax, (left, 26), (right, 26), "purple")

    ax.text(
        50,
        7.5,
        "关键隔离：ML 可以描述动作，但不能生成、取消或修改角度报警、压力停止与机械动作。",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["red"],
        bbox={"boxstyle": "round,pad=0.5", "facecolor": COLORS["red_light"], "edgecolor": COLORS["red"]},
    )
    return save(fig, "01_system_flow")


def draw_structure() -> list[str]:
    tests_passed, tests_total = current_test_count()
    source_module_count = len([path for path in (ROOT / "src" / "shewrist").glob("*.py") if path.name != "__init__.py"])
    script_count = len(list((ROOT / "scripts").glob("*.py")))
    test_file_count = len(list((ROOT / "tests").glob("test_*.py")))
    fig, ax = canvas(
        18,
        11,
        "当前代码与目录结构",
        "工程分为数据/配置、运动学核心、确定性安全、ML 影子、入口脚本、验证与产物六层。",
    )
    rounded_box(
        ax,
        2,
        10,
        36,
        77,
        "仓库树",
        "Algorithm/\n"
        "├─ config/       阈值、ML 与解释 API 配置\n"
        "├─ data/\n"
        "│  ├─ raw/       公开 IMU 与 Opto 样例\n"
        "│  └─ processed/ 标准 joint_state\n"
        f"├─ src/shewrist/ {source_module_count} 个核心模块\n"
        f"├─ scripts/      {script_count} 个训练/分析入口\n"
        f"├─ tests/        {test_file_count} 个测试文件，{tests_passed}/{tests_total} 通过\n"
        "├─ examples/     CSV 模板与合成 A/B/C\n"
        "├─ outputs/      验证、模型、预测与报告\n"
        "└─ docs/         设计、接口、证据边界",
        face="panel",
        edge="line",
        title_size=13,
        body_size=10.2,
    )

    rounded_box(ax, 43, 72, 24, 15, "数据与配置", "data.py · ml_data.py\nthresholds.yaml · ml_activity.json", face="blue_light", edge="blue", badge="输入层", badge_color="blue")
    rounded_box(ax, 72, 72, 26, 15, "入口脚本", "run_all · analyze_*\ntrain_activity_model · validate_opto", face="blue_light", edge="blue", badge="CLI", badge_color="blue")
    rounded_box(ax, 43, 51, 24, 15, "运动学核心", "quaternion · fusion · calibration\nkinematics · quality", face="teal_light", edge="teal", badge="共享核心", badge_color="teal")
    rounded_box(ax, 72, 51, 26, 15, "确定性安全", "exposure · metrics · analysis\n阈值、状态机、A/B/C", face="green_light", edge="green", badge="有控制权限", badge_color="green")
    rounded_box(ax, 43, 30, 24, 15, "ML 影子", "ml · hmm · tokens\nLOSO、拒识、Token", face="purple_light", edge="purple", badge="无控制权限", badge_color="purple")
    rounded_box(ax, 72, 30, 26, 15, "编排与验证", "pipeline · validation\n双分支隔离、指标与报告", face="amber_light", edge="amber", badge="审计层", badge_color="amber")
    rounded_box(ax, 43, 10, 55, 13, "可复现产物", "outputs/public_dataset · outputs/ml · outputs/project_overview\n模型、OOF 预测、Token、验证报告和本次图表", face="panel", edge="line", badge="机器生成", badge_color="blue")

    arrow(ax, (55, 72), (55, 66), "blue")
    arrow(ax, (67, 58.5), (72, 58.5), "green")
    arrow(ax, (55, 51), (55, 45), "purple")
    arrow(ax, (67, 37.5), (72, 37.5), "amber")
    arrow(ax, (85, 51), (85, 45), "amber")
    arrow(ax, (70.5, 30), (70.5, 23), "line")
    arrow(ax, (85, 30), (85, 23), "line")
    arrow(ax, (85, 72), (85, 66), "blue")
    ax.text(70.5, 68.5, "调用", fontsize=8, color=COLORS["muted"], ha="center")
    ax.text(69.5, 47.7, "并行", fontsize=8, color=COLORS["muted"], ha="center")
    return save(fig, "02_repository_structure")


def draw_data_roadmap(summary: dict[str, Any], opto: dict[str, Any]) -> list[str]:
    fig, ax = canvas(
        20,
        12,
        "数据现状与下一步路线",
        "现有数据足以验证代码和公开动作基线，但不足以训练 SheWrist 的产品目标或证明人体效果。",
    )
    datasets = [
        (
            "公开双 IMU",
            f"已下载 · {summary['dataset']['subjects']} 人 · {summary['dataset']['windows']} 窗口",
            "用途：相对腕姿、动作方向、CNN-HMM 基线\n缺口：无独立角度、压力、疼痛或疾病真值",
            "green_light",
            "green",
            "可用",
        ),
        (
            "Opto 参考样例",
            "已下载 · 单公开参与者 · 源工具箱基线",
            f"FE MAE {opto['results']['wrist_flexion_extension']['mae_deg']:.1f}°；RUD MAE {opto['results']['wrist_radial_ulnar_deviation']['mae_deg']:.1f}°\n用途：误差评估流程，不代表当前算法精度",
            "blue_light",
            "blue",
            "参考",
        ),
        (
            "合成 A/B/C",
            "已生成 · 软件演示数据",
            "用途：状态机、FSR 代理量、统计和三态判定\n缺口：不能替代人体暴露、舒适度或效果证据",
            "amber_light",
            "amber",
            "仅演示",
        ),
        (
            "目标硬件台架",
            "尚未采集",
            "需要：双 IMU 同步、角度夹具、FSR 标定\n重戴、串扰、滑移与快速释放",
            "red_light",
            "red",
            "后续暂缓",
        ),
        (
            "目标硬件人体标签",
            "尚未采集",
            "需要：多人、多会话、重戴、独立视频标签\n长时背景任务；合规后再做 A/B/C",
            "purple_light",
            "purple",
            "之后",
        ),
    ]
    for index, item in enumerate(datasets):
        x = 2 + index * 19.5
        rounded_box(
            ax,
            x,
            64,
            18,
            23,
            item[0],
            f"{item[1]}\n\n{item[2]}",
            face=item[3],
            edge=item[4],
            badge=item[5],
            badge_color=item[4],
            title_size=10.5,
            body_size=7.8,
        )

    phases = [
        ("0", "公开基线冻结", "已完成", "保留结果，不再靠调参硬凑产品结论", "green"),
        ("1", "离线 v0.8", "已完成", "原始双 IMU → 回放 → 双分支 → 报告与审计", "green"),
        ("2", "离线评审", "当前", "演示、故障边界、模型与 API 契约冻结", "amber"),
        ("3", "硬件与台架", "暂缓", "同步、角度真值、FSR 标定、重戴、滑移与释放", "blue"),
        ("4", "目标标签", "需合规", "多人多会话与独立标签，重新按人评估，仍先 shadow", "purple"),
        ("5", "人体 A/B/C", "长期", "压力、舒适、效率和提醒接受；不能由合成数据替代", "purple"),
    ]
    y = 48
    ax.text(2, 55, "实施顺序", fontsize=15, fontweight="bold", color=COLORS["ink"])
    for index, (number, title, status, body, color) in enumerate(phases):
        x = 2 + index * 16.2
        ax.add_patch(plt.Circle((x + 2.2, y), 2.2, facecolor=COLORS[f"{color}_light"] if f"{color}_light" in COLORS else COLORS["panel"], edgecolor=COLORS[color], linewidth=1.5, zorder=3))
        ax.text(x + 2.2, y, number, ha="center", va="center", fontsize=11, fontweight="bold", color=COLORS[color], zorder=4)
        if index < len(phases) - 1:
            arrow(ax, (x + 4.5, y), (x + 15.4, y), "line")
        ax.text(x, y - 5, title, fontsize=9.5, fontweight="bold", va="top", color=COLORS["ink"])
        ax.text(x, y - 8.2, status, fontsize=8, fontweight="bold", va="top", color=COLORS[color])
        ax.text(x, y - 11.2, body, fontsize=7.4, va="top", color=COLORS["muted"], linespacing=1.35, wrap=True)

    rounded_box(
        ax,
        2,
        6,
        96,
        13,
        "下一批数据的最小统一字段",
        "raw IMU: device_ms, sensor_id, ax, ay, az, gx, gy, gz, quality    |    "
        "joint_state: timestamp_ms, theta_FE, theta_RUD, calibration_id, quality    |    "
        "mechanical: device_ms, condition, support_level, fsr_raw_adc, discomfort_nrs, safety_symptom_flag",
        face="panel",
        edge="line",
        body_size=8.2,
    )
    return save(fig, "03_data_roadmap")


def draw_ml_evidence(summary: dict[str, Any]) -> list[str]:
    fig = plt.figure(figsize=(18, 11), dpi=180, facecolor=COLORS["bg"])
    grid = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.86, bottom=0.09, hspace=0.38, wspace=0.24)
    fig.text(0.04, 0.95, "机器学习证据面板", fontsize=22, fontweight="bold", color=COLORS["ink"])
    fig.text(0.04, 0.91, "公开数据上的五类腕部动作影子模型：展示真实能力，也展示为什么现在不能进入控制。", fontsize=10.5, color=COLORS["muted"])
    fig.text(0.96, 0.95, TODAY.isoformat(), fontsize=9, ha="right", color=COLORS["muted"])

    counts = summary["dataset"]["label_counts"]
    labels = ["背景", "伸展", "屈曲", "桡偏", "尺偏"]
    count_values = [counts[key] for key in ("background", "extension", "flexion", "radial_deviation", "ulnar_deviation")]
    ax1 = fig.add_subplot(grid[0, 0])
    bars = ax1.bar(labels, count_values, color=[COLORS["blue"], COLORS["teal"], COLORS["teal"], COLORS["teal"], COLORS["teal"]])
    ax1.set_title("窗口标签分布", loc="left", fontsize=13, fontweight="bold")
    ax1.set_ylabel("窗口数")
    ax1.grid(axis="y", alpha=0.22)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, count_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 22, str(value), ha="center", fontsize=8.5, color=COLORS["ink"])
    ax1.text(0.01, 0.93, f"背景占 {100 * count_values[0] / sum(count_values):.1f}%：类别明显不平衡", transform=ax1.transAxes, fontsize=9, color=COLORS["red"])

    aggregate = summary["aggregate"]
    metric_labels = ["CNN 准确率", "HMM 准确率", "CNN macro-F1", "HMM macro-F1", "接受样本准确率", "覆盖率", "事件 F1"]
    metric_keys = ["raw_accuracy", "hmm_accuracy", "raw_macro_f1", "hmm_macro_f1", "accepted_selective_accuracy", "coverage", "event_f1"]
    values = [100 * aggregate[key]["mean"] for key in metric_keys]
    errors = [100 * aggregate[key]["std"] for key in metric_keys]
    colors = [COLORS["blue"], COLORS["teal"], COLORS["blue"], COLORS["teal"], COLORS["purple"], COLORS["amber"], COLORS["purple"]]
    ax2 = fig.add_subplot(grid[0, 1])
    positions = np.arange(len(values))
    bars = ax2.barh(positions, values, xerr=errors, color=colors, alpha=0.92, capsize=3)
    ax2.set_yticks(positions, metric_labels)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 105)
    ax2.xaxis.set_major_formatter(PercentFormatter())
    ax2.set_title("11 折按人评估", loc="left", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.22)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax2.text(min(value + 1.5, 98), bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=8.5)

    robustness = summary["robustness_aggregate"]
    missing_rows = sorted((row for row in robustness if row["noise_fraction_of_feature_std"] == 0.0), key=lambda row: row["missing_sample_fraction"])
    missing = [100 * row["missing_sample_fraction"] for row in missing_rows]
    ax3 = fig.add_subplot(grid[1, 0])
    for key, label, color in (
        ("macro_f1_mean", "拒识后 macro-F1", "blue"),
        ("coverage_mean", "覆盖率", "amber"),
        ("event_f1_mean", "事件 F1", "purple"),
    ):
        ax3.plot(missing, [100 * row[key] for row in missing_rows], marker="o", linewidth=2.2, label=label, color=COLORS[color])
    ax3.set_title("随机缺失退化", loc="left", fontsize=13, fontweight="bold")
    ax3.set_xlabel("缺失样本比例")
    ax3.set_ylabel("指标")
    ax3.set_xticks(missing)
    ax3.xaxis.set_major_formatter(PercentFormatter())
    ax3.yaxis.set_major_formatter(PercentFormatter())
    ax3.set_ylim(0, 75)
    ax3.grid(alpha=0.22)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.legend(frameon=False, fontsize=8.5)

    ax4 = fig.add_subplot(grid[1, 1])
    ax4.axis("off")
    status_items = [
        ("已验证", "训练、LOSO、HMM、拒识、保存/加载、故障注入", "green_light", "green"),
        ("有限能力", f"HMM macro-F1 均值 {aggregate['hmm_macro_f1']['mean']:.3f}，最差 {aggregate['hmm_macro_f1']['min']:.3f}", "amber_light", "amber"),
        ("校准不足", f"ECE 均值 {aggregate['calibration_error']['mean']:.3f}，覆盖率波动大", "amber_light", "amber"),
        ("禁止用途", "劳损/疾病预测、压力安全判断、报警与机械控制", "red_light", "red"),
        ("升级条件", "目标硬件、多人多会话、重戴、独立标签、长时背景", "purple_light", "purple"),
    ]
    for index, (title, body, face, edge) in enumerate(status_items):
        y = 0.82 - index * 0.18
        patch = FancyBboxPatch((0.02, y), 0.95, 0.135, transform=ax4.transAxes, boxstyle="round,pad=0.012", facecolor=COLORS[face], edgecolor=COLORS[edge], linewidth=1.1)
        ax4.add_patch(patch)
        ax4.text(0.05, y + 0.092, title, transform=ax4.transAxes, fontsize=10, fontweight="bold", color=COLORS[edge], va="center")
        ax4.text(0.05, y + 0.045, body, transform=ax4.transAxes, fontsize=8.5, color=COLORS["ink"], va="center")
    ax4.set_title("可用性判断", loc="left", fontsize=13, fontweight="bold")

    fig.text(0.5, 0.025, "结论：模型代码已完成，但现有数据只支持公开动作影子基线；下一次提升必须来自新数据，而不是继续调参。", ha="center", fontsize=11.5, fontweight="bold", color=COLORS["red"])
    return save(fig, "04_ml_evidence")


def main() -> None:
    configure_style()
    summary = load_json(ROOT / "outputs" / "ml" / "summary.json")
    opto = load_json(ROOT / "outputs" / "opto_reference_validation.json")
    generated: list[str] = []
    generated.extend(draw_dashboard(summary))
    generated.extend(draw_flow())
    generated.extend(draw_structure())
    generated.extend(draw_data_roadmap(summary, opto))
    generated.extend(draw_ml_evidence(summary))
    current_architecture = [
        "outputs/project_overview/05_current_model_architecture.png",
        "outputs/project_overview/05_current_model_architecture.svg",
    ]
    generated.extend(path for path in current_architecture if (ROOT / path).exists())
    manifest = {
        "generated_on": TODAY.isoformat(),
        "project": "SheWrist",
        "status": "engineering_prototype_ready_human_go_no_go_not_established",
        "sources": [
            "outputs/ml/summary.json",
            "outputs/opto_reference_validation.json",
            "docs/algorithm_design.md",
            "docs/data_interface.md",
            "docs/validation_report.md",
        ],
        "figures": generated,
        "key_conclusion": "Offline v0.8 is the current completed deliverable. Hardware remains deferred; target-hardware bench logs and independently labeled multi-session recordings are required later for product evidence.",
    }
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()