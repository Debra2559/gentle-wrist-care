#!/usr/bin/env python3
"""生成「校准阶段状态图」SVG：手工排版的 UML 状态机，两行蛇形主链，16:9。"""

from __future__ import annotations

import html
from pathlib import Path

W, H = 1600, 900

FONT = "PingFang SC, Hiragino Sans GB, Heiti SC, Microsoft YaHei, sans-serif"
C_TEXT = "#1f2933"
C_SUB = "#52606d"
C_LINE = "#7b8794"

FILL_NORMAL, STROKE_NORMAL = "#e8eef4", "#9fb3c8"
FILL_BAD, STROKE_BAD = "#fbe3e3", "#dc9a9a"
FILL_OK, STROKE_OK = "#e2f4e4", "#86bf92"


class Box:
    def __init__(self, key, cx, cy, w, h, title, lines, kind="normal"):
        self.key, self.cx, self.cy, self.w, self.h = key, cx, cy, w, h
        self.title, self.lines, self.kind = title, lines, kind

    @property
    def left(self): return self.cx - self.w / 2

    @property
    def right(self): return self.cx + self.w / 2

    @property
    def top(self): return self.cy - self.h / 2

    @property
    def bottom(self): return self.cy + self.h / 2

    def port(self, side, off=0.0):
        if side == "l": return (self.left, self.cy + off)
        if side == "r": return (self.right, self.cy + off)
        if side == "t": return (self.cx + off, self.top)
        return (self.cx + off, self.bottom)


BOXES = [
    Box("neutral", 250, 250, 250, 86, "中立位采集", ["约 5 s 静止"]),
    Box("neutral_bad", 250, 96, 220, 52, "中立位不合格", [], kind="bad"),
    Box("functional", 660, 250, 330, 108, "功能动作采集",
        ["掌屈 / 背伸 / 尺偏（桡偏可选）", "每段 ≤ 15 s"]),
    Box("functional_bad", 660, 90, 230, 52, "功能校准不合格", [], kind="bad"),
    Box("profile", 1160, 250, 372, 122, "生成校准档案",
        ["置零 + 陀螺零偏 + 主轴 SVD 解剖轴", "→ calibration_id"]),
    Box("capture", 372, 592, 350, 100, "自然放松工作基线采集",
        ["自然工作片段，单次 ≤ 5 min"]),
    Box("provisional", 872, 592, 250, 92, "暂定基线", ["个人 p50 / p90"]),
    Box("official", 1322, 592, 200, 60, "正式基线", [], kind="ok"),
]
B = {b.key: b for b in BOXES}


def style(kind):
    return {
        "normal": (FILL_NORMAL, STROKE_NORMAL),
        "bad": (FILL_BAD, STROKE_BAD),
        "ok": (FILL_OK, STROKE_OK),
    }[kind]


def esc(s): return html.escape(s, quote=False)


def draw_box(b: Box) -> str:
    fill, stroke = style(b.kind)
    out = [
        f'<rect x="{b.left:.1f}" y="{b.top:.1f}" width="{b.w}" height="{b.h}" rx="14" ry="14" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>'
    ]
    n = 1 + len(b.lines)
    title_size, sub_size, lh = 21, 17, 25
    total = title_size + (n - 1) * lh
    y = b.cy - total / 2 + title_size - 4
    out.append(
        f'<text x="{b.cx:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{title_size}" '
        f'fill="{C_TEXT}" text-anchor="middle">{esc(b.title)}</text>'
    )
    for i, line in enumerate(b.lines):
        yy = y + (i + 1) * lh
        out.append(
            f'<text x="{b.cx:.1f}" y="{yy:.1f}" font-family="{FONT}" font-size="{sub_size}" '
            f'fill="{C_SUB}" text-anchor="middle">{esc(line)}</text>'
        )
    return "\n".join(out)


def path(d: str) -> str:
    return (f'<path d="{d}" fill="none" stroke="{C_LINE}" stroke-width="1.6" '
            f'marker-end="url(#arrow)" stroke-linecap="round"/>')


def label(x, y, lines, anchor="middle", size=16):
    out = []
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{x:.1f}" y="{y + i * 20:.1f}" font-family="{FONT}" font-size="{size}" '
            f'fill="{C_TEXT}" text-anchor="{anchor}" '
            f'style="paint-order:stroke;stroke:#ffffff;stroke-width:5px;stroke-linejoin:round">'
            f'{esc(line)}</text>'
        )
    return "\n".join(out)


def build() -> str:
    p, lab = [], []

    # 初始伪状态
    nb = B["neutral"]
    p.append(f'<circle cx="96" cy="{nb.cy}" r="9" fill="#3e4c59"/>')
    p.append(path(f'M 105 {nb.cy} L {nb.left - 6:.1f} {nb.cy}'))

    # 中立位 <-> 中立位不合格
    n, nbad = B["neutral"], B["neutral_bad"]
    p.append(path(f'M {n.cx - 46} {n.top - 4} C {n.cx - 46} {n.top - 40}, '
                  f'{nbad.cx - 46} {nbad.bottom + 40}, {nbad.cx - 46} {nbad.bottom + 5}'))
    lab.append(label(n.cx - 148, (n.top + nbad.bottom) / 2 + 6, ["[晃动 /", "样本不足]"]))
    p.append(path(f'M {nbad.cx + 46} {nbad.bottom + 5} C {nbad.cx + 46} {nbad.bottom + 40}, '
                  f'{n.cx + 46} {n.top - 40}, {n.cx + 46} {n.top - 4}'))
    lab.append(label(n.cx + 132, (n.top + nbad.bottom) / 2 + 6, ["[重新", "录制]"]))

    # 中立位 -> 功能动作采集
    f = B["functional"]
    p.append(path(f'M {n.right + 4} {n.cy} L {f.left - 6} {f.cy}'))
    lab.append(label((n.right + f.left) / 2, n.cy - 30, ["[静止样本 ≥ 30", "且占比 ≥ 70%]"]))

    # 功能采集 <-> 功能校准不合格
    fbad = B["functional_bad"]
    p.append(path(f'M {f.cx - 60} {f.top - 4} C {f.cx - 60} {f.top - 36}, '
                  f'{fbad.cx - 60} {fbad.bottom + 36}, {fbad.cx - 60} {fbad.bottom + 5}'))
    lab.append(label(f.cx - 172, (f.top + fbad.bottom) / 2 + 6, ["[缺段 /", "区间超 15 s]"]))
    p.append(path(f'M {fbad.cx + 60} {fbad.bottom + 5} C {fbad.cx + 60} {fbad.bottom + 36}, '
                  f'{f.cx + 60} {f.top - 36}, {f.cx + 60} {f.top - 4}'))
    lab.append(label(f.cx + 160, (f.top + fbad.bottom) / 2 + 6, ["[重新", "录制]"]))

    # 功能采集 -> 生成校准档案
    pr = B["profile"]
    p.append(path(f'M {f.right + 4} {f.cy} L {pr.left - 6} {pr.cy}'))
    lab.append(label((f.right + pr.left) / 2, f.cy - 30, ["[四段齐全", "且区间有效]"]))

    # 生成校准档案 -> 自然放松工作基线采集（折返到第二行）
    cap = B["capture"]
    turn_y = 430.0
    p.append(path(
        f'M {pr.cx} {pr.bottom + 4} C {pr.cx} {turn_y - 10}, {pr.cx} {turn_y}, {pr.cx - 90} {turn_y} '
        f'L {cap.left - 78} {turn_y} C {cap.left - 118} {turn_y}, {cap.left - 118} {turn_y + 10}, '
        f'{cap.left - 118} {cap.cy - 34} C {cap.left - 118} {cap.cy}, {cap.left - 90} {cap.cy}, {cap.left - 6} {cap.cy}'
    ))
    lab.append(label((pr.cx + cap.cx) / 2 + 40, turn_y - 14, ["（校准档案生成后进入个人基线阶段）"], size=15))

    # 基线采集自环
    p.append(path(
        f'M {cap.cx - 52} {cap.top - 4} C {cap.cx - 84} {cap.top - 76}, '
        f'{cap.cx + 84} {cap.top - 76}, {cap.cx + 52} {cap.top - 6}'
    ))
    lab.append(label(cap.cx, cap.top - 88, ["[单次 > 5 min 或质量不足]", "该次跳过，不计入"]))

    # 基线采集 -> 暂定基线
    pv = B["provisional"]
    p.append(path(f'M {cap.cx + 60} {cap.bottom + 4} C {cap.cx + 60} {cap.bottom + 86}, '
                  f'{pv.cx - 40} {pv.bottom + 86}, {pv.cx - 40} {pv.bottom + 6}'))
    lab.append(label((cap.cx + pv.cx) / 2 + 10, cap.bottom + 104,
                     ["[单次时长 ≥ 1 min 且有效样本 ≥ 60%]"]))

    # 暂定基线自环
    p.append(path(
        f'M {pv.cx - 52} {pv.top - 4} C {pv.cx - 78} {pv.top - 74}, '
        f'{pv.cx + 78} {pv.top - 74}, {pv.cx + 52} {pv.top - 6}'
    ))
    lab.append(label(pv.cx, pv.top - 84, ["[新数据 EWMA α = 0.3 更新]", "越测越修正"]))

    # 暂定基线 -> 正式基线
    of = B["official"]
    p.append(path(f'M {pv.right + 4} {pv.cy} L {of.left - 6} {of.cy}'))
    lab.append(label((pv.right + of.left) / 2, pv.cy - 34, ["[累计有效", "≥ 30 min]"]))

    # 正式基线自环
    p.append(path(
        f'M {of.cx - 44} {of.top - 4} C {of.cx - 70} {of.top - 70}, '
        f'{of.cx + 70} {of.top - 70}, {of.cx + 44} {of.top - 6}'
    ))
    lab.append(label(of.cx, of.top - 80, ["[新数据 EWMA α = 0.3]", "持续修正"]))

    boxes = "\n".join(draw_box(b) for b in BOXES)

    legend_y = 812
    legend = []
    items = [("过程状态", FILL_NORMAL, STROKE_NORMAL),
             ("失败态（需重新录制）", FILL_BAD, STROKE_BAD),
             ("正式完成态", FILL_OK, STROKE_OK)]
    x = 512
    for text, fill, stroke in items:
        legend.append(f'<rect x="{x}" y="{legend_y - 12}" width="26" height="15" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
        legend.append(f'<text x="{x + 36}" y="{legend_y}" font-family="{FONT}" font-size="17" '
                      f'fill="{C_SUB}">{esc(text)}</text>')
        x += 40 + 26 + len(text) * 17

    caption = ('图　校准阶段状态图：中立位校准 → 功能校准 → 个人基线（暂定 / 正式）')

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M 0 1 L 9 5 L 0 9 z" fill="{C_LINE}"/>
</marker>
</defs>
<rect width="{W}" height="{H}" fill="#ffffff"/>
{chr(10).join(p)}
{boxes}
{chr(10).join(lab)}
{chr(10).join(legend)}
<text x="{W / 2}" y="856" font-family="{FONT}" font-size="23" fill="{C_TEXT}" text-anchor="middle">{esc(caption)}</text>
</svg>
'''


def main():
    out = Path(__file__).resolve().parent / "calibration_state_machine.svg"
    out.write_text(build(), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
