import { createFileRoute } from "@tanstack/react-router";
import { Activity, BellRing, Hand, Radio, Waves } from "lucide-react";
import { useState } from "react";

import { IPIntro, IPWhisper } from "@/components/ip-anan";
import { MobileShell } from "@/components/mobile-shell";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { correctionLog, vibrationStages, todayStats } from "@/lib/wrist-data";

export const Route = createFileRoute("/device")({
  head: () => ({
    meta: [
      { title: "腕安 · 护腕能力设置" },
      {
        name: "description",
        content: "开启实时监控、渐进式震动提醒与自动矫正，让智能护腕温柔地扶住你的手腕。",
      },
      { property: "og:title", content: "腕安 · 护腕能力设置" },
      {
        property: "og:description",
        content: "监控、预警、渐进式震动提醒、自动矫正，四种照顾方式都在这里调节。",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: DevicePage;
});

function DevicePage() {
  const [monitor, setMonitor] = useState(true);
  const [alertOn, setAlertOn] = useState(true);
  const [buzzOn, setBuzzOn] = useState(true);
  const [buzzLevel, setBuzzLevel] = useState([2]);
  const [autoFix, setAutoFix] = useState(true);
  const [fixForce, setFixForce] = useState([45]);
  const [playing, setPlaying] = useState(false);

  const level = buzzLevel[0] ?? 2;
  const stage = vibrationStages[level - 1] ?? vibrationStages[1]!;

  const preview = () => {
    setPlaying(true);
    if (typeof navigator !== "undefined" && "vibrate" in navigator) {
      navigator.vibrate?.(stage.pattern);
    }
    window.setTimeout(() => setPlaying(false), 1600);
  };

  return (
    <MobileShell>
      <IPIntro
        eyebrow="DEVICE · 护腕的四种照顾"
        title={"监控、预警、轻震、\n自动扶一把"}
        line={`护腕已连接 · ${todayStats.syncedAt}同步 · 电量 ${todayStats.batteryPercent}%`}
      />

      {/* 能力总览：柔和的一行光点 */}
      <section className="-mx-5 overflow-x-auto px-5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        <div className="flex w-max gap-2.5">
          {[
            { icon: <Activity className="h-3.5 w-3.5" strokeWidth={1.7} />, t: "实时监控", on: monitor },
            { icon: <BellRing className="h-3.5 w-3.5" strokeWidth={1.7} />, t: "智能预警", on: alertOn },
            { icon: <Waves className="h-3.5 w-3.5" strokeWidth={1.7} />, t: "渐进震动", on: buzzOn },
            { icon: <Hand className="h-3.5 w-3.5" strokeWidth={1.7} />, t: "自动矫正", on: autoFix },
          ].map((c) => (
            <span
              key={c.t}
              className={`orbit-chip flex items-center gap-1.5 rounded-full px-3.5 py-2 text-[0.68rem] ${
                c.on ? "text-foreground" : "text-muted-foreground opacity-70"
              }`}
            >
              <span className={c.on ? "text-primary" : ""}>{c.icon}</span>
              {c.t}
              <span className={`h-1.5 w-1.5 rounded-full ${c.on ? "bg-calm" : "bg-border"}`} />
            </span>
          ))}
        </div>
      </section>

      <div className="hairline mt-1 h-px" />

      {/* 1 监控 */}
      <Row
        no="01"
        title="实时监控"
        line="六轴姿态 + 压力 + 温度，每 5 秒回传一次，安安只在异常时开口。"
        control={<Switch checked={monitor} onCheckedChange={setMonitor} />}
      >
        <div className="mt-3 flex gap-2 text-[0.62rem] text-muted-foreground">
          {["屈伸角度", "重复频次", "佩戴压力", "腕部温度"].map((t) => (
            <span key={t} className="rounded-full bg-secondary/60 px-2.5 py-1">
              {t}
            </span>
          ))}
        </div>
      </Row>

      {/* 2 预警 */}
      <Row
        no="02"
        title="智能预警"
        line="连续动作超过 45 分钟、单日重复次数越界时，先给一次轻声提醒。"
        control={<Switch checked={alertOn} onCheckedChange={setAlertOn} />}
      />

      {/* 3 渐进式震动 */}
      <Row
        no="03"
        title="渐进式震动提醒"
        line="从几乎察觉不到开始，一层层加深，直到你放松手腕就停下。"
        control={<Switch checked={buzzOn} onCheckedChange={setBuzzOn} />}
      >
        <div className={buzzOn ? "" : "pointer-events-none opacity-45"}>
          <div className="mt-4 flex items-end gap-1.5">
            {stage.wave.map((v, i) => (
              <span
                key={i}
                className={`flex-1 rounded-full ${playing ? "breathe" : ""}`}
                style={{
                  height: `${12 + v * 0.42}px`,
                  animationDelay: `${i * 90}ms`,
                  background:
                    "linear-gradient(180deg, var(--primary), color-mix(in oklab, var(--sky) 60%, var(--card)))",
                }}
              />
            ))}
          </div>
          <div className="mt-4 flex items-center gap-3">
            <Slider value={buzzLevel} onValueChange={setBuzzLevel} min={1} max={3} step={1} />
            <span className="w-16 shrink-0 text-right text-[0.68rem] text-muted-foreground">
              {stage.label}
            </span>
          </div>
          <p className="mt-2 text-[0.68rem] leading-relaxed text-muted-foreground">{stage.desc}</p>
          <button
            type="button"
            onClick={preview}
            className="mt-3 inline-flex items-center gap-1.5 rounded-full gradient-soft px-4 py-2.5 text-xs text-secondary-foreground"
          >
            <Radio className="h-3.5 w-3.5" strokeWidth={1.8} />
            {playing ? "正在轻轻震动…" : "试一次这层力度"}
          </button>
        </div>
      </Row>

      {/* 4 自动矫正 */}
      <Row
        no="04"
        title="自动矫正"
        line="检测到腕部下垂或过度背屈时，气囊缓慢充压把手腕托回中立位。"
        control={<Switch checked={autoFix} onCheckedChange={setAutoFix} />}
      >
        <div className={autoFix ? "" : "pointer-events-none opacity-45"}>
          <div className="mt-4 flex items-center gap-3">
            <Slider value={fixForce} onValueChange={setFixForce} min={20} max={80} step={5} />
            <span className="w-16 shrink-0 text-right text-[0.68rem] text-muted-foreground">
              支撑 {fixForce[0]}%
            </span>
          </div>
          <div className="mt-4 space-y-2.5">
            {correctionLog.map((c) => (
              <div key={c.time} className="flex items-start gap-2.5">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-calm" />
                <p className="text-[0.7rem] leading-relaxed text-muted-foreground">
                  <span className="text-foreground">{c.time}</span> {c.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      </Row>

      <section className="pt-1">
        <IPWhisper text="所有震动都从最轻的一层开始，如果你没反应，安安才会再稍微用力一点。" />
      </section>
    </MobileShell>
  );
}

function Row({
  no,
  title,
  line,
  control,
  children,
}: {
  no: string;
  title: string;
  line: string;
  control: React.ReactNode;
  children?: React.ReactNode;
}) {
  return (
    <section className="border-b border-border/70 py-5 last:border-0">
      <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-4">
        <p className="w-8 font-display text-xl leading-none text-muted-foreground/70">{no}</p>
        <div>
          <p className="font-display text-lg leading-snug">{title}</p>
          <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{line}</p>
        </div>
        <div className="pt-1">{control}</div>
      </div>
      {children ? <div className="pl-12">{children}</div> : null}
    </section>
  );
}
