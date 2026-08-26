import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowUpRight, BatteryMedium, Moon, Thermometer, Waves } from "lucide-react";

import { IPHero, IPWhisper } from "@/components/ip-anan";
import { MobileShell } from "@/components/mobile-shell";
import { alerts, severityStyles, hourlyStrain, todayStats } from "@/lib/wrist-data";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "腕安 · 今日手腕状态" },
      {
        name: "description",
        content: "查看今日护腕佩戴时长、重复动作次数、劳损指数与温柔的康复提醒。",
      },
      { property: "og:title", content: "腕安 · 今日手腕状态" },
      {
        property: "og:description",
        content: "智能护腕实时同步数据，用温柔的方式陪你照顾手腕。",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  const latest = alerts[0]!;
  const s = severityStyles[latest.severity];
  const pct = Math.min(100, Math.round((todayStats.wearMinutes / todayStats.wearGoal) * 100));
  const hours = Math.floor(todayStats.wearMinutes / 60);
  const mins = todayStats.wearMinutes % 60;
  const peak = hourlyStrain.reduce((a, b) => (b.value > a.value ? b : a), hourlyStrain[0]!);

  return (
    <MobileShell>
      <IPHero
        greeting="早安，今天也慢一点"
        line="安安先替你看着手腕。"
        ringPct={pct}
        wearText={`${hours}h${mins}m`}
        syncedAt={todayStats.syncedAt}
      />

      {/* 劳损带：一条呼吸式的横向光带，代替方块 */}
      <section className="-mt-1">
        <div className="flex items-baseline justify-between">
          <p className="font-display text-base">劳损指数 {todayStats.strainIndex}</p>
          <p className="text-[0.7rem] text-muted-foreground">温和区间 · 峰值 {peak.hour}:00</p>
        </div>
        <div className="mt-3 flex items-end gap-1.5">
          {hourlyStrain.map((h) => (
            <div key={h.hour} className="flex flex-1 flex-col items-center gap-1.5">
              <div className="flex h-14 w-full items-end">
                <div
                  className="w-full rounded-full"
                  style={{
                    height: `${Math.max(14, h.value)}%`,
                    background:
                      h.value > 60
                        ? "color-mix(in oklab, var(--warn) 70%, transparent)"
                        : "color-mix(in oklab, var(--sky) 70%, transparent)",
                  }}
                />
              </div>
              <span className="text-[0.55rem] text-muted-foreground">{h.hour}</span>
            </div>
          ))}
        </div>
        <div className="hairline mt-3 h-px" />
      </section>

      {/* 数据以柔和横向卷轴呈现，而非四宫格 */}
      <section className="-mx-5 overflow-x-auto px-5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        <div className="flex w-max gap-3 pb-1">
          <Stat
            icon={<Waves className="h-4 w-4" strokeWidth={1.6} />}
            label="重复动作"
            value={`${todayStats.repetitiveActions}`}
            hint={`≤ ${todayStats.actionGoal}`}
          />
          <Stat
            icon={<Moon className="h-4 w-4" strokeWidth={1.6} />}
            label="休息次数"
            value={`${todayStats.restBreaks}`}
            hint={`建议 ${todayStats.restGoal}`}
          />
          <Stat
            icon={<Thermometer className="h-4 w-4" strokeWidth={1.6} />}
            label="腕部温度"
            value={`${todayStats.temperature}°`}
            hint="循环良好"
          />
          <Stat
            icon={<BatteryMedium className="h-4 w-4" strokeWidth={1.6} />}
            label="设备电量"
            value={`${todayStats.batteryPercent}%`}
            hint="约 2 天"
          />
        </div>
      </section>

      {/* 最新提醒：安安口吻的一封小笺，非规整卡片 */}
      <section className="relative">
        <div className="card-soft relative rounded-[2rem] rounded-tl-md p-5">
          <div className="mb-2 flex items-center gap-2">
            <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} />
            <span className="text-[0.68rem] tracking-[0.2em] text-muted-foreground">
              {s.label} · {latest.time}
            </span>
          </div>
          <p className="font-display text-lg leading-snug">{latest.title}</p>
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{latest.detail}</p>
          <Link
            to="/advice"
            className="mt-4 inline-flex items-center gap-1.5 rounded-full gradient-soft px-4 py-2.5 text-xs text-secondary-foreground"
          >
            现在做一次舒展
            <ArrowUpRight className="h-3.5 w-3.5" strokeWidth={1.8} />
          </Link>
        </div>
      </section>

      <section className="px-1">
        <IPWhisper text="安安记着你的节奏：今天已经比昨天多休息了两次，手腕会慢慢记得这份温柔。" />
      </section>

      <div className="flex items-center justify-center gap-2 pt-1 text-[0.62rem] tracking-[0.3em] text-muted-foreground">
        <span className="h-px w-8 bg-border" />
        与安安同行
        <span className="h-px w-8 bg-border" />
      </div>
    </MobileShell>
  );
}

function Stat({
  icon,
  label,
  value,
  hint,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="orbit-chip w-[6.6rem] rounded-[1.6rem] px-3.5 py-3">
      <div className="text-muted-foreground">{icon}</div>
      <p className="mt-2 font-display text-lg leading-none">{value}</p>
      <p className="mt-1.5 text-[0.65rem]">{label}</p>
      <p className="text-[0.6rem] text-muted-foreground">{hint}</p>
    </div>
  );
}
