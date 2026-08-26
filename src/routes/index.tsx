import { createFileRoute, Link } from "@tanstack/react-router";
import { BatteryMedium, Moon, Thermometer, Waves } from "lucide-react";

import { IPStage, IPWhisper } from "@/components/ip-anan";
import { MobileShell } from "@/components/mobile-shell";
import { alerts, severityStyles, todayStats } from "@/lib/wrist-data";


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
    ],
  }),
  component: Index,
});

function Ring({ value, max }: { value: number; max: number }) {
  const pct = Math.min(100, Math.round((value / max) * 100));
  return (
    <div
      className="relative grid h-28 w-28 shrink-0 place-items-center rounded-full"
      style={{
        background: `conic-gradient(var(--sky) ${pct * 3.6}deg, var(--mist) 0deg)`,
      }}
    >
      <div className="grid h-[5.5rem] w-[5.5rem] place-items-center rounded-full bg-card">
        <span className="font-display text-2xl">{pct}%</span>
        <span className="text-[0.65rem] text-muted-foreground">佩戴完成</span>
      </div>
    </div>
  );
}

function Index() {
  const latest = alerts[0]!;
  const s = severityStyles[latest.severity];

  return (
    <MobileShell title="早安，今天也慢一点" subtitle={`护腕已同步 · ${todayStats.syncedAt}`}>
      <section className="card-soft overflow-hidden">
        <IPStage line="我先替你看着手腕，今天慢慢来就好。" />
        <div className="flex items-center gap-4 p-5">
          <Ring value={todayStats.wearMinutes} max={todayStats.wearGoal} />
          <div className="min-w-0 space-y-1">
            <p className="text-sm">
              今日佩戴 <span className="font-display text-lg">{Math.floor(todayStats.wearMinutes / 60)}</span> 小时
              {todayStats.wearMinutes % 60} 分
            </p>
            <p className="text-xs leading-relaxed text-muted-foreground">
              劳损指数 {todayStats.strainIndex}／100，处于温和区间。疼痛自评 {todayStats.painScore}，比昨天更轻一些。
            </p>
          </div>
        </div>
      </section>


      <section className="grid grid-cols-2 gap-4">
        <MetricCard
          icon={<Waves className="h-4 w-4" strokeWidth={1.6} />}
          label="重复动作"
          value={`${todayStats.repetitiveActions}`}
          hint={`目标 ≤ ${todayStats.actionGoal} 次`}
        />
        <MetricCard
          icon={<Moon className="h-4 w-4" strokeWidth={1.6} />}
          label="休息次数"
          value={`${todayStats.restBreaks}`}
          hint={`建议 ${todayStats.restGoal} 次`}
        />
        <MetricCard
          icon={<Thermometer className="h-4 w-4" strokeWidth={1.6} />}
          label="腕部温度"
          value={`${todayStats.temperature}°C`}
          hint="血流循环良好"
        />
        <MetricCard
          icon={<BatteryMedium className="h-4 w-4" strokeWidth={1.6} />}
          label="设备电量"
          value={`${todayStats.batteryPercent}%`}
          hint="约可再用 2 天"
        />
      </section>

      <section className="card-soft p-5">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="text-base">最新提醒</h2>
          <span className={`rounded-full px-3 py-1 text-[0.7rem] ${s.chip}`}>{s.label}</span>
        </div>
        <p className="text-sm">{latest.title}</p>
        <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{latest.detail}</p>
        <Link
          to="/advice"
          className="mt-4 inline-flex w-full items-center justify-center rounded-2xl gradient-soft px-4 py-3 text-sm text-secondary-foreground"
        >
          现在做一次舒展
        </Link>
      </section>

      <section className="card-soft p-5">
        <IPWhisper text="安安记着你的节奏：今天已经比昨天多休息了两次，手腕会慢慢记得这份温柔。" />
      </section>

    </MobileShell>
  );
}

function MetricCard({
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
    <div className="card-soft p-4">
      <div className="flex items-center gap-2 text-muted-foreground">
        {icon}
        <span className="truncate text-xs">{label}</span>
      </div>
      <p className="mt-2 font-display text-xl">{value}</p>
      <p className="mt-1 text-[0.7rem] text-muted-foreground">{hint}</p>
    </div>
  );
}
