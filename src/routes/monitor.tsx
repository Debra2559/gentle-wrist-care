import { createFileRoute } from "@tanstack/react-router";

import { MobileShell } from "@/components/mobile-shell";
import { hourlyStrain, recoveryNotes, weekTrend } from "@/lib/wrist-data";

export const Route = createFileRoute("/monitor")({
  head: () => ({
    meta: [
      { title: "数据监测 · 腕安智能护腕" },
      {
        name: "description",
        content: "护腕上传的一周劳损趋势、逐时负荷分布与康复变化记录。",
      },
      { property: "og:title", content: "数据监测 · 腕安智能护腕" },
      { property: "og:description", content: "一周劳损趋势与逐时负荷，看见手腕的节奏。" },
    ],
  }),
  component: MonitorPage,
});

function MonitorPage() {
  const maxHour = Math.max(...hourlyStrain.map((h) => h.value));

  return (
    <MobileShell title="数据监测" subtitle="护腕每 5 分钟自动上传一次">
      <section className="card-soft p-5">
        <h2 className="text-base">一周劳损趋势</h2>
        <p className="mt-1 text-xs text-muted-foreground">数值越低，手腕越轻松</p>
        <div className="mt-5 flex items-end justify-between gap-2">
          {weekTrend.map((d) => (
            <div key={d.day} className="flex min-w-0 flex-1 flex-col items-center gap-2">
              <div className="flex h-32 w-full items-end justify-center">
                <div
                  className="w-full max-w-6 rounded-t-full gradient-soft"
                  style={{ height: `${d.strain}%` }}
                />
              </div>
              <span className="text-[0.65rem] text-muted-foreground">{d.day.slice(1)}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="card-soft p-5">
        <h2 className="text-base">今日逐时负荷</h2>
        <div className="mt-4 space-y-3">
          {hourlyStrain.map((h) => (
            <div key={h.hour} className="flex items-center gap-3">
              <span className="w-10 shrink-0 text-xs text-muted-foreground">{h.hour}:00</span>
              <div className="h-2.5 min-w-0 flex-1 overflow-hidden rounded-full bg-mist">
                <div
                  className="h-full rounded-full bg-sky"
                  style={{ width: `${(h.value / maxHour) * 100}%` }}
                />
              </div>
              <span className="w-8 shrink-0 text-right text-xs">{h.value}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="card-soft p-5">
        <h2 className="text-base">恢复记录</h2>
        <div className="mt-4 space-y-4">
          {recoveryNotes.map((n) => (
            <div key={n.date} className="border-l-2 border-blush pl-4">
              <p className="text-xs text-muted-foreground">{n.date}</p>
              <p className="mt-1 text-sm leading-relaxed">{n.text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="card-soft p-5">
        <h2 className="text-base">同步说明</h2>
        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
          护腕内置压力、角度与温度传感器，通过蓝牙低功耗把数据传到手机；离线时会本地缓存
          7 天，重新连接后自动补传。
        </p>
      </section>
    </MobileShell>
  );
}
