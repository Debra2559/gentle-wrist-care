import { createFileRoute } from "@tanstack/react-router";

import { MobileShell } from "@/components/mobile-shell";
import { alerts, severityStyles } from "@/lib/wrist-data";

export const Route = createFileRoute("/alerts")({
  head: () => ({
    meta: [
      { title: "智能预警 · 腕安智能护腕" },
      {
        name: "description",
        content: "护腕在手腕负荷偏高、佩戴松动或温度异常时，用温柔的方式提醒你。",
      },
      { property: "og:title", content: "智能预警 · 腕安智能护腕" },
      { property: "og:description", content: "负荷偏高、佩戴松动、温度异常，都会被温柔地提醒。" },
    ],
  }),
  component: AlertsPage,
});

function AlertsPage() {
  return (
    <MobileShell title="智能预警" subtitle="只在需要的时候轻轻提醒你">
      <section className="card-soft p-5">
        <p className="text-sm leading-relaxed">
          今天共 <span className="font-display text-lg">1</span> 条预警、2 条留意提示。整体节奏比上周更平稳。
        </p>
      </section>

      <div className="space-y-4">
        {alerts.map((a) => {
          const s = severityStyles[a.severity];
          return (
            <article key={a.id} className="card-soft p-5">
              <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  <span className={`h-2 w-2 shrink-0 rounded-full ${s.dot}`} />
                  <h2 className="truncate text-base">{a.title}</h2>
                </div>
                <span className={`shrink-0 rounded-full px-3 py-1 text-[0.7rem] ${s.chip}`}>
                  {s.label}
                </span>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{a.detail}</p>
              <p className="mt-3 text-[0.7rem] text-muted-foreground">{a.time}</p>
            </article>
          );
        })}
      </div>

      <section className="card-soft p-5">
        <h2 className="text-base">预警规则</h2>
        <ul className="mt-3 space-y-2 text-xs leading-relaxed text-muted-foreground">
          <li>· 连续重复动作超过 45 分钟</li>
          <li>· 单日重复动作次数超过 5000 次</li>
          <li>· 佩戴压力低于设定范围（可能松动）</li>
          <li>· 腕部温度低于 31°C 或高于 36°C</li>
        </ul>
      </section>
    </MobileShell>
  );
}
