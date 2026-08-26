import { createFileRoute } from "@tanstack/react-router";

import { IPIntro } from "@/components/ip-anan";
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
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: AlertsPage,
});

const rules = [
  "连续重复动作超过 45 分钟",
  "单日重复动作次数超过 5000 次",
  "佩戴压力低于设定范围（可能松动）",
  "腕部温度低于 31°C 或高于 36°C",
];

function AlertsPage() {
  const counts = {
    warn: alerts.filter((a) => a.severity === "warn").length,
    notice: alerts.filter((a) => a.severity === "notice").length,
    calm: alerts.filter((a) => a.severity === "calm").length,
  };

  return (
    <MobileShell>
      <IPIntro
        eyebrow="ALERTS · 智能预警"
        title="只在需要的时候，轻轻叫你"
        line={`今天 ${counts.warn} 条预警、${counts.notice} 条留意，整体节奏比上周更平稳。`}
      />

      {/* 概览：三个呼吸点，取代卡片 */}
      <section className="flex items-end justify-between px-1">
        {(
          [
            ["warn", counts.warn],
            ["notice", counts.notice],
            ["calm", counts.calm],
          ] as const
        ).map(([key, n]) => (
          <div key={key} className="flex items-center gap-2.5">
            <span className={`h-2.5 w-2.5 rounded-full ${severityStyles[key].dot}`} />
            <div>
              <p className="font-display text-xl leading-none">{n}</p>
              <p className="mt-1 text-[0.65rem] text-muted-foreground">
                {severityStyles[key].label}
              </p>
            </div>
          </div>
        ))}
      </section>

      {/* 时间轴：一条细线串起所有提醒，去掉方块 */}
      <section className="relative pl-6">
        <span className="absolute left-[0.3rem] top-2 bottom-6 w-px bg-border" />
        <div className="space-y-7">
          {alerts.map((a) => {
            const s = severityStyles[a.severity];
            return (
              <article key={a.id} className="relative">
                <span
                  className={`absolute -left-[1.42rem] top-1.5 h-2.5 w-2.5 rounded-full ring-4 ring-background ${s.dot}`}
                />
                <div className="flex items-baseline gap-2">
                  <h2 className="font-display text-base leading-snug">{a.title}</h2>
                  <span className="shrink-0 text-[0.62rem] tracking-[0.16em] text-muted-foreground">
                    {s.label}
                  </span>
                </div>
                <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{a.detail}</p>
                <p className="mt-2 text-[0.65rem] text-muted-foreground/80">{a.time}</p>
              </article>
            );
          })}
        </div>
      </section>

      {/* 规则：横向柔和标签流 */}
      <section className="pt-2">
        <div className="hairline h-px" />
        <p className="mt-4 text-[0.62rem] tracking-[0.3em] text-muted-foreground">安安会在这些时刻开口</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {rules.map((r) => (
            <span
              key={r}
              className="orbit-chip rounded-full px-3.5 py-2 text-[0.7rem] text-muted-foreground"
            >
              {r}
            </span>
          ))}
        </div>
      </section>
    </MobileShell>
  );
}
