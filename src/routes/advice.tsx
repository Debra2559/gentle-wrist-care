import { createFileRoute } from "@tanstack/react-router";
import { Check, Clock3 } from "lucide-react";
import { useState } from "react";

import { IPIntro, IPAvatar } from "@/components/ip-anan";
import { MobileShell } from "@/components/mobile-shell";
import { suggestions } from "@/lib/wrist-data";

export const Route = createFileRoute("/advice")({
  head: () => ({
    meta: [
      { title: "康复建议 · 腕安智能护腕" },
      {
        name: "description",
        content: "根据护腕数据生成的腕部舒展、工作节奏与夜间护理建议。",
      },
      { property: "og:title", content: "康复建议 · 腕安智能护腕" },
      { property: "og:description", content: "舒展、节奏、夜间护理，一点一点把手腕养回来。" },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: AdvicePage,
});

function AdvicePage() {
  const [done, setDone] = useState<Record<string, boolean>>({});
  const finished = Object.values(done).filter(Boolean).length;

  return (
    <MobileShell>
      <IPIntro
        eyebrow="CARE · 今天的照顾方式"
        title="四件小事，慢慢来就好"
        line={`已完成 ${finished} / ${suggestions.length}，安安会一直陪着你。`}
      />

      {/* 进度：一条柔和的光带，取代进度卡片 */}
      <section className="px-1">
        <div className="flex gap-1.5">
          {suggestions.map((s) => (
            <span
              key={s.id}
              className="h-1 flex-1 rounded-full transition-colors"
              style={{
                background: done[s.id]
                  ? "var(--primary)"
                  : "color-mix(in oklab, var(--border) 90%, transparent)",
              }}
            />
          ))}
        </div>
      </section>

      {/* 仪式清单：以序号与细线分隔，去掉层层方块 */}
      <section className="divide-y divide-border/70">
        {suggestions.map((s, i) => {
          const isDone = !!done[s.id];
          return (
            <article key={s.id} className="py-5 first:pt-1">
              <div className="grid grid-cols-[auto_minmax(0,1fr)] gap-4">
                <div className="pt-1 text-center">
                  <p className="font-display text-xl leading-none text-muted-foreground/70">
                    0{i + 1}
                  </p>
                  <p className="mt-2 text-[0.6rem] tracking-[0.16em] text-blush-foreground">
                    {s.tag}
                  </p>
                </div>
                <div className={isDone ? "opacity-55 transition-opacity" : "transition-opacity"}>
                  <div className="flex items-baseline justify-between gap-3">
                    <h2 className="font-display text-base leading-snug">{s.title}</h2>
                    <span className="flex shrink-0 items-center gap-1 text-[0.65rem] text-muted-foreground">
                      <Clock3 className="h-3.5 w-3.5" strokeWidth={1.6} />
                      {s.minutes} 分钟
                    </span>
                  </div>
                  <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{s.body}</p>
                  <button
                    type="button"
                    onClick={() => setDone((d) => ({ ...d, [s.id]: !d[s.id] }))}
                    className="mt-3 inline-flex items-center gap-2 rounded-full px-3.5 py-2 text-[0.7rem] transition-colors"
                    style={{
                      background: isDone
                        ? "color-mix(in oklab, var(--calm) 22%, transparent)"
                        : "color-mix(in oklab, var(--card) 70%, transparent)",
                      border: "1px solid color-mix(in oklab, var(--border) 70%, transparent)",
                    }}
                  >
                    <span
                      className="grid h-4 w-4 place-items-center rounded-full"
                      style={{
                        background: isDone ? "var(--calm)" : "transparent",
                        border: isDone ? "none" : "1px solid var(--border)",
                      }}
                    >
                      {isDone ? <Check className="h-2.5 w-2.5" strokeWidth={3} /> : null}
                    </span>
                    {isDone ? "已完成" : "标记完成"}
                  </button>
                </div>
              </div>
            </article>
          );
        })}
      </section>

      {/* 温柔提示：安安的落款 */}
      <section className="relative pt-2">
        <div className="hairline h-px" />
        <div className="mt-4 flex items-start gap-3">
          <IPAvatar className="h-12 w-12" />
          <div>
            <p className="text-[0.62rem] tracking-[0.3em] text-muted-foreground">安安的悄悄话</p>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
              这些建议来自护腕数据与常见康复方法，不能替代医生诊断。如果疼痛持续超过两周、出现明显肿胀或夜间痛醒，请及时就医。
            </p>
          </div>
        </div>
      </section>
    </MobileShell>
  );
}
