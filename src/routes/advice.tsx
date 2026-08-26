import { createFileRoute } from "@tanstack/react-router";
import { Clock3 } from "lucide-react";

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
    ],
  }),
  component: AdvicePage,
});

function AdvicePage() {
  return (
    <MobileShell title="今天的照顾方式" subtitle="根据你的数据，为你挑了 4 件小事">
      <div className="space-y-4">
        {suggestions.map((s) => (
          <article key={s.id} className="card-soft p-5">
            <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
              <span className="w-fit rounded-full bg-blush/40 px-3 py-1 text-[0.7rem] text-blush-foreground">
                {s.tag}
              </span>
              <span className="flex shrink-0 items-center gap-1 text-[0.7rem] text-muted-foreground">
                <Clock3 className="h-3.5 w-3.5" strokeWidth={1.6} />
                {s.minutes} 分钟
              </span>
            </div>
            <h2 className="mt-3 text-base">{s.title}</h2>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{s.body}</p>
            <button
              type="button"
              className="mt-4 w-full rounded-2xl border border-border bg-muted/60 px-4 py-2.5 text-sm transition-colors hover:bg-secondary"
            >
              标记为已完成
            </button>
          </article>
        ))}
      </div>

      <section className="card-soft p-5">
        <h2 className="text-base">温柔提示</h2>
        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
          这些建议来自护腕数据与常见康复方法，不能替代医生诊断。如果疼痛持续超过两周、出现明显肿胀或夜间痛醒，请及时就医。
        </p>
      </section>
    </MobileShell>
  );
}
