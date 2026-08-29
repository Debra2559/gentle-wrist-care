import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { useServerFn } from "@tanstack/react-start";
import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

import { IPIntro, IPWhisper } from "@/components/ip-anan";
import { MobileShell } from "@/components/mobile-shell";
import { getSessionReport } from "@/lib/shewrist.functions";
import type { SessionReport, TimelineRow } from "@/lib/shewrist-types";

const POLL_MS = 5000;

export const Route = createFileRoute("/report")({
  head: () => ({
    meta: [
      { title: "算法报告 · 腕安智能护腕" },
      {
        name: "description",
        content: "接入 SheWrist 分析后端，展示腕部暴露剂量、高暴露占比、预警与解释摘要。",
      },
      { property: "og:title", content: "算法报告 · 腕安智能护腕" },
      { property: "og:description", content: "会话级腕部暴露分析结果与逐时角度轨迹。" },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: ReportPage,
});

const zoneColor: Record<string, string> = {
  green: "var(--calm)",
  yellow: "var(--warn)",
  red: "var(--destructive)",
};

function fmt(v: number | null | undefined, unit = "", digits = 1) {
  if (v === null || v === undefined) return "—";
  return `${Number(v).toFixed(digits)}${unit}`;
}

function ReportPage() {
  const fetchReport = useServerFn(getSessionReport);
  const [sessionId, setSessionId] = useState("S001");
  const [committed, setCommitted] = useState("S001");
  const [polling, setPolling] = useState(true);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  const { data, isFetching, isLoading, error, refetch, dataUpdatedAt } =
    useQuery<SessionReport>({
      queryKey: ["shewrist-session", committed],
      queryFn: () => fetchReport({ data: { sessionId: committed } }),
      refetchInterval: polling ? POLL_MS : false,
      refetchIntervalInBackground: false,
      retry: 2,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
      placeholderData: (prev) => prev,
    });

  useEffect(() => {
    if (dataUpdatedAt) setUpdatedAt(new Date(dataUpdatedAt));
  }, [dataUpdatedAt]);

  return (
    <MobileShell>
      <IPIntro
        eyebrow="SHEWRIST ANALYSIS"
        title="这次会话，手腕经历了什么"
        line="安安把算法后端的暴露剂量、预警与解释，翻译成你看得懂的一页。"
      />

      <form
        className="flex items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          setCommitted(sessionId.trim() || "S001");
        }}
      >
        <input
          value={sessionId}
          onChange={(e) => setSessionId(e.target.value)}
          aria-label="会话 ID"
          placeholder="会话 ID，例如 S001"
          className="min-w-0 flex-1 rounded-full border border-border bg-card/70 px-4 py-2.5 text-xs outline-none placeholder:text-muted-foreground focus:border-sky"
        />
        <button
          type="submit"
          className="shrink-0 rounded-full gradient-soft px-4 py-2.5 text-xs text-secondary-foreground"
        >
          {isFetching ? "读取中" : "读取"}
        </button>
      </form>

      {/* 实时刷新控制条 */}
      <section className="flex items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => setPolling((p) => !p)}
          aria-pressed={polling}
          className="orbit-chip flex items-center gap-2 rounded-full px-3 py-1.5 text-[0.65rem]"
        >
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{
              background: polling ? "var(--calm)" : "var(--muted-foreground)",
              animation: polling ? "breathe 2.2s ease-in-out infinite" : undefined,
            }}
          />
          {polling ? "实时刷新中 · 5s" : "已暂停刷新"}
        </button>
        <div className="flex items-center gap-2 text-[0.62rem] text-muted-foreground">
          {updatedAt ? (
            <span>
              更新于{" "}
              {updatedAt.toLocaleTimeString("zh-CN", { hour12: false })}
            </span>
          ) : null}
          <button
            type="button"
            aria-label="立即刷新"
            onClick={() => refetch()}
            className="orbit-chip rounded-full p-1.5"
          >
            <RefreshCw
              className={`h-3 w-3 ${isFetching ? "animate-spin" : ""}`}
              style={{ animationDuration: "1.2s" }}
            />
          </button>
        </div>
      </section>

      {isLoading ? (
        <ReportSkeleton />
      ) : error ? (
        <ReportError onRetry={() => refetch()} />
      ) : !data ? (
        <ReportEmpty onRetry={() => refetch()} />
      ) : (
        <ReportBody report={data} />
      )}
    </MobileShell>
  );
}

function ReportSkeleton() {
  return (
    <section aria-busy="true" aria-label="正在加载" className="space-y-6">
      <p className="text-xs text-muted-foreground">安安正在连接分析后端…</p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="space-y-2">
            <div className="h-5 w-16 animate-pulse rounded-full bg-muted/60" />
            <div className="h-2.5 w-12 animate-pulse rounded-full bg-muted/40" />
          </div>
        ))}
      </div>
      <div className="h-24 w-full animate-pulse rounded-3xl bg-muted/40" />
      <div className="space-y-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="h-3 w-full animate-pulse rounded-full bg-muted/40" />
        ))}
      </div>
    </section>
  );
}

function ReportError({ onRetry }: { onRetry: () => void }) {
  return (
    <section className="flex flex-col items-start gap-3">
      <p className="font-display text-base">这次没有连上</p>
      <p className="text-xs leading-relaxed text-muted-foreground">
        网络轻轻抖了一下，数据没有传过来。休息一下再试试，安安在这儿等你。
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="rounded-full gradient-soft px-5 py-2.5 text-xs text-secondary-foreground"
      >
        重新连接
      </button>
    </section>
  );
}

function ReportEmpty({ onRetry }: { onRetry: () => void }) {
  return (
    <section className="flex flex-col items-start gap-3">
      <p className="font-display text-base">还没有这次会话的数据</p>
      <p className="text-xs leading-relaxed text-muted-foreground">
        这个会话 ID 暂时没有回传任何结果。确认护腕已完成上传，或换一个会话试试。
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="rounded-full gradient-soft px-5 py-2.5 text-xs text-secondary-foreground"
      >
        再查一次
      </button>
    </section>
  );
}

function GateNotice({ reasons, warnings }: { reasons: string[]; warnings: string[] }) {
  return (
    <section
      role="status"
      className="rounded-3xl border px-4 py-4"
      style={{
        borderColor: "color-mix(in oklab, var(--warn) 45%, transparent)",
        background: "color-mix(in oklab, var(--warn) 10%, transparent)",
      }}
    >
      <p className="font-display text-sm">本次数据未通过质量门控</p>
      <p className="mt-1.5 text-[0.7rem] leading-relaxed text-muted-foreground">
        下面的指标仅供参考，不能作为暴露评估依据。常见原因是佩戴时间太短或同步误差偏大。
      </p>
      {reasons.length ? (
        <ul className="mt-2.5 space-y-1.5">
          {reasons.map((r) => (
            <li key={r} className="flex gap-2 text-[0.7rem] leading-relaxed">
              <span
                className="mt-1.5 h-1 w-1 shrink-0 rounded-full"
                style={{ background: "var(--warn)" }}
              />
              {r}
            </li>
          ))}
        </ul>
      ) : null}
      {warnings.map((w) => (
        <p key={w} className="mt-2 text-[0.65rem] text-muted-foreground">
          {w}
        </p>
      ))}
    </section>
  );
}

function ReportBody({ report }: { report: SessionReport }) {
  const { result, timeline } = report;
  const m = result.metrics;
  const accepted = result.analysis_status === "accepted";

  return (
    <>
      <section className="flex flex-wrap items-center gap-2">
        <span
          className="orbit-chip rounded-full px-3 py-1.5 text-[0.65rem]"
          style={{ color: accepted ? undefined : "var(--destructive)" }}
        >
          {accepted ? "分析已接受" : "门控未通过"} · {result.analysis_status}
        </span>
        <span className="orbit-chip rounded-full px-3 py-1.5 text-[0.65rem]">
          {report.source === "live" ? "实时接口" : "演示数据"}
        </span>
        <span className="orbit-chip rounded-full px-3 py-1.5 text-[0.65rem]">
          {result.evidence_type} · {result.algorithm_release}
        </span>
      </section>

      {report.note ? <p className="text-[0.68rem] text-muted-foreground">{report.note}</p> : null}

      {/* 核心指标 */}
      <section>
        <p className="font-display text-base">暴露概览</p>
        <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-4">
          <Metric label="高暴露占比" value={fmt(m.high_posture_time_pct, "%")} />
          <Metric label="最长连续高暴露" value={fmt(m.longest_high_posture_s, " s")} />
          <Metric label="总超量剂量" value={fmt(m.total_excess_dose_deg_s, " °·s", 0)} />
          <Metric label="屈伸 / 桡尺偏剂量" value={`${fmt(m.fe_excess_dose_deg_s, "", 0)} / ${fmt(m.rud_excess_dose_deg_s, "", 0)}`} />
          <Metric label="屈伸峰值角" value={fmt(m.max_abs_fe_deg, "°")} />
          <Metric label="桡尺偏峰值角" value={fmt(m.max_abs_rud_deg, "°")} />
          <Metric label="屈伸循环" value={fmt(m.fe_cycles_per_min, " 次/分")} />
          <Metric label="压力峰值" value={fmt(m.max_pressure_kpa, " kPa")} />
        </div>
        <div className="hairline mt-4 h-px" />
      </section>

      {/* 角度轨迹 */}
      <section>
        <div className="flex items-baseline justify-between">
          <p className="font-display text-base">腕角轨迹</p>
          <p className="text-[0.68rem] text-muted-foreground">
            {timeline.total} 样本 · {fmt(result.data_quality.sample_rate_hz, " Hz", 0)}
          </p>
        </div>
        <AngleTrack rows={timeline.items} />
        <div className="mt-3 flex items-center gap-3 text-[0.62rem] text-muted-foreground">
          {(["green", "yellow", "red"] as const).map((z) => (
            <span key={z} className="flex items-center gap-1.5">
              <span
                className="h-1.5 w-1.5 rounded-full"
                style={{ background: zoneColor[z] }}
              />
              {z === "green" ? "安全区" : z === "yellow" ? "黄区" : "红区"}
            </span>
          ))}
        </div>
        <div className="hairline mt-4 h-px" />
      </section>

      {/* 预警 */}
      <section>
        <p className="font-display text-base">接口回传的预警（{result.alerts.length}）</p>
        <div className="mt-3 divide-y divide-border/70">
          {result.alerts.map((a) => (
            <div key={`${a.timestamp_ms}-${a.reason}`} className="flex items-start gap-3 py-3.5">
              <span
                className="mt-1.5 h-2 w-2 shrink-0 rounded-full"
                style={{ background: zoneColor[a.zone] ?? "var(--sky)" }}
              />
              <div className="min-w-0">
                <p className="text-xs">{reasonLabel[a.reason] ?? a.reason}</p>
                <p className="mt-1 text-[0.65rem] text-muted-foreground">
                  {msToClock(a.timestamp_ms)} · {a.zone}
                  {a.recommend_mechanical ? " · 建议机械支撑" : ""}
                  {a.safety_stop ? " · 安全停止" : ""}
                </p>
              </div>
            </div>
          ))}
          {result.alerts.length === 0 ? (
            <p className="py-3 text-xs text-muted-foreground">本次会话没有触发预警。</p>
          ) : null}
        </div>
        <div className="hairline mt-1 h-px" />
      </section>

      {/* 解释摘要 */}
      <section>
        <p className="font-display text-base">算法解释</p>
        <p className="mt-2 text-xs leading-relaxed">{result.explanation.summary}</p>
        <ul className="mt-3 space-y-2">
          {result.explanation.observations.map((o) => (
            <li key={o} className="flex gap-2 text-[0.72rem] leading-relaxed text-muted-foreground">
              <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-sky" />
              {o}
            </li>
          ))}
        </ul>
        {result.explanation.next_steps.length ? (
          <div className="mt-4 flex flex-wrap gap-2">
            {result.explanation.next_steps.map((n) => (
              <span key={n} className="orbit-chip rounded-full px-3 py-1.5 text-[0.65rem]">
                {n}
              </span>
            ))}
          </div>
        ) : null}
        <div className="hairline mt-4 h-px" />
      </section>

      {/* 数据质量与通道 */}
      <section>
        <p className="font-display text-base">数据质量</p>
        <div className="mt-3 space-y-2 text-[0.7rem] text-muted-foreground">
          <Row
            k="有效样本率"
            v={`${fmt(result.data_quality.valid_sample_pct, "%")}（门限 ${fmt(result.data_quality.valid_sample_pct_min, "%", 0)}，${result.data_quality.valid_sample_gate_passed ? "通过" : "未通过"}）`}
          />
          <Row
            k="同步误差 p95 / max"
            v={`${fmt(result.data_quality.p95_sync_error_ms, " ms")} / ${fmt(result.data_quality.max_sync_error_ms, " ms")}`}
          />
          <Row k="样本数" v={`${result.data_quality.sample_count}`} />
          <Row
            k="影子模型"
            v={`${result.ml_shadow.operating_mode} · ${result.ml_shadow.accepted_window_count}/${result.ml_shadow.window_count} 窗口 · 控制权限 ${result.ml_shadow.safety_effect}`}
          />
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          {Object.entries(result.channels).map(([key, c]) => (
            <span
              key={key}
              className="orbit-chip rounded-full px-3 py-1.5 text-[0.62rem]"
              style={{ opacity: c.available ? 1 : 0.45 }}
            >
              {channelLabel[key] ?? key} {c.available ? "✓" : "—"}
            </span>
          ))}
        </div>
      </section>

      <IPWhisper
        text={`${result.evidence_limits[0] ?? "仅用于工程原型与工效暴露研究。"} 15°/20°/30°/4.4 kPa 都是工程筛查参数，不是诊断阈值。`}
      />
    </>
  );
}

const reasonLabel: Record<string, string> = {
  sustained_high_posture: "持续高暴露姿势",
  fe_excess_dose_accumulating: "屈伸超量剂量累积",
  rud_excess_dose_accumulating: "桡尺偏超量剂量累积",
  pressure_over_screening: "压力超过筛查参数",
  discomfort_reported: "记录到不适反馈",
};

const channelLabel: Record<string, string> = {
  wrist_angles: "腕角",
  thumb_angle: "拇指角",
  pressure: "压力",
  tension: "拉力",
  discomfort: "不适",
  user_continues: "继续意愿",
};

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span>{k}</span>
      <span className="text-right text-foreground">{v}</span>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="font-display text-lg leading-none">{value}</p>
      <p className="mt-1.5 text-[0.65rem] text-muted-foreground">{label}</p>
    </div>
  );
}

function msToClock(ms: number) {
  const total = Math.round(ms / 1000);
  const mm = String(Math.floor(total / 60)).padStart(2, "0");
  const ss = String(total % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

/** 双通道角度轨迹：屈伸为面积带，桡尺偏为细线，底部为区带条。 */
function AngleTrack({ rows }: { rows: TimelineRow[] }) {
  if (rows.length === 0) return <p className="mt-3 text-xs text-muted-foreground">暂无时间轴数据。</p>;
  const w = 320;
  const h = 96;
  const max = Math.max(20, ...rows.map((r) => Math.max(Math.abs(r.theta_FE), Math.abs(r.theta_RUD))));
  const x = (i: number) => (i / (rows.length - 1 || 1)) * w;
  const y = (v: number) => h / 2 - (v / max) * (h / 2 - 4);
  const line = (key: "theta_FE" | "theta_RUD") =>
    rows.map((r, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(r[key]).toFixed(1)}`).join(" ");

  return (
    <div className="mt-3">
      <svg viewBox={`0 0 ${w} ${h}`} className="h-24 w-full" preserveAspectRatio="none">
        <line x1="0" y1={h / 2} x2={w} y2={h / 2} stroke="var(--border)" strokeWidth="1" />
        <path
          d={`${line("theta_FE")} L${w},${h / 2} L0,${h / 2} Z`}
          fill="color-mix(in oklab, var(--sky) 30%, transparent)"
        />
        <path d={line("theta_FE")} fill="none" stroke="var(--sky)" strokeWidth="1.6" />
        <path
          d={line("theta_RUD")}
          fill="none"
          stroke="color-mix(in oklab, var(--blush) 85%, black 5%)"
          strokeWidth="1.2"
          strokeDasharray="3 3"
        />
      </svg>
      <div className="mt-2 flex h-1.5 overflow-hidden rounded-full">
        {rows.map((r, i) => (
          <span
            key={i}
            className="flex-1"
            style={{ background: zoneColor[r.angle_zone] ?? "var(--muted)" }}
          />
        ))}
      </div>
      <div className="mt-2 flex items-center gap-3 text-[0.62rem] text-muted-foreground">
        <span>— 屈伸 θFE</span>
        <span>-- 桡尺偏 θRUD</span>
        <span>峰值 ±{max.toFixed(0)}°</span>
      </div>
    </div>
  );
}
