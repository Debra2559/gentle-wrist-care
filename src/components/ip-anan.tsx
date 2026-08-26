import ipAvatar from "@/assets/ip-anan-avatar.png";
import ipFull from "@/assets/ip-anan.png";

export function IPAvatar({ className = "h-10 w-10" }: { className?: string }) {
  return (
    <img
      src={ipAvatar}
      alt="腕安 IP 形象「安安」头像"
      width={816}
      height={816}
      loading="lazy"
      className={`shrink-0 object-contain ${className}`}
    />
  );
}

export function IPStage({ line }: { line: string }) {
  return (
    <div className="relative overflow-hidden rounded-[calc(var(--radius)+20px)]">
      <div className="ip-stage absolute inset-0" />
      <div className="relative grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2 px-5 pt-5">
        <div className="pb-6">
          <p className="text-[0.65rem] tracking-[0.28em] text-muted-foreground">ANAN · 安安</p>
          <p className="mt-2 font-display text-base leading-relaxed">{line}</p>
        </div>
        <img
          src={ipFull}
          alt="腕安 IP 形象「安安」：戴着柔软护腕的白兔"
          width={1024}
          height={1024}
          className="ip-float -mb-1 h-40 w-auto object-contain"
        />
      </div>
    </div>
  );
}

export function IPWhisper({ text }: { text: string }) {
  return (
    <div className="flex items-start gap-3">
      <IPAvatar className="h-11 w-11" />
      <p className="rounded-2xl bg-muted/70 px-4 py-3 text-xs leading-relaxed text-muted-foreground">
        {text}
      </p>
    </div>
  );
}

/** 沉浸式主视觉：安安占据画面中心，数据以光弧的方式围绕她。 */
export function IPHero({
  greeting,
  line,
  ringPct,
  wearText,
  syncedAt,
}: {
  greeting: string;
  line: string;
  ringPct: number;
  wearText: string;
  syncedAt: string;
}) {
  return (
    <section className="relative -mx-5 overflow-hidden pb-2">
      <div className="ip-aura breathe pointer-events-none absolute inset-x-0 -top-10 h-[26rem]" />

      <div className="relative px-5 pt-2">
        <p className="text-[0.62rem] tracking-[0.38em] text-muted-foreground">WRIST&nbsp;·&nbsp;腕安</p>
        <h1 className="mt-3 font-display text-[1.75rem] leading-[1.35]">
          {greeting}
          <br />
          <span className="text-muted-foreground">{line}</span>
        </h1>
        <p className="mt-2 text-[0.7rem] text-muted-foreground">护腕已同步 · {syncedAt}</p>
      </div>

      <div className="relative mt-1 grid place-items-center">
        {/* 光弧进度：环绕在安安身后 */}
        <svg viewBox="0 0 240 240" className="absolute h-[19rem] w-[19rem] -rotate-90">
          <circle
            cx="120"
            cy="120"
            r="104"
            fill="none"
            stroke="var(--mist)"
            strokeWidth="2.5"
            strokeDasharray="1 7"
            strokeLinecap="round"
          />
          <circle
            cx="120"
            cy="120"
            r="104"
            fill="none"
            stroke="var(--sky)"
            strokeWidth="4"
            strokeLinecap="round"
            strokeDasharray={`${(ringPct / 100) * 653} 653`}
            className="transition-[stroke-dasharray] duration-1000"
          />
        </svg>
        <span className="sparkle absolute left-6 top-10 h-1.5 w-1.5 rounded-full bg-gold" />
        <span className="sparkle absolute right-8 top-24 h-1 w-1 rounded-full bg-blush [animation-delay:1.4s]" />
        <span className="sparkle absolute bottom-16 left-12 h-1 w-1 rounded-full bg-sky [animation-delay:2.6s]" />

        <img
          src={ipFull}
          alt="腕安 IP 形象「安安」：戴着柔软护腕的白兔"
          width={1024}
          height={1024}
          className="ip-float relative h-64 w-auto object-contain drop-shadow-[0_18px_28px_rgba(120,130,170,0.22)]"
        />

        <div className="orbit-chip absolute bottom-6 left-1 rounded-full px-3.5 py-2 text-center">
          <p className="font-display text-base leading-none">{ringPct}%</p>
          <p className="mt-1 text-[0.6rem] text-muted-foreground">佩戴完成</p>
        </div>
        <div className="orbit-chip absolute right-1 top-14 rounded-full px-3.5 py-2 text-center">
          <p className="font-display text-base leading-none">{wearText}</p>
          <p className="mt-1 text-[0.6rem] text-muted-foreground">今日佩戴</p>
        </div>
      </div>
    </section>
  );
}

/** 页面开场：安安半身探出，与标题构成留白式主视觉。 */
export function IPIntro({
  eyebrow,
  title,
  line,
}: {
  eyebrow: string;
  title: string;
  line: string;
}) {
  return (
    <section className="relative -mx-5 overflow-hidden px-5 pb-3 pt-1">
      <div className="ip-aura breathe pointer-events-none absolute -right-16 -top-20 h-64 w-64 rounded-full" />
      <div className="relative grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
        <div className="pb-2">
          <p className="text-[0.62rem] tracking-[0.38em] text-muted-foreground">{eyebrow}</p>
          <h1 className="mt-3 font-display text-[1.6rem] leading-[1.35]">{title}</h1>
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{line}</p>
        </div>
        <img
          src={ipFull}
          alt="腕安 IP 形象「安安」：戴着柔软护腕的白兔"
          width={1024}
          height={1024}
          className="ip-float -mr-3 -mb-2 h-32 w-auto object-contain drop-shadow-[0_14px_22px_rgba(120,130,170,0.2)]"
        />
      </div>
      <div className="hairline mt-3 h-px" />
    </section>
  );
}
