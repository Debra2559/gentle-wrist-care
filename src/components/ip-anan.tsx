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
