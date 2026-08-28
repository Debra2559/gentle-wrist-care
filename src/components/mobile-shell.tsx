import { Link } from "@tanstack/react-router";
import { Activity, BellRing, FileText, Home, Sparkles, Watch } from "lucide-react";
import type { ReactNode } from "react";

const tabs = [
  { to: "/", label: "今日", icon: Home },
  { to: "/monitor", label: "监测", icon: Activity },
  { to: "/alerts", label: "预警", icon: BellRing },
  { to: "/advice", label: "建议", icon: Sparkles },
  { to: "/report", label: "报告", icon: FileText },
  { to: "/device", label: "护腕", icon: Watch },
] as const;

export function MobileShell({
  title,
  subtitle,
  children,
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <div className="mx-auto min-h-screen w-full max-w-[26rem] px-5 pb-28 pt-8">
      {title ? (
        <header className="mb-6">
          <h1 className="text-2xl text-foreground">{title}</h1>
          {subtitle ? <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p> : null}
        </header>
      ) : null}
      <main className="space-y-5">{children}</main>

      <nav className="fixed inset-x-0 bottom-0 z-20 flex justify-center pb-4">
        <div className="tabbar mx-4 flex w-full max-w-[24rem] items-center justify-between gap-1 rounded-3xl px-2 py-2">
          {tabs.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              activeOptions={{ exact: to === "/" }}
              className="flex flex-1 flex-col items-center gap-1 rounded-2xl px-2 py-2 text-[0.62rem] text-muted-foreground transition-colors"
              activeProps={{ className: "bg-secondary text-secondary-foreground" }}
            >
              <Icon className="h-[1.15rem] w-[1.15rem]" strokeWidth={1.6} />
              <span>{label}</span>
            </Link>
          ))}
        </div>
      </nav>
    </div>
  );
}
