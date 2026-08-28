import { createServerFn } from "@tanstack/react-start";

import { buildDemoReport } from "./shewrist-demo";
import type { SessionReport, SessionResult, TimelineResponse } from "./shewrist-types";

/**
 * 拉取一次会话的分析结果 + 抽样时间轴。
 * 后端地址通过服务端环境变量 SHEWRIST_API_BASE_URL 配置（例如 http://127.0.0.1:8000）。
 * 未配置或不可用时回退到结构一致的演示数据，前端展示逻辑不变。
 */
export const getSessionReport = createServerFn({ method: "GET" })
  .inputValidator((input: { sessionId?: string; limit?: number }) => ({
    sessionId: (input.sessionId ?? "S001").trim().slice(0, 64) || "S001",
    limit: Math.min(Math.max(input.limit ?? 400, 50), 2000),
  }))
  .handler(async ({ data }): Promise<SessionReport> => {
    const base = process.env["SHEWRIST_API_BASE_URL"]?.replace(/\/+$/, "");
    if (!base) return buildDemoReport(data.sessionId);

    const headers: Record<string, string> = { accept: "application/json" };
    const token = process.env["SHEWRIST_API_TOKEN"];
    if (token) headers["authorization"] = `Bearer ${token}`;

    try {
      const [resultRes, timelineRes] = await Promise.all([
        fetch(`${base}/api/v1/sessions/${encodeURIComponent(data.sessionId)}`, { headers }),
        fetch(
          `${base}/api/v1/sessions/${encodeURIComponent(data.sessionId)}/timeline?offset=0&limit=${data.limit}`,
          { headers },
        ),
      ]);

      if (!resultRes.ok || !timelineRes.ok) {
        const body = (await resultRes.text().catch(() => "")).slice(0, 300);
        const demo = buildDemoReport(data.sessionId);
        return {
          ...demo,
          note: `接口返回 ${resultRes.status}${body ? `：${body}` : ""}，暂以演示数据展示。`,
        };
      }

      const result = (await resultRes.json()) as SessionResult;
      const timeline = (await timelineRes.json()) as TimelineResponse;
      return { source: "live", note: "", result, timeline };
    } catch (error) {
      console.error("SheWrist API 请求失败", error);
      const demo = buildDemoReport(data.sessionId);
      return { ...demo, note: "无法连接 SheWrist 后端，暂以演示数据展示。" };
    }
  });
