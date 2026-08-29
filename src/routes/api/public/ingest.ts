import { createFileRoute } from "@tanstack/react-router";
import { z } from "zod";

/**
 * 硬件遥测上报接口（对护腕固件 / 网关开放，公网可访问）。
 * 鉴权：请求头携带 x-device-token: <DEVICE_INGEST_TOKEN>。
 * 支持单条或批量：samples 数组最多 200 条。
 */
const sampleSchema = z.object({
  device_id: z.string().min(1).max(64),
  session_id: z.string().min(1).max(64),
  timestamp_ms: z.number().int().nonnegative(),
  theta_fe: z.number().min(-180).max(180).optional(),
  theta_rud: z.number().min(-180).max(180).optional(),
  pressure_kpa: z.number().min(0).max(100).optional(),
  temperature_c: z.number().min(-20).max(80).optional(),
  battery_pct: z.number().int().min(0).max(100).optional(),
  quality: z.number().min(0).max(1).optional(),
});

const payloadSchema = z.object({
  samples: z.array(sampleSchema).min(1).max(200),
});

export const Route = createFileRoute("/api/public/ingest")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const expected = process.env["DEVICE_INGEST_TOKEN"];
        const token = request.headers.get("x-device-token");
        if (!expected || token !== expected) {
          return Response.json({ error: "invalid_token" }, { status: 401 });
        }

        let payload: unknown;
        try {
          payload = await request.json();
        } catch {
          return Response.json({ error: "invalid_json" }, { status: 400 });
        }

        const parsed = payloadSchema.safeParse(payload);
        if (!parsed.success) {
          return Response.json(
            { error: "invalid_payload", issues: parsed.error.issues.slice(0, 5) },
            { status: 422 },
          );
        }

        const rows = parsed.data.samples.map((s) => ({
          device_id: s.device_id,
          session_id: s.session_id,
          timestamp_ms: s.timestamp_ms,
          theta_fe: s.theta_fe ?? null,
          theta_rud: s.theta_rud ?? null,
          pressure_kpa: s.pressure_kpa ?? null,
          temperature_c: s.temperature_c ?? null,
          battery_pct: s.battery_pct ?? null,
          quality: s.quality ?? null,
          raw: s,
        }));

        const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
        const { error } = await supabaseAdmin.from("wrist_samples").insert(rows);
        if (error) {
          console.error("wrist_samples insert failed", error);
          return Response.json({ error: "store_failed" }, { status: 500 });
        }

        return Response.json({ ok: true, stored: rows.length }, { status: 201 });
      },
    },
  },
});
