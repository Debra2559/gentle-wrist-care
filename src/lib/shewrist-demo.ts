import type { SessionReport, SessionResult, TimelineResponse } from "./shewrist-types";

/** 后端未接通时的演示负载，字段与 SheWrist Backend API v1.0 完全一致。 */
export function buildDemoReport(sessionId: string): SessionReport {
  const items = Array.from({ length: 120 }, (_, i) => {
    const t = i * 250;
    const fe = 18 * Math.sin(i / 7) + 9 * Math.sin(i / 2.3) + 6;
    const rud = 11 * Math.sin(i / 5 + 1.2);
    const zone =
      Math.abs(fe) > 30 || Math.abs(rud) > 20
        ? "red"
        : Math.abs(fe) > 15 || Math.abs(rud) > 10
          ? "yellow"
          : "green";
    return {
      timestamp_ms: t,
      theta_FE: Number(fe.toFixed(2)),
      theta_RUD: Number(rud.toFixed(2)),
      quality: 0.97,
      angle_zone: zone,
      activity_shadow: i % 23 === 0 ? "static_hold" : "repetitive_flexion",
      alert: zone === "red",
      alert_reason: zone === "red" ? "sustained_high_posture" : "",
      pressure_zone: zone === "red" ? "over_screening" : "under_screening",
      discomfort: false,
      safety_stop: false,
      user_continues: true,
    };
  });

  const timeline: TimelineResponse = {
    schema_version: "1.0",
    session_id: sessionId,
    offset: 0,
    limit: items.length,
    total: items.length,
    items,
  };

  const result: SessionResult = {
    schema_version: "1.0",
    job_id: "job_demo00000000000000",
    session_id: sessionId,
    status: "succeeded",
    analysis_status: "accepted",
    rejection_reasons: [],
    evidence_type: "simulation",
    algorithm_release: "engineering_v1",
    channels: {
      wrist_angles: { available: true, source: "joint_state" },
      thumb_angle: { available: false, source: null },
      pressure: { available: true, source: "mechanical_file" },
      tension: { available: true, source: "mechanical_file" },
      discomfort: { available: true, source: "mechanical_file" },
      user_continues: { available: true, source: "mechanical_file" },
    },
    data_quality: {
      sample_count: 1200,
      valid_sample_pct: 96.4,
      valid_sample_pct_min: 80,
      valid_sample_gate_passed: true,
      sample_rate_hz: 50,
      median_sync_error_ms: 4.1,
      p95_sync_error_ms: 11.8,
      max_sync_error_ms: 17.2,
      sync_gate_passed: true,
      sync_limit_ms: 20,
    },
    metrics: {
      task_duration_s: 1830,
      valid_sample_pct: 96.4,
      high_posture_time_pct: 27.5,
      fe_excess_dose_deg_s: 1420.6,
      rud_excess_dose_deg_s: 486.3,
      total_excess_dose_deg_s: 1906.9,
      longest_high_posture_s: 132.5,
      max_abs_fe_deg: 41.2,
      max_abs_rud_deg: 22.8,
      fe_cycles_per_min: 18.4,
      rud_cycles_per_min: 7.1,
      max_pressure_kpa: 5.2,
      pressure_over_screening_s: 96.5,
      alert_count: 6,
      safety_stop_count: 0,
      mechanical_recommendation_count: 3,
      max_external_assist_torque_nm: 0.42,
      mean_external_assist_torque_nm: 0.18,
    },
    alerts: [
      {
        timestamp_ms: 322000,
        zone: "yellow",
        reason: "sustained_high_posture",
        recommend_mechanical: false,
      },
      {
        timestamp_ms: 764500,
        zone: "red",
        reason: "fe_excess_dose_accumulating",
        recommend_mechanical: true,
        safety_stop: false,
      },
      {
        timestamp_ms: 1288000,
        zone: "red",
        reason: "pressure_over_screening",
        recommend_mechanical: true,
        safety_stop: false,
      },
    ],
    ml_shadow: {
      operating_mode: "shadow",
      timing_semantics: "window_end_aligned",
      window_count: 24,
      accepted_window_count: 21,
      rejected_window_count: 3,
      token_count: 21,
      tokens_url: `/api/v1/sessions/${sessionId}/tokens`,
      safety_effect: "none",
    },
    explanation: {
      provider: "local_template",
      model: "template_zh_CN",
      api_called: false,
      summary: "本次会话中高暴露姿势占 27.5%，屈伸方向剂量明显高于桡尺偏，建议缩短连续操作时长。",
      observations: [
        "屈伸循环 18.4 次/分，属于高重复区间。",
        "最长连续高暴露 132.5 秒，出现在会话中段。",
        "压力峰值 5.2 kPa，超过 4.4 kPa 工程筛查参数约 96 秒。",
      ],
      limitations: ["工程筛查参数不是疾病或人体安全阈值。", "影子模型不参与任何报警与控制。"],
      next_steps: ["每 45 分钟安排一次腕部舒展。", "复核护腕拇指侧支撑贴合度。"],
      safety_effect: "none",
    },
    warnings: ["拇指角通道缺失，相关指标不可用。"],
    evidence_limits: ["仅用于工程原型与工效暴露研究，不作诊断依据。"],
  };

  return {
    source: "demo",
    note: "尚未配置 SHEWRIST_API_BASE_URL，当前展示演示数据（结构与真实接口一致）。",
    result,
    timeline,
  };
}
