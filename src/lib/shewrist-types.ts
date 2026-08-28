/** SheWrist Backend API v1.0 —— 公共响应类型（对应 docs/backend_api.md） */

export type ChannelDescriptor = { available: boolean; source?: string | null };

export type Channels = {
  wrist_angles: ChannelDescriptor;
  thumb_angle: ChannelDescriptor;
  pressure: ChannelDescriptor;
  tension: ChannelDescriptor;
  discomfort: ChannelDescriptor;
  user_continues: ChannelDescriptor;
};

export type DataQuality = {
  sample_count: number;
  valid_sample_pct: number;
  valid_sample_pct_min: number;
  valid_sample_gate_passed: boolean;
  sample_rate_hz: number;
  median_sync_error_ms?: number | null;
  p95_sync_error_ms?: number | null;
  max_sync_error_ms?: number | null;
  sync_gate_passed?: boolean | null;
  sync_limit_ms?: number | null;
};

export type Metrics = {
  task_duration_s?: number | null;
  valid_sample_pct?: number | null;
  high_posture_time_pct?: number | null;
  fe_excess_dose_deg_s?: number | null;
  rud_excess_dose_deg_s?: number | null;
  total_excess_dose_deg_s?: number | null;
  longest_high_posture_s?: number | null;
  max_abs_fe_deg?: number | null;
  max_abs_rud_deg?: number | null;
  fe_cycles_per_min?: number | null;
  rud_cycles_per_min?: number | null;
  max_pressure_kpa?: number | null;
  pressure_over_screening_s?: number | null;
  alert_count?: number | null;
  safety_stop_count?: number | null;
  mechanical_recommendation_count?: number | null;
  max_external_assist_torque_nm?: number | null;
  mean_external_assist_torque_nm?: number | null;
};

export type Alert = {
  timestamp_ms: number;
  zone: string;
  reason: string;
  recommend_mechanical: boolean;
  safety_stop?: boolean | null;
};

export type ExplanationSummary = {
  provider: string;
  model: string;
  api_called: boolean;
  summary: string;
  observations: string[];
  limitations: string[];
  next_steps: string[];
  safety_effect: string;
};

export type MlShadowSummary = {
  operating_mode: string;
  timing_semantics: string;
  window_count: number;
  accepted_window_count: number;
  rejected_window_count: number;
  token_count: number;
  tokens_url: string;
  safety_effect: string;
};

export type SessionResult = {
  schema_version: string;
  job_id: string;
  session_id: string;
  status: string;
  analysis_status: string;
  rejection_reasons: string[];
  evidence_type: string;
  algorithm_release: string;
  channels: Channels;
  data_quality: DataQuality;
  metrics: Metrics;
  alerts: Alert[];
  ml_shadow: MlShadowSummary;
  explanation: ExplanationSummary;
  warnings: string[];
  evidence_limits: string[];
};

export type TimelineRow = {
  timestamp_ms: number;
  theta_FE: number;
  theta_RUD: number;
  quality: number;
  angle_zone: string;
  activity_shadow: string;
  alert: boolean;
  alert_reason: string;
  pressure_zone?: string | null;
  discomfort?: boolean | null;
  safety_stop?: boolean | null;
  user_continues?: boolean | null;
};

export type TimelineResponse = {
  schema_version: string;
  session_id: string;
  offset: number;
  limit: number;
  total: number;
  items: TimelineRow[];
};

export type JobStatus = {
  schema_version: string;
  job_id: string;
  session_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  stage?: string;
  progress_pct?: number;
  status_url: string;
  result_url: string;
  error?: { code: string; message: string } | null;
};

/** 页面所需的一次性负载：会话结果 + 抽样时间轴。 */
export type SessionReport = {
  source: "live" | "demo";
  note: string;
  result: SessionResult;
  timeline: TimelineResponse;
};
