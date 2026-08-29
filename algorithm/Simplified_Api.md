# SheWrist 简化版 API v1.0

本接口用于离线腕部姿势暴露分析，不输出腱鞘炎诊断、发病概率或临床安全结论。算法超参数由服务端固定，调用方无需传入。

## 1. 提交分析

```http
POST /api/v1/analysis-jobs
Content-Type: multipart/form-data
Idempotency-Key: <可选的稳定请求键>
```

表单字段：

| 字段 | 必需 | 说明 |
| --- | --- | --- |
| `metadata` | 是 | JSON 字符串 |
| `data_file` | 是 | 任务 CSV，可为原始双 IMU 或已计算腕角 |
| `mechanical_file` | 否 | 原始 FSR、主观不适和安全症状等时序字段 |
| `calibration_file` | 原始 A/B/C 必需 | 独立 CAL 与静态验证 CSV；仅用于 `raw_dual_imu` |

### 现场原始双 IMU 元数据

```json
{
  "schema_version": "1.0",
  "session_id": "20260828_S01_A",
  "input_type": "raw_dual_imu",
  "evidence_type": "bench",
  "timestamp_basis": "device_ms",
  "condition": "A",
  "support_level": 0,
  "reminder_enabled": false,
  "firmware_version": "fw-1.0.0",
  "task_version": "typing-mouse-v1",
  "sensor_units": {
    "acceleration": "m/s2",
    "angular_velocity": "rad/s"
  },
  "sensors": [
    {
      "sensor_id": "forearm",
      "placement": "right_distal_forearm",
      "coordinate_frame": "sensor_local"
    },
    {
      "sensor_id": "hand",
      "placement": "right_hand_third_metacarpal_dorsum",
      "coordinate_frame": "sensor_local"
    }
  ],
  "calibration": {
    "calibration_id": "S01-CAL-001",
    "segments": [
      {"type": "neutral", "start_ms": 0, "end_ms": 4900},
      {"type": "extension", "start_ms": 6000, "end_ms": 10900},
      {"type": "flexion", "start_ms": 12000, "end_ms": 16900},
      {"type": "ulnar_deviation", "start_ms": 18000, "end_ms": 22900}
    ]
  },
  "options": {
    "enable_ml_shadow": true,
    "threshold_version": "engineering_v1",
    "explanation_provider": "local_template",
    "enable_external_api": false,
    "generate_charts": false
  }
}
```

`calibration.segments` 的时间属于 `calibration_file`。最新版现场流程使用中立、掌屈、背伸、尺偏四段；`radial_deviation` 可选。A/B/C 任务文件不需要额外插入中立段，但 CAL 后不得移动 IMU 或护腕；移动后必须重新 CAL 并更换 `calibration_id`。

`data_file` 或 `calibration_file` 的最小字段：

```csv
device_ms,sensor_id,ax,ay,az,gx,gy,gz,quality
1000000,forearm,0.01,0.03,9.80,0.001,-0.002,0.003,0.98
1000000,hand,0.02,0.04,9.79,0.002,-0.001,0.004,0.97
```

安装枚举固定为：

```text
forearm: right_distal_forearm | left_distal_forearm
hand:    right_hand_third_metacarpal_dorsum | left_hand_third_metacarpal_dorsum
```

两颗 IMU 必须同侧、轴向一致。算法计算：

```text
q_rel = inverse(q_forearm) × q_hand
```

### 现场机械/状态文件

```csv
device_ms,condition,support_level,fsr_raw_adc,discomfort_nrs,safety_symptom_flag,user_continues
1000000,A,0,1018,1,0,1
1000020,A,0,1021,1,0,1
```

`fsr_raw_adc` 是未标定 RFP-602 代理量，不能解释为 `N` 或 `kPa`，也不会触发压力阈值。`discomfort_nrs` 范围为 `0–10`，只用于记录；`safety_symptom_flag=1` 表示疼痛、麻木、发凉、皮肤变色、明显压痕等停止条件，会立即触发停止提示。

冻结条件：

| 条件 | `support_level` | `reminder_enabled` | 行为 |
| --- | ---: | --- | --- |
| A | 0 | `false` | 关闭提醒，记录 `would_alert` |
| B | 1 | `false` | 关闭提醒，记录 `would_alert` |
| C | 1 | `true` | 实际提醒；只回到舒适中立位，不额外收紧 |

### 已有腕角

无需双 IMU 运动学时，可提交：

```json
{
  "schema_version": "1.0",
  "session_id": "S001",
  "input_type": "joint_state",
  "evidence_type": "replay",
  "timestamp_basis": "session_relative_ms"
}
```

```csv
timestamp_ms,theta_FE,theta_RUD,quality
0,5.2,-2.1,0.98
20,6.1,-2.5,0.97
```

### 返回

```json
{
  "schema_version": "1.0",
  "job_id": "job_xxx",
  "session_id": "20260828_S01_A",
  "status": "queued",
  "status_url": "/api/v1/analysis-jobs/job_xxx",
  "result_url": "/api/v1/sessions/20260828_S01_A"
}
```

## 1.5 校准档案（可选，一次录制多次复用）

用于避免每次分析都重传 CAL 文件。先把一次合格的中立+功能校准录制成服务端档案，之后的分析任务只引用 `calibration_id`。

### 录制并保存校准

```http
POST /api/v1/calibrations
Content-Type: multipart/form-data
```

表单字段：

| 字段 | 必需 | 说明 |
| --- | --- | --- |
| `metadata` | 是 | JSON 字符串，见下 |
| `calibration_file` | 是 | 覆盖中立、掌屈、背伸、尺偏四段动作的双 IMU CAL CSV |

`metadata` 结构（`calibration_id` 提升到顶层；`sensors`/`sensor_units`/`calibration.segments` 与原始双 IMU 一致）：

```json
{
  "schema_version": "1.0",
  "calibration_id": "S01-CAL-001",
  "participant_id": "S01",
  "sensor_units": {"acceleration": "m/s2", "angular_velocity": "rad/s"},
  "sensors": [
    {"sensor_id": "forearm", "placement": "right_distal_forearm", "coordinate_frame": "sensor_local"},
    {"sensor_id": "hand", "placement": "right_hand_third_metacarpal_dorsum", "coordinate_frame": "sensor_local"}
  ],
  "calibration": {
    "segments": [
      {"type": "neutral", "start_ms": 0, "end_ms": 4900},
      {"type": "extension", "start_ms": 6000, "end_ms": 10900},
      {"type": "flexion", "start_ms": 12000, "end_ms": 16900},
      {"type": "ulnar_deviation", "start_ms": 18000, "end_ms": 22900}
    ]
  }
}
```

返回 `201`：

```json
{
  "schema_version": "1.0",
  "calibration_id": "S01-CAL-001",
  "participant_id": "S01",
  "status": "passed",
  "quality_gate_passed": true,
  "quality_reasons": [],
  "neutral_quaternion": [1.0, 0.0, 0.0, 0.0],
  "flexion_extension_axis": [0.99, 0.02, -0.01],
  "self_url": "/api/v1/calibrations/S01-CAL-001"
}
```

只有质量门控通过才会保存。中立位晃动或缺段会返回 `422 CALIBRATION_QUALITY_FAILED`（含 `quality_reasons`）；`calibration_id` 已存在返回 `409 CALIBRATION_EXISTS`。IMU 或护腕移动后必须重新录制并更换 `calibration_id`。

### 查询校准

```http
GET /api/v1/calibrations/{calibration_id}
```

不存在返回 `404 CALIBRATION_NOT_FOUND`。

### 分析任务引用已存校准

`raw_dual_imu` 任务此时不再需要 `calibration.segments`，也不要上传 `calibration_file`，只需在 `metadata.calibration` 里引用：

```json
"calibration": {
  "use_stored_profile": true,
  "calibration_id": "S01-CAL-001"
}
```

结果中 `calibration.application_mode` 为 `stored_calibration_profile`，`calibration.source_calibration_id` 回显所用档案。若引用的 `calibration_id` 不存在，创建任务时即返回 `404 CALIBRATION_NOT_FOUND`；同时传 `calibration_file` 返回 `400 INVALID_SCHEMA`。

## 2. 查询任务

```http
GET /api/v1/analysis-jobs/{job_id}
```

`status` 为 `queued`、`running`、`succeeded` 或 `failed`。只有 `succeeded` 后才读取结果。

## 3. 获取结果

```http
GET /api/v1/sessions/{session_id}
```

后端重点字段：

```json
{
  "analysis_status": "accepted",
  "trial_condition": {
    "condition": "A",
    "support_level": 0,
    "reminder_enabled": false,
    "protocol_order": "A_then_B_then_C"
  },
  "channels": {
    "pressure": {"available": false, "source": null},
    "fsr_raw": {"available": true, "source": "fsr_raw_adc"},
    "safety_symptom": {"available": true, "source": "operator_safety_stop_flag"}
  },
  "fsr_proxy": {
    "available": true,
    "source": "fsr_raw_adc",
    "unit": "adc_count",
    "calibrated_to_pressure": false,
    "mean": 1024.0,
    "p95": 1090.0,
    "max": 1110.0
  },
  "metrics": {
    "high_posture_time_pct": 22.4,
    "total_excess_dose_deg_s": 630.7,
    "alert_count": 0,
    "would_alert_count": 3,
    "safety_stop_count": 0,
    "max_pressure_kpa": null
  }
}
```

`analysis_status=accepted` 只表示本次数据通过同步、校准和质量门控。它不代表产品有效或医学安全。

## 4. 时间线

```http
GET /api/v1/sessions/{session_id}/timeline?offset=0&limit=1000
```

时间线包含 `alert`、`would_alert`、`fsr_raw`、`discomfort_nrs`、`safety_symptom` 和 `safety_stop`。A/B 的角度事件表现为 `would_alert=true`、`alert=false`；安全停止不受 A/B 提醒开关影响。

其他端点：

```text
GET /healthz
GET /api/v1/sessions/{session_id}/tokens
GET /api/v1/sessions/{session_id}/artifacts/{name}
```

完整契约见 `docs/backend_api.md`，机器可读契约见 `docs/openapi.json`。
