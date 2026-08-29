# SheWrist Backend API v1.0

该接口将纯离线腕部暴露算法封装为异步 HTTP 服务。运动学、阈值和状态机由服务端固定；ML 与解释服务的提醒、停止和机械控制权限始终为 `none`。

> 接口仅用于工程原型和工效暴露研究，不输出腱鞘炎诊断、发病概率或临床安全结论。

## 启动

```bash
python3 -m pip install -r requirements.txt -r requirements-api.txt
PYTHONPATH=src python3 scripts/run_api.py --host 127.0.0.1 --port 8000
```

交互文档位于 `http://127.0.0.1:8000/docs`，机器可读契约位于 `docs/openapi.json`。当前任务和会话使用本地文件存储，只支持单进程 `--workers 1`。默认单文件上传上限为 `256 MiB`。

## 端点

```text
GET  /healthz
POST /api/v1/calibrations
GET  /api/v1/calibrations/{calibration_id}
POST /api/v1/analysis-jobs
GET  /api/v1/analysis-jobs/{job_id}
GET  /api/v1/sessions/{session_id}
GET  /api/v1/sessions/{session_id}/timeline?offset=0&limit=1000
GET  /api/v1/sessions/{session_id}/tokens
GET  /api/v1/sessions/{session_id}/artifacts/{name}
```

允许下载的产物：

```text
analysis.json
joint_state.csv
timeline.csv
tokens.json
manifest.json
session_report.png
session_report.svg
```

## 提交流程

`POST /api/v1/analysis-jobs` 使用 `multipart/form-data`：

| 表单项 | 必需 | 说明 |
| --- | --- | --- |
| `metadata` | 是 | UTF-8 JSON 字符串 |
| `data_file` | 是 | `raw_dual_imu` 或 `joint_state` CSV |
| `mechanical_file` | 否 | RFP-602、已标定压力、主观评分和安全状态 CSV |
| `calibration_file` | 原始 A/B/C 必需 | 独立 CAL/静态验证双 IMU CSV，仅适用于 `raw_dual_imu` |
| `Idempotency-Key` 请求头 | 推荐 | 同键同请求复用任务；同键不同请求返回 `409` |

流程：提交后获得 HTTP `202`，轮询任务直到 `succeeded` 或 `failed`；成功后读取会话结果，并按需分页读取时间线。

## 现场原始双 IMU

元数据示例：

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
    "mode": "neutral_plus_static_validation",
    "reference_pose": "anatomical_neutral",
    "segments": [
      {"type": "neutral", "start_ms": 0, "end_ms": 4900},
      {"type": "extension", "start_ms": 6000, "end_ms": 10900},
      {"type": "flexion", "start_ms": 12000, "end_ms": 16900},
      {"type": "ulnar_deviation", "start_ms": 18000, "end_ms": 22900}
    ]
  },
  "options": {
    "enable_ml_shadow": true,
    "chunk_size": 128,
    "threshold_version": "engineering_v1",
    "explanation_provider": "local_template",
    "enable_external_api": false,
    "generate_charts": true,
    "language": "zh-CN"
  }
}
```

合法安装枚举：

```text
forearm: right_distal_forearm | left_distal_forearm
hand:    right_hand_third_metacarpal_dorsum | left_hand_third_metacarpal_dorsum
```

`forearm` 位于腕关节线向肘部约 `5 cm`、允许 `4–6 cm` 的前臂背侧中线；`hand` 位于距腕关节线约 `2–3 cm` 的手背第 3 掌骨中段。两个节点必须同侧、轴向一致，坐标系固定为 `sensor_local`。接口只验证声明，不能证明实物安装正确。

任务与 CAL CSV 最小字段：

```csv
device_ms,sensor_id,ax,ay,az,gx,gy,gz,quality
1000000,forearm,0.01,0.03,9.80,0.001,-0.002,0.003,0.98
1000000,hand,0.02,0.04,9.79,0.002,-0.001,0.004,0.97
```

也可使用 `timestamp_ms`。`device_ms` 会归一为各文件自己的会话相对时间。`host_unix_ms` 可保留在现场文件中用于审计，但当前不参与运动学。

现有有线样机产生的 `device_us + hand_* + arm_* + pressure_adc*` 宽表不是本接口的直接上传格式。先运行：

```bash
.venv/bin/python scripts/import_hardware_captures.py \
  --source-dir datasets \
  --output-dir outputs/hardware_capture_import
```

该适配器生成标准交错 `forearm/hand` CSV，完成微秒转毫秒及 `g/deg/s → m/s²/rad/s`，同时写入缺口、全零、冻结和饱和质量标记。它不重采样、不修改源文件，也不补造参与者、A/B/C、任务或 CAL 标签。

标准化副本通过 `parse_raw_dual_imu` 仅代表格式兼容。没有独立 `calibration_file` 和明确 `calibration_id` 时不得提交为正式 A/B/C 或目标腕角分析；API 的现有校准门控保持不变。

A/B/C 原始会话必须上传独立 `calibration_file`：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analysis-jobs \
  -H 'Idempotency-Key: 20260828_S01_A' \
  -F 'metadata=<examples/api_metadata_raw_dual_imu.json' \
  -F 'data_file=@path/to/S01_A_imu.csv;type=text/csv' \
  -F 'calibration_file=@path/to/S01_CAL_imu.csv;type=text/csv' \
  -F 'mechanical_file=@path/to/S01_A_mechanical.csv;type=text/csv'
```

`calibration.segments` 的时间属于 `calibration_file`。必需类型是 `neutral / flexion / extension / ulnar_deviation`，`radial_deviation` 可选。任务文件不要求额外中立段，也不会重新估计功能轴或中立位。移动 IMU 或护腕后必须重新 CAL 并更换 `calibration_id`。

独立 CAL 使用加速度倾角初始化且初始航向固定为零。六轴数据不能独立观测绕重力方向的航向，因此接口会返回安装不变与启动姿态可重复的警告；它不构成目标硬件精度证明。

## 校准档案（一次录制，多次复用）

为避免每次分析都重传 CAL 文件，可先把一次合格的中立 + 功能校准录制成服务端档案，后续分析任务只引用 `calibration_id`。运动学阈值与门控全部在服务端，调用方只负责提供原始分段数据并读取结果。

### 录制并保存

```http
POST /api/v1/calibrations
Content-Type: multipart/form-data
```

| 表单项 | 必需 | 说明 |
| --- | --- | --- |
| `metadata` | 是 | UTF-8 JSON 字符串，见下 |
| `calibration_file` | 是 | 覆盖中立、掌屈、背伸、尺偏（桡偏可选）的双 IMU CAL CSV |

`metadata` 结构：`calibration_id` 提升到顶层；`sensor_units` / `sensors` / `calibration.segments` 与原始双 IMU 任务一致。

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

`calibration.segments` 的时间属于 `calibration_file`。必需类型是 `neutral / flexion / extension / ulnar_deviation`，`radial_deviation` 可选；功能段单段不超过 `15 s`。`calibration_file` 的最小字段与任务/CAL CSV 相同（`device_ms|timestamp_ms, sensor_id, ax..az, gx..gz, quality`）。

`curl` 示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/calibrations \
  -F 'metadata=<examples/api_metadata_calibration.json' \
  -F 'calibration_file=@path/to/S01_CAL_imu.csv;type=text/csv'
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
  "neutral_stationary_sample_pct": 98.3,
  "flexion_extension_axis": [0.99, 0.02, -0.01],
  "radial_ulnar_axis": [0.03, 0.99, 0.02],
  "pronation_supination_axis": [0.01, -0.02, 0.99],
  "neutral_quaternion": [1.0, 0.0, 0.0, 0.0],
  "self_url": "/api/v1/calibrations/S01-CAL-001"
}
```

只有质量门控通过才会保存（中立位静止样本 `≥30` 且占比 `≥70%`）。中立位晃动或缺段返回 `422 CALIBRATION_QUALITY_FAILED`（`details.quality_reasons` 说明原因，需重录）；`calibration_id` 已存在返回 `409 CALIBRATION_EXISTS`。移动 IMU 或护腕后必须重新录制并更换 `calibration_id`。

### 查询

```http
GET /api/v1/calibrations/{calibration_id}
```

不存在返回 `404 CALIBRATION_NOT_FOUND`。

### 分析任务引用已存档案

`raw_dual_imu` 任务此时不再需要 `calibration.segments`，也不要上传 `calibration_file`，只在 `metadata.calibration` 引用：

```json
"calibration": {
  "use_stored_profile": true,
  "calibration_id": "S01-CAL-001"
}
```

结果中 `calibration.application_mode` 为 `stored_calibration_profile`，`calibration.source_calibration_id` 回显所用档案。引用不存在的 `calibration_id` 会在创建任务时返回 `404 CALIBRATION_NOT_FOUND`；同时携带 `calibration_file` 返回 `400 INVALID_SCHEMA`。

### 校准状态流

```text
中立位采集 ──[静止≥30 且占比≥70%]──▶ 功能动作采集 ──[四段齐全且区间有效]──▶ 生成档案(calibration_id)
    │                                        │
 [晃动/样本不足]                          [缺段 / 单段>15s]
    ▼                                        ▼
中立位不合格 ──[重录]──▶ 中立位采集       功能校准不合格 ──[重录]──▶ 功能动作采集
```

## 现场机械与状态文件

推荐字段：

```csv
device_ms,host_unix_ms,condition,support_level,fsr_raw_adc,discomfort_nrs,safety_symptom_flag,user_continues
1000000,1787932800000,A,0,1018,1,0,1
1000020,1787932800020,A,0,1021,1,0,1
```

规则：

- `condition` 和 `support_level` 若存在，必须与元数据一致。
- `fsr_raw_adc` 或别名 `fsr_raw` 必须非负；`fsr_normalized_pct` 必须在 `0..100`。
- `discomfort_nrs` 范围为 `0..10`，只记录，不直接停止。
- `safety_symptom_flag` 只接受 `0/1`，触发确定性释放/停止提示。
- `discomfort` 是旧版 `0/1` 兼容安全通道。
- `user_continues` 只接受 `0/1`。

未标定 RFP-602 只产生 ADC 或归一化代理量，不进入 kPa 压力筛查，不生成 `pressure_zone`。兼容的 `p_radial_kPa / p_dorsal_kPa / p_ulnar_kPa` 仅适用于已经独立标定为 kPa 的压力通道。

## A/B/C 冻结语义

| 条件 | 支撑 | 角度提醒 | 机械建议 |
| --- | ---: | --- | --- |
| A | 0 | 关闭，记录 `would_alert` | 禁用 |
| B | 1 | 关闭，记录 `would_alert` | 禁用 |
| C | 1 | 开启 | 禁用；提醒后只回到舒适中立位 |

顺序固定为 `A → B → C`。比较语义为：A vs B 是支撑增量，B vs C 是相同支撑下的提醒增量，A vs C 是组合影响。安全症状和已标定压力停止不受 A/B 的角度提醒开关影响。

人体 A/B/C 元数据还必须提供 `firmware_version`、`task_version`、`calibration.calibration_id`，并设置：

```json
{
  "compliance": {
    "deidentified": true,
    "consent_confirmed": true
  }
}
```

## 已有腕角

`joint_state` 元数据必须使用 `timestamp_basis=session_relative_ms`。最小 CSV：

```csv
timestamp_ms,theta_FE,theta_RUD,quality
0,5.2,-2.1,0.98
20,6.1,-2.5,0.97
```

完整可选列为 `theta_thumb / angular_velocity / calibration_id / quality`。选择该入口表示上游负责安装、同步、融合和标定。

## 任务状态

```json
{
  "schema_version": "1.0",
  "job_id": "job_0123456789abcdefabcd",
  "session_id": "20260828_S01_A",
  "status": "running",
  "stage": "deterministic_analysis",
  "progress_pct": 50,
  "status_url": "/api/v1/analysis-jobs/job_0123456789abcdefabcd",
  "result_url": "/api/v1/sessions/20260828_S01_A"
}
```

`status` 为 `queued / running / succeeded / failed`。`stage` 为 `validation / synchronization / deterministic_analysis / ml_finalize / reporting`。算法完成但同步、校准或有效率门控失败时，任务仍可为 `succeeded`，但 `analysis_status=rejected`。

## 会话结果

重点字段：

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
    "discomfort_nrs": {"available": true, "source": "participant_report_0_10"},
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

公共语义：

- `analysis_status=accepted` 只表示数据通过同步、校准和质量门控。
- `sensor_installation.contract_validated=true` 只表示元数据合法。
- 未标定 FSR 存在时，`fsr_proxy.available=true`，但 `pressure.available=false`、`max_pressure_kpa=null`、`pressure_over_screening_s=null`。
- 没有任何已标定压力或安全症状通道时，`safety_stop_count=null`；提供安全通道且未触发时为 `0`。
- `metrics.would_alert_count` 记录角度规则本来会触发的次数；A/B 的 `alert_count` 仍为 `0`。
- `control_policy.pressure_stop_authority` 为 `deterministic_calibrated_pressure_or_safety_symptom`。
- A/B 的 `angle_alert_authority` 为 `disabled_by_trial_condition`；C 为 `deterministic_exposure_engine`。
- A/B/C 的 `mechanical_action` 为 `disabled_by_trial_protocol`。

## 时间线

分页时间线字段：

```text
timestamp_ms,theta_FE,theta_RUD,quality,
angle_zone,pressure_zone,discomfort,user_continues,
activity_shadow,alert,would_alert,alert_reason,safety_stop,
fsr_raw,discomfort_nrs,safety_symptom
```

A/B 的角度事件为 `would_alert=true`、`alert=false`。安全停止不受该开关影响。

## 幂等、审计与错误

同一 `Idempotency-Key` 与完全相同的请求返回原任务；同键不同请求返回 `IDEMPOTENCY_CONFLICT`。Manifest 记录 `metadata.json`、任务 CSV、可选机械 CSV、可选 `calibration.csv`、配置、模型和输出的 SHA-256。

稳定错误码包括：

```text
INVALID_SCHEMA
INVALID_UNIT
INVALID_TRIAL_CONDITION
UNSUPPORTED_OPTION
NON_MONOTONIC_TIMESTAMP
MISSING_SENSOR_NODE
INVALID_SENSOR_PLACEMENT
CALIBRATION_REQUIRED
CALIBRATION_QUALITY_FAILED
CALIBRATION_EXISTS
CALIBRATION_NOT_FOUND
INSUFFICIENT_VALID_DATA
HUMAN_DATA_CONFIRMATION_REQUIRED
EXPLANATION_CONFIG_ERROR
UPLOAD_TOO_LARGE
SESSION_EXISTS
SESSION_NOT_FOUND
IDEMPOTENCY_CONFLICT
JOB_NOT_FOUND
RESULT_NOT_READY
JOB_FAILED
ARTIFACT_NOT_FOUND
ANALYSIS_FAILED
```

## 工程边界

`15°/20°/30°`、`10 s`、`300 s`、舒适度 `5/7` 和已标定压力 `3.0/4.4 kPa` 都是首版工程参数，不是疾病或人体安全阈值。原始 RFP-602 不适用 kPa 阈值。所有释放/停止结果均为软件提示，不是执行器动作。
