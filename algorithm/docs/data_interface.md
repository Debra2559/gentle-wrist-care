# 数据接口

本文件描述当前现场采集和算法输入的唯一工程契约。若与旧文档冲突，以最新版 `SheWrist_现场采集协作与传感器安装手册2026-08-28.docx` 为准。

## 坐标、单位与时间

- 四元数内部统一为 `[w, x, y, z]`。
- `theta_FE` 单位为度，正值表示腕背伸，负值表示腕掌屈。
- `theta_RUD` 单位为度，正值表示尺偏，负值表示桡偏。
- 加速度单位为 `m/s²`，角速度单位为 `rad/s`，磁场单位为 `µT`。
- `quality` 范围为 `0..1`；低于 `0.2` 的样本不参与角度暴露或角度提醒。
- 双 IMU 同步门限为 `20 ms`，同时检查首尾偏移和双向最近邻 `p95/max`。
- 会话只有在有效角度样本率 `>=80%` 时才被接受。

原始双 IMU、CAL 和机械文件可使用以下任一时间列：

```text
timestamp_ms
```

```text
device_ms
```

`device_ms` 会按各文件首个设备时间归一为会话相对毫秒。元数据保留 `source_timestamp_basis`，算法内部统一为 `session_relative_ms`。`host_unix_ms` 可随现场文件保存用于审计，但当前不参与运动学、重采样或状态机。`joint_state` 输入只接受 `timestamp_ms` 与 `timestamp_basis=session_relative_ms`。

## 目标双 IMU

| 物理位置 | `sensor_id` | 合法 `placement` |
| --- | --- | --- |
| 腕关节线向肘部约 5 cm、允许 4–6 cm 的前臂背侧中线 | `forearm` | `right_distal_forearm` 或 `left_distal_forearm` |
| 距腕关节线约 2–3 cm 的手背第 3 掌骨中段 | `hand` | `right_hand_third_metacarpal_dorsum` 或 `left_hand_third_metacarpal_dorsum` |

两个节点必须同侧、采用一致轴向，并声明 `coordinate_frame=sensor_local`。后端拒绝未知、交换或左右不一致的声明。元数据校验不能代替对实物安装位置、轴向、滑移和重戴的现场确认。

腕部相对姿态固定为：

```text
q_rel = inverse(q_forearm) × q_hand
```

## 原始双 IMU CSV

最小字段：

```csv
device_ms,sensor_id,ax,ay,az,gx,gy,gz,quality
1000000,forearm,0.01,0.03,9.80,0.001,-0.002,0.003,0.98
1000000,hand,0.02,0.04,9.79,0.002,-0.001,0.004,0.97
```

`quality` 可省略；省略后由算法质量评分提供。若提供磁力计，必须同时提供 `mx,my,mz`。默认使用六轴融合，未经硬铁、软铁和现场环境标定的磁场不能用于九轴定量分析。

原始 RFP-602 也可嵌入任务双 IMU CSV，字段使用 `fsr_raw_adc` 或兼容别名 `fsr_raw`，二者不能同时出现。同一时间戳若在两行 IMU 中都给出 FSR，数值必须一致。若已在 `mechanical_file` 提供 FSR，则不得在 `data_file` 重复提供。

## 现有有线样机宽表适配

2026-08-28 的有线采集文件采用每个时间点一行的宽表，不是 Backend API 的直接输入：

```text
device_us 或 device_ms,
pressure_adc 或 pressure_adc_raw,
hand_ax_g..hand_gz_dps,
arm_ax_g..arm_gz_dps
```

使用以下命令生成只读派生副本和审计报告：

```bash
.venv/bin/python scripts/import_hardware_captures.py \
  --source-dir datasets \
  --output-dir outputs/hardware_capture_import
```

转换固定执行：

- `device_us ÷ 1000 → device_ms`；原本为 `device_ms` 时保持设备时间。
- `hand_* → sensor_id=hand`，`arm_* → sensor_id=forearm`。
- 加速度 `g × 9.80665 → m/s²`；角速度 `deg/s × π/180 → rad/s`。
- `pressure_adc_raw` 优先于兼容的 `pressure_adc`，统一写入 `fsr_raw_adc`；只写在 `forearm` 行，避免同时间戳重复值冲突。
- 不插值、不降采样、不修改原文件；每个源文件保存 SHA-256，标准化结果写入 `outputs/hardware_capture_import/standardized/`。
- 时间缺口、全零加速度、冻结/饱和传感器样本写入 `quality=0` 和 `quality_flags`；`quality_flags` 是额外审计列，现有解析器会忽略它。

适配器不会猜测 `participant_id`、`condition`、`task_type` 或 `calibration_id`。没有独立 CAL 的标准化副本虽然能通过格式解析，但 `analysis_ready=false`，只允许用于数据链路验证、解析兼容和故障测试，不能作为正式腕角、A/B/C 效果或临床结论的输入。

`wrist_*.csv` 中的 `flex_deg/deviation_deg` 是旧处理结果。适配器只审计其范围与角度环绕，不会自动重命名为 `theta_FE/theta_RUD` 或提升为 `joint_state`，因为当前没有外部角度真值和 CAL 身份。

## 独立 CAL 文件

现场 A/B/C 的 `raw_dual_imu` 会话必须单独上传 `calibration_file`。CAL 文件使用与任务文件相同的双 IMU 列结构，元数据中的 `calibration.segments` 时间属于 CAL 文件，而不是任务文件。

必需区间：

```text
neutral
flexion
extension
ulnar_deviation
```

可选兼容区间：

```text
radial_deviation
```

最新版流程为：约 5 秒中立位，以及中立、约 20° 背伸、约 20° 掌屈和约 20° 尺偏的静态验证。任务 A/B/C 文件不要求额外中立前导段。

CAL 用于估计陀螺零偏、中立相对四元数、FE 功能轴、RUD 功能轴和第三正交轴。若同时有桡偏和尺偏，使用双向样本定轴；若只有尺偏，则由尺偏相对中立位的旋转方向确定 RUD 正向。

独立 CAL 路径以开始约 `0.5 s` 的有效加速度中值初始化倾角并固定初始航向为零。六轴数据无法独立观测绕重力方向的航向，因此 CAL 与任务之间必须保持安装位置和轴向不变，并保证启动姿态可重复；移动 IMU 或护腕后必须重新 CAL 并更换 `calibration_id`。

## joint_state.csv

最小字段：

```text
timestamp_ms,theta_FE,theta_RUD
```

推荐字段：

```text
timestamp_ms,theta_FE,theta_RUD,theta_thumb,angular_velocity,calibration_id,quality
```

选择该入口表示上游负责安装、同步、融合和标定。未安装独立拇指传感器时，`theta_thumb` 留空，系统只提供腕部暴露监测。

## mechanical.csv

当前现场主模板：

```csv
device_ms,host_unix_ms,condition,support_level,fsr_raw_adc,discomfort_nrs,safety_symptom_flag,user_continues
1000000,1787932800000,A,0,1018,1,0,1
1000020,1787932800020,A,0,1021,1,0,1
```

字段规则：

| 字段 | 规则 |
| --- | --- |
| `condition` | 若提供，必须与元数据 A/B/C 一致 |
| `support_level` | 若提供，必须与冻结条件一致 |
| `fsr_raw_adc` | 非负 ADC 代理量；兼容别名 `fsr_raw` |
| `fsr_normalized_pct` | 可选归一化代理量，范围 `0..100` |
| `discomfort_nrs` | `0..10` 主观评分，只记录，不直接停止 |
| `safety_symptom_flag` | `0/1`；疼痛、麻木、发凉、皮肤变色、明显压痕等安全停止通道 |
| `user_continues` | `0/1` 兼容状态通道 |
| `discomfort` | 旧版 `0/1` 兼容安全通道，等价并入停止触发 |

离散状态使用前值保持，连续量使用线性插值。`safety_symptom_flag`、旧 `discomfort` 和 `user_continues` 一旦提供，必须覆盖完整分析时间轴。

RFP-602 安装在护腕内层、腕关节线向肘部约 `1–2 cm` 的腕背中央。未完成载荷、迟滞、蠕变和有效接触面积标定前：

- 只输出 `fsr_raw_adc` 或 `fsr_normalized_pct` 的代理量摘要。
- 不输出 N 或 kPa。
- 不生成 `pressure_zone` 或 `max_pressure_kpa`。
- 不套用 `3.0/4.4 kPa` 筛查线。
- 不解释为肌腱力或腱鞘内压力。

为向后兼容，已完成独立标定的压力系统仍可提交：

```text
p_radial_kPa,p_dorsal_kPa,p_ulnar_kPa
```

只有这些明确以 kPa 提交的已标定通道会进入压力筛查。若提供 `cable_tension_N` 和正数 `lever_arm_m`，可计算装置外部辅助力矩 `tau_assist = cable_tension_N × lever_arm_m`；该值不是肌腱力。

## A/B/C 冻结条件

| 条件 | `support_level` | `reminder_enabled` | 输出行为 |
| --- | ---: | --- | --- |
| A | 0 | `false` | 关闭实际角度提醒，保留 `would_alert` |
| B | 1 | `false` | 关闭实际角度提醒，保留 `would_alert` |
| C | 1 | `true` | 开启实际角度提醒；提醒后只回到舒适中立位 |

顺序固定为 `A → B → C`。A/B/C 均禁用额外收紧建议。安全症状和已标定压力停止通道不受角度提醒开关影响。

比较语义：

```text
A vs B = 支撑增量影响
B vs C = 相同支撑下的提醒增量影响
A vs C = 支撑加提醒的组合影响
```

## 公共结果与时间线

会话结果新增或明确以下字段：

```text
trial_condition
channels.fsr_raw
channels.discomfort_nrs
channels.safety_symptom
fsr_proxy
metrics.alert_count
metrics.would_alert_count
```

`fsr_proxy` 只返回 `available/source/unit/calibrated_to_pressure/mean/p95/max`。未标定时 `calibrated_to_pressure=false`。

时间线字段：

```text
timestamp_ms,theta_FE,theta_RUD,quality,
angle_zone,pressure_zone,discomfort,user_continues,
activity_shadow,alert,would_alert,alert_reason,safety_stop,
fsr_raw,discomfort_nrs,safety_symptom
```

A/B 的角度事件表现为 `would_alert=true` 且 `alert=false`。没有已标定压力时，`pressure_zone=null`；只有未标定 FSR 并不使压力通道可评估。

## ML 与解释层

CNN-HMM 每 `0.5 s` 输出一次影子窗口结果，只识别：

```text
background
extension
flexion
radial_deviation
ulnar_deviation
```

低质量或低置信窗口输出 `unknown`。ML Token 的 `safety_effect` 固定为 `none`，不得触发、取消或修改确定性提醒、停止或机械动作。

默认解释器为本地模板且不联网。外部适配器只允许接收筛选后的指标和 Token，不上传原始 IMU，其控制权限始终为 `none`。

## 隐私与证据边界

人体数据必须去标识化并确认知情同意。事件文件不保存实际键盘输入文本。接口通过、会话接受、FSR 代理量存在或 `would_alert` 出现，都不能被解释为临床有效性、疾病预测或人体安全证明。
