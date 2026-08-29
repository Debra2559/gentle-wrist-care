# SheWrist 腕部暴露算法原型

SheWrist 当前实现的是“腕部相对姿态与腕角估计 → 确定性阈值和状态机 → 姿势暴露与工效提醒”的纯离线工程原型。它不诊断、预测或预防 De Quervain 腱鞘炎，也不输出临床风险概率。

> 工程事实口径以 `SheWrist_实现方案_医学指标与硬件软件.docx` 为总体基线，并由最新版 `SheWrist_现场采集协作与传感器安装手册2026-08-28.docx` 覆盖安装位置、采集字段、校准流程和 A/B/C 条件中的冲突内容；`SheWrist 今日进展与待办交接` 冻结当前实物与实采状态。单 Hub、BLE、定制 PCB、尺寸和续航仍是后续工程路线，不是当前已实现能力。

## 当前实现

- 现有有线原型已能连续采集双 IMU 与单点 RFP-602 原始数据；`datasets/` 中的现场文件按只读原始证据管理。
- `scripts/import_hardware_captures.py` 将采集端宽表转换为后端标准交错双节点 CSV：`device_us → device_ms`、`g → m/s²`、`deg/s → rad/s`、`arm → forearm`、`pressure_adc* → fsr_raw_adc`，并生成逐文件质量审计；转换不重采样、不改写源文件。
- 当前实采文件没有参与者、A/B/C、任务和 CAL 身份，也没有独立 CAL 记录，因此标准化副本只用于数据链路、解析兼容和故障测试，不直接进入腕角精度或效果分析。
- 目标硬件为 `2 个 IMU + 1 个 RFP-602`。
- `forearm` IMU 安装在腕关节线向肘部约 `5 cm`、允许 `4–6 cm` 的前臂背侧中线。
- `hand` IMU 安装在距腕关节线约 `2–3 cm` 的手背第 3 掌骨中段；两颗 IMU 同侧且轴向一致。
- 腕部相对姿态固定为 `q_rel = inverse(q_forearm) × q_hand`。
- `theta_FE` 正值为背伸、负值为掌屈；`theta_RUD` 正值为尺偏、负值为桡偏。
- 默认融合为六轴 Madgwick，磁力计默认关闭；九轴模式只有完成硬铁、软铁和现场环境标定后才适合启用。
- 现场 A/B/C 原始 IMU 会话使用独立 `calibration_file`。CAL 必需中立、掌屈、背伸和尺偏四类区间，桡偏可选；任务文件不需要额外中立段。
- 原始双 IMU 与机械文件支持 `timestamp_ms` 或 `device_ms`；`device_ms` 会归一为会话相对毫秒。`host_unix_ms` 仅可作为审计字段，不参与运动学。
- RFP-602 位于护腕内层、腕关节线向肘部约 `1–2 cm` 的腕背中央。未标定时只输出 ADC 或归一化代理量，不输出 N/kPa，不参与压力阈值。
- `discomfort_nrs` 为 `0–10` 记录通道；`safety_symptom_flag` 为独立的 `0/1` 停止通道。
- A 为支撑 0 档且提醒关闭，B 为支撑 1 档且提醒关闭，C 为相同支撑 1 档且提醒开启；顺序固定为 `A → B → C`。
- A/B 仍记录 `would_alert`，但不产生实际角度提醒；C 产生实际提醒。三个条件均禁止额外收紧建议。
- 轻量 `1D-CNN + HMM` 只识别五类公开腕部动作并保持 `shadow`；ML 与解释服务的提醒、停止和机械控制权限永久为 `none`。
- `config/datasets.json` 统一登记数据源、许可证、传感器要求、标签映射、用途和证据边界；窗口会保留 `dataset_id` 与 `missing_fraction`。
- 当前只有 `Upper-body movements` 是可训练的五分类活动集；Optotrak 仅接入单参与者来源工具箱角度样例，ULTRA-MoCap、OpenPack、LARa 和 MyoKi 只有专家契约与拒绝占位，未下载时不会伪装为已训练。
- 多专家融合默认禁用。只有取得带参与者、会话、CAL、动作和重戴身份的目标硬件验证集后，才能写入验证权重；缺失比例超过 `10%` 的窗口直接输出 `unknown`。

## 当前验证

- `89/89` 项单元测试通过，`compileall` 通过。
- 数据集深度审计通过：登记 7 个来源，`Upper-body movements` 产出 11 人、2,090 个窗口；Optotrak 单人样例可运行 FE/RUD 误差报告；其余公开候选未安装，当前跨数据集活动评估为 `not_evaluable`。
- 13 份实采原始 CSV 已全部转换并通过现有 `raw_dual_imu` 解析器，共 `259,755` 个双节点采样时刻、约 `726.2 s`；原始文件哈希在转换前后保持一致。
- 实采审计确认：1 份早期记录仅 `5 Hz`，其余约 `369–405 Hz`；3 份存在时间缺口；1 份包含 `6,563` 个前臂加速度全零样本；12/13 份压力多数时间达到 `4095`。两份旧角度输出无外部真值，其中第一份 RUD 有 10 次跨 `±180°` 跳变。
- 真实 HTTP 冒烟通过：A/B 实际提醒数为 0、C 为 1；三个条件的 `would_alert` 均为 1；独立 CAL、`device_ms`、原始 FSR、安全症状、时间线字段和权限隔离均通过。
- 最新 8 名合成参与者 A/B/C 联调中，`D_total` 的 A→B、B→C、A→C 平均降幅分别为 `27.66%`、`53.98%`、`66.71%`。
- 合成数据只有未标定 FSR，没有 kPa 压力证据，因此 Go/No-Go 结果为 `NOT-EVALUABLE`，不是 GO，也不是人体效果证据。
- 公开原始 IMU 留出动作方向命中为 `38/43 = 88.37%`，但没有本项目对应的独立角度真值。
- CNN macro-F1 均值约 `0.534`，HMM 后约 `0.559`；拒识后覆盖率约 `50.61%`，跨人波动明显，只能作为影子基线。

## 环境

- Python `>=3.9`
- NumPy `>=1.20`
- API 可选依赖见 `requirements-api.txt`
- 项目图表额外使用 Matplotlib `>=3.5`

安装：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-api.txt
```

## 后端 API

启动单进程服务：

```bash
PYTHONPATH=src python3 scripts/run_api.py --host 127.0.0.1 --port 8000
```

交互文档位于 `http://127.0.0.1:8000/docs`。完整接入契约见 `docs/backend_api.md`，简化版见 `Simplified_Api.md`，机器可读契约见 `docs/openapi.json`。当前文件型任务存储只支持 `--workers 1`。

现场原始 A/B/C 请求通过 `multipart/form-data` 上传：

```text
metadata         必需，JSON
data_file        必需，任务双 IMU CSV
calibration_file 必需，独立 CAL/静态验证双 IMU CSV
mechanical_file  可选，RFP-602、主观评分与安全状态 CSV
```

## 运行验证

```bash
.venv/bin/python scripts/import_hardware_captures.py \
  --source-dir datasets \
  --output-dir outputs/hardware_capture_import
.venv/bin/python scripts/audit_ml_datasets.py
.venv/bin/python scripts/evaluate_cross_dataset_activity.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python scripts/mock_api_smoke.py --port 58902
```

重新生成最新版 180 秒 A/B/C 合成数据：

```bash
.venv/bin/python scripts/generate_demo.py
```

重新导出 OpenAPI：

```bash
PYTHONPATH=src .venv/bin/python scripts/export_openapi.py
```

重新生成项目图表：

```bash
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/draw_current_architecture.py
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/visualize_project_status.py
```

## 主要输出

- `outputs/datasets/readiness_report.json`：各数据源的本地安装、适配能力、阻塞项、标签用途和 Optotrak 样例角度结果。
- `outputs/datasets/cross_dataset_activity.json`：留一数据集评估；当前因仅有一个兼容标注活动集而明确为 `not_evaluable`。
- `outputs/hardware_capture_import/audit_report.json`：实采文件哈希、格式、采样时间、压力饱和、时间缺口和传感器失效审计。
- `outputs/hardware_capture_import/standardized/`：SI 单位、交错 `forearm/hand` 节点的标准化副本；不含 CAL 时不得作为正式腕角输入。
- `outputs/demo_summary.json`：最新版合成 A/B/C 汇总与三态 Go/No-Go。
- `outputs/ml/activity_cnn_hmm_shadow.npz`：可加载的影子动作模型。
- `outputs/ml/loso_report.json`：11 折参与者留一评估。
- `outputs/offline_session/manifest.json`：离线会话输入、配置、模型和产物校验值。
- `outputs/fault_suite/fault_report.json`：系统级故障注入结果。
- `outputs/project_overview/`：当前架构、证据边界和路线图 PNG/SVG。
- `examples/synthetic_abc/`：每条件 180 秒、前 90 秒键入和后 90 秒鼠标的合成联调数据。

## 证据边界

`analysis_status=accepted` 只表示数据通过同步、校准和质量门控，不表示产品有效或医学安全。`sensor_installation.contract_validated=true` 只表示元数据声明合法，不证明实物安装正确。当前无标签实采只能证明有线双 IMU 与 RFP-602 数据链路已跑通；格式转换成功不能替代独立 CAL、外部角度真值、压力标定或 A/B/C 标签。数据集注册、专家接口或 Optotrak 单人样例通过也不等于多数据集模型已训练；融合权重仍为空且被代码门控。角度、持续时间、舒适度和已标定压力边界均为首版工程参数；目标硬件角度精度、重戴、滑移、跨轴串扰、RFP-602 载荷/迟滞/蠕变/接触面积以及人体效果仍需后续验证。