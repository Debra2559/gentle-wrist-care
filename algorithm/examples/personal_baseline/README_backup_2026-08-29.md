# 个人基线 · 软件端对接包

这份目录是给软件端同事的**即用交付包**：mock 数据 + 接口契约 + 可跑客户端。目标是让软件端不看算法内部就能对接"个人基线（L1–L3）"。

边界（务必转达）：本能力**纯建议、描述性**，无 ML、无控制权，不诊断、不预测疾病风险、不改任何报警/机械动作。硬件仅两个 IMU。

## 1. 目录内容

| 文件 | 说明 |
|---|---|
| `metadata_dNN.json` | 第 NN 天分析任务的 `metadata`（multipart 的一个文本字段） |
| `joint_state_dNN.csv` | 第 NN 天关节角序列（`data_file`） |
| `mechanical_dNN.csv` | 第 NN 天疼痛自评（`discomfort_nrs`）等（`mechanical_file`） |
| `post_order.json` | 建议提交顺序（按天，同一 `participant_id=DEMO01`） |
| `sample_result_d10.json` | 真实后端返回的 `GET /sessions/{id}` 响应样例（含 `personal_baseline`） |
| `personal_baseline_evolution.json` | 逐天 `personal_baseline` 演化，便于对拍 |

数据是确定性合成的（`scripts/generate_personal_baseline_examples.py`，固定随机种子），可重生成。**其中疼痛被人为造成"跟随前一天暴露"，仅用于打通链路，不代表真实相关强度。**

## 2. 启动服务

项目根目录执行：

```bash
PYTHONPATH=src .venv/bin/python -m uvicorn shewrist.api:app --port 8000
```

健康检查：`GET http://127.0.0.1:8000/healthz`

## 3. 需要用到的 4 个接口

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/v1/analysis-jobs` | 提交一次分析（异步，返回 `job_id`，202） |
| GET | `/api/v1/analysis-jobs/{job_id}` | 轮询任务状态（`queued/running/succeeded/failed`） |
| GET | `/api/v1/sessions/{session_id}` | 取结果（含 `personal_baseline`） |
| GET | `/api/v1/sessions/{session_id}/artifacts/personal_baseline.json` | 下载完整个人基线报告工件 |

### 3.1 提交请求（multipart/form-data）

字段：

- `metadata`：JSON 字符串（见 `metadata_dNN.json`）。**必须含 `participant_id` 才会启用个人基线。**
- `data_file`：关节角 CSV。
- `mechanical_file`（可选）：含 `discomfort_nrs`（0–10）；有它才有 L3 的疼痛序列。
- 可选请求头 `Idempotency-Key`：重复提交去重。

metadata 关键字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `session_id` | 是 | 每次会话唯一 |
| `participant_id` | 启用个人基线时必填 | 个人基线的键；跨会话累积 |
| `input_type` | 是 | 本包用 `joint_state` |
| `evidence_type` | 是 | 本包用 `simulation` |
| `options.personal_baseline_role` | 否 | `auto`(默认) / `enroll`(严格 1–5min 入组) / `update` |
| `options.baseline_target_reduction_pct` | 否 | L2 目标线下降百分比，默认 20 |

### 3.2 curl 示例（提交第 1 天）

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/analysis-jobs \
  -H "Idempotency-Key: demo01-d01" \
  -F "metadata=$(cat examples/personal_baseline/metadata_d01.json)" \
  -F "data_file=@examples/personal_baseline/joint_state_d01.csv;type=text/csv" \
  -F "mechanical_file=@examples/personal_baseline/mechanical_d01.csv;type=text/csv"
# → {"job_id":"job_xx…","status":"queued","status_url":"…","result_url":"…"}

# 轮询直到 succeeded，然后取结果：
curl -s http://127.0.0.1:8000/api/v1/sessions/DEMO01-d01 | python -m json.tool
```

### 3.3 一键跑完整 10 天（stdlib 客户端）

```bash
.venv/bin/python scripts/post_personal_baseline_examples.py
# 或指定地址： SHEWRIST_BASE_URL=http://host:port .venv/bin/python scripts/post_personal_baseline_examples.py
```

## 4. 响应里的 `personal_baseline`

`GET /sessions/{id}` 结果新增字段 `personal_baseline`（无 `participant_id` 时为 `null`）。结构与字段含义见 `docs/api_personal_baseline.md` 第 3.9 / 5.3 节。要点：

```jsonc
"personal_baseline": {
  "participant_id": "DEMO01",
  "status": "provisional|established|rejected|not_established",
  "role": "auto",
  "update_applied": true,
  "session_count": 10,
  "observed_minutes": 15.0,
  "relative_exposure": { "dose_rate_deg_s_per_min": { "today":…, "baseline":…, "pct_vs_baseline":… }, … },
  "goal_line": { "target_reduction_pct": 20.0, "targets": { … } },
  "symptom_association": { "status":"evaluable", "pearson_r":…, "spearman_r":…, "pearson_r_ci95":[…], "n_pairs":9, … },
  "exposure_tolerance": { "status":"evaluable", "tolerance_exposure":…, … },
  "suggestions": [ { "code":…, "message":…, "control_effect":"none", "requires_human_action":true } ],
  "control_effect": "none",
  "artifact_url": "/api/v1/sessions/DEMO01-d10/artifacts/personal_baseline.json"
}
```

## 5. 用这份 mock 数据能观察到的行为（对拍基准）

同一 `participant_id`，随天数累积：

| 天 | status | symptom_association | pearson_r | tolerance |
|---|---|---|---|---|
| 1–7 | provisional | **not_evaluable** | — | — |
| 8 | provisional | **evaluable** | ~0.99 | ~446 |
| 10 | provisional | evaluable | ~0.98（CI≈[0.92,1.0], n=9） | ~383 |

关键契约点，请软件端据此写健壮 UI：

- **L3 前期一定是 `not_evaluable`**（配对天 < 7）。UI 必须能优雅显示"数据积累中"，不能当报错。
- `relative_exposure` / `goal_line` 从第 1 天就有（L1/L2）。
- 每个 `suggestions` 都带 `control_effect:"none"`、`requires_human_action:true`——**只展示、不自动执行**。
- `status` 到 `established` 需累计有效分钟 ≥ 30min，否则一直 `provisional`（正常现象）。

## 6. 重生成数据

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_personal_baseline_examples.py
```
