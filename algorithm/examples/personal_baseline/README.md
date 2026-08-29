# 个人基线 API · 软件端对接指南

> 面向：负责接入的**软件工程师**。你不需要懂算法内部，照这份文档就能对接。
> 一句话：上传"手腕角度 + 疼痛自评"，服务返回一个 `personal_baseline` 对象，里面是"今天 vs 平时"、目标线、以及（数据够时）暴露-疼痛关联与建议。你只负责**展示**，不要自动执行任何建议。

---

## 0. 30 秒了解它是什么

- 输入：一个人（`participant_id`）**每天一次**会话，含手腕角度 CSV + 当天疼痛评分（0–10）。
- 输出：该人**跨天累积**的个人基线对象 `personal_baseline`。
- 特性：纯规则计算、**无 AI**、**无控制权**、**不诊断**。所有建议都是"仅供参考，需人来决定"。

```
每天一次:  你 POST(角度CSV + 疼痛CSV) ──▶ 服务
            服务累积该 participant 的历史 ──▶ 返回 personal_baseline
你的前端:  展示 relative_exposure / goal_line / 建议
            (数据满 8 天后) 展示 symptom_association / 疼痛容量
```

---

## 1. 最快跑通（3 步）

**第 1 步 · 启动服务**（项目根目录）
```bash
PYTHONPATH=src .venv/bin/python -m uvicorn shewrist.api:app --port 8000
curl http://127.0.0.1:8000/healthz     # 返回 {"status":"ok",...} 即正常
```

**第 2 步 · 一键把 10 天示例数据全跑一遍**
```bash
.venv/bin/python scripts/post_personal_baseline_examples.py
```
你会看到逐天输出（关键是第 8 天 `symptom_association` 从 `not_evaluable` 变 `evaluable`）：
```
day       status sessions          assoc      r  tolerance  #sugg
  1  provisional        1  not_evaluable      -          -      0
  ...
  8  provisional        8      evaluable   0.99      445.9      1
 10  provisional       10      evaluable   0.98      383.4      1
```

**第 3 步 · 看一份真实响应长什么样**
```bash
cat examples/personal_baseline/sample_result_d10.json
```

搞定。下面是对接细节。

---

## 2. 你要调的 4 个接口

| 步骤 | 方法 | 路径 | 说明 |
|---|---|---|---|
| ① 提交 | `POST` | `/api/v1/analysis-jobs` | 异步，立即返回 `job_id`（HTTP 202） |
| ② 轮询 | `GET` | `/api/v1/analysis-jobs/{job_id}` | 直到 `status` 变 `succeeded`/`failed` |
| ③ 取结果 | `GET` | `/api/v1/sessions/{session_id}` | 结果里含 `personal_baseline` |
| ④ 下报告 | `GET` | `/api/v1/sessions/{session_id}/artifacts/personal_baseline.json` | 完整报告工件（可选） |

> 提交后**不能马上取结果**，必须先轮询 ①→②→③。

---

## 3. 怎么提交一次（①）

`POST /api/v1/analysis-jobs`，`Content-Type: multipart/form-data`，三个部分：

| 表单字段 | 类型 | 必填 | 内容 |
|---|---|---|---|
| `metadata` | 文本(JSON 字符串) | 是 | 见下方 3.1，示例见 `metadata_dNN.json` |
| `data_file` | 文件(CSV) | 是 | 手腕角度，见 `joint_state_dNN.csv` |
| `mechanical_file` | 文件(CSV) | 否* | 疼痛自评，见 `mechanical_dNN.csv` |

\* 没有 `mechanical_file` 也能提交，但**没有疼痛数据就永远算不出 L3**（关联/疼痛容量）。

可选请求头：`Idempotency-Key: <任意唯一串>`（同一 key 重复提交只算一次，防重复）。

### 3.1 metadata 字段

```json
{
  "schema_version": "1.0",
  "session_id": "DEMO01-d01",
  "participant_id": "DEMO01",
  "input_type": "joint_state",
  "evidence_type": "simulation",
  "timestamp_basis": "session_relative_ms",
  "options": {
    "personal_baseline_role": "auto",
    "baseline_target_reduction_pct": 20,
    "enable_ml_shadow": true,
    "threshold_version": "engineering_v1",
    "explanation_provider": "local_template",
    "enable_external_api": false,
    "generate_charts": false
  }
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `session_id` | 是 | 每次会话唯一（重复会 409） |
| `participant_id` | **启用个人基线必填** | 同一个人跨会话用同一个值；不填则响应里 `personal_baseline=null` |
| `input_type` | 是 | 本场景固定 `joint_state` |
| `evidence_type` | 是 | 示例用 `simulation`；真人数据用 `human`（需额外合规字段） |
| `options.personal_baseline_role` | 否 | `auto`(默认，推荐)｜`enroll`(严格 1–5 分钟入组)｜`update` |
| `options.baseline_target_reduction_pct` | 否 | L2 目标线降幅%，默认 20 |

> 其余 `options.*` 照抄示例即可，是通用分析选项。

### 3.2 CSV 列格式

`data_file`（手腕角度，每行一个采样）：
```
timestamp_ms,theta_FE,theta_RUD,theta_thumb,angular_velocity,calibration_id,quality
0.0,0.0,0.0,,87.39,DEMO-CAL,1.0
100.0,8.05,3.41,,83.45,DEMO-CAL,1.0
```
- `timestamp_ms` 必须严格递增；`theta_FE`/`theta_RUD` 单位度；`quality` 0–1（<0.2 的样本不计入）。

`mechanical_file`（当天疼痛，几行即可，整段填同一个值也行）：
```
timestamp_ms,discomfort_nrs,user_continues
0.0,3.0,1
1000.0,3.0,1
```
- `discomfort_nrs` 0–10（当天疼痛自评）。服务取该会话有效值的**均值**作为这一天的疼痛点。

---

## 4. 响应里的 `personal_baseline`（③ 的重点）

`GET /api/v1/sessions/{session_id}` 返回一个大对象，你主要关心 `personal_baseline` 字段（无 `participant_id` 时为 `null`）：

```jsonc
"personal_baseline": {
  "participant_id": "DEMO01",
  "status": "provisional",          // provisional | established | rejected | not_established
  "role": "auto",
  "update_applied": true,           // false=本次质量太低未并入基线
  "session_count": 10,
  "observed_minutes": 15.0,

  // —— L1：今天 vs 你平时（第 1 天起就有）——
  "relative_exposure": {
    "dose_rate_deg_s_per_min": { "today": 120.5, "baseline": 100.0, "ratio": 1.2, "pct_vs_baseline": 20.5 },
    "abs_fe_deg_p90": { ... }, "abs_rud_deg_p90": { ... }, "P_high_pct": { ... }
  },

  // —— L2：行为目标线（第 1 天起就有）——
  "goal_line": { "target_reduction_pct": 20.0, "targets": { "dose_rate_deg_s_per_min": 80.0, ... } },

  // —— L3：暴露-疼痛关联（配对满 7 天才 evaluable）——
  "symptom_association": {
    "status": "evaluable",          // 否则 "not_evaluable" + reasons
    "pearson_r": 0.98, "spearman_r": 0.99, "pearson_r_ci95": [0.92, 1.0], "n_pairs": 9, "lag_days": 1
  },

  // —— L3：个人"疼痛容量"观察估计 ——
  "exposure_tolerance": {
    "status": "evaluable",
    "tolerance_exposure": 383.4, "elevated_median_exposure": 761.0
  },

  // —— 行为建议：只展示，不自动执行 ——
  "suggestions": [
    { "code": "exposure_pain_association_observed", "message": "……", "control_effect": "none", "requires_human_action": true }
  ],

  "control_effect": "none",
  "artifact_url": "/api/v1/sessions/DEMO01-d10/artifacts/personal_baseline.json"
}
```

字段完整定义见 `docs/api_personal_baseline.md` 第 3、5 节。

---

## 5. 前端必须处理的 4 条规则（很重要）

1. **L3 前期一定是 `not_evaluable`**：配对疼痛天数 < 7 时，`symptom_association`/`exposure_tolerance` 都是 `not_evaluable`。这是**正常状态**，UI 显示"数据积累中（还差 N 天）"，**不要当错误弹红**。
2. **L1/L2 从第 1 天就有**：`relative_exposure`、`goal_line` 一直可用，先把这两块展示出来。
3. **建议只读**：每条 `suggestions` 都带 `control_effect:"none"`、`requires_human_action:true`。只展示文案，**禁止**据此自动改设置/报警/发通知给医生。
4. **status=provisional 是常态**：升到 `established` 需累计有效数据 ≥ 30 分钟；短期一直 `provisional` 正常。`rejected` 表示本次入组质量不合格（看 `personal_baseline.json` 里的 `reasons`）。

---

## 6. 常见错误返回（都是这个结构）

错误统一返回：
```json
{ "schema_version": "1.0", "error": { "code": "...", "message": "...", "retryable": false, "field": "..." } }
```

| HTTP | code（示例） | 含义 / 处理 |
|---|---|---|
| 400 | `INVALID_SCHEMA` | metadata/CSV 字段不对，看 `field` 定位 |
| 409 | `SESSION_EXISTS` | `session_id` 重复，换一个 |
| 409 | `RESULT_NOT_READY` | 结果没好，继续轮询 job |
| 404 | `SESSION_NOT_FOUND` / `ARTIFACT_NOT_FOUND` | 路径/名字不对 |
| 422 | `INSUFFICIENT_VALID_DATA` | 会话太短或有效样本太少 |

---

## 7. curl 速查

```bash
# 提交第 1 天
curl -s -X POST http://127.0.0.1:8000/api/v1/analysis-jobs \
  -H "Idempotency-Key: demo01-d01" \
  -F "metadata=$(cat examples/personal_baseline/metadata_d01.json)" \
  -F "data_file=@examples/personal_baseline/joint_state_d01.csv;type=text/csv" \
  -F "mechanical_file=@examples/personal_baseline/mechanical_d01.csv;type=text/csv"

# 轮询（把 job_id 换成上一步返回的）
curl -s http://127.0.0.1:8000/api/v1/analysis-jobs/job_xxxxxxxx

# 取结果
curl -s http://127.0.0.1:8000/api/v1/sessions/DEMO01-d01 | python -m json.tool
```

---

## 8. 这个目录里有什么

| 文件 | 用途 |
|---|---|
| `metadata_d01..d10.json` | 每天的 metadata |
| `joint_state_d01..d10.csv` | 每天角度输入 |
| `mechanical_d01..d10.csv` | 每天疼痛输入 |
| `post_order.json` | 建议提交顺序 |
| `sample_result_d10.json` | **真实响应样例**（含 personal_baseline） |
| `personal_baseline_evolution.json` | 逐天演化，供对拍 |
| `README_backup_2026-08-29.md` | 上一版 README 备份 |

重生成数据（确定性，固定随机种子）：
```bash
PYTHONPATH=src .venv/bin/python scripts/generate_personal_baseline_examples.py
```

---

## 9. 两个诚实提醒

- 示例里的疼痛是**人为造得跟暴露强相关**的，只为演示打通，`r≈0.99` 不代表真实世界强度。
- 真实上线要出 L3，必须采集**同一个人、多天、带疼痛自评**的数据；否则 L3 一直 `not_evaluable`（这是设计，不是 bug）。
