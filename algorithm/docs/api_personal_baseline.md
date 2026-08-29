# 个人基线接口设计文档

本文件描述个人基线模块 `src/shewrist/baseline.py` 的工程契约：字段、输入、输出、单位与门禁语义。所有描述对齐当前代码实现（`SCHEMA_VERSION = "1.0"`）。

## 0. 定位与边界

| 项 | 值 |
|---|---|
| 模块 | `src/shewrist/baseline.py` |
| 依赖 | `numpy`、`shewrist.metrics`（复用 `exposure_metrics` / `sample_durations`） |
| 性质 | 纯确定性、**无 ML**、**无控制权**（不报警、不改安全阈值、无机械动作） |
| 输入源 | 仅两个 IMU 算出的角度（`theta_fe_deg` / `theta_rud_deg`）+ 可选自评疼痛 NRS |
| 不声称 | 疾病风险、腱鞘炎诊断、临床疗效、安全保证、组织应变 |

单位约定：角度 `deg`，时间 `s`，剂量 `deg·s`，剂量率 `deg·s/min`，占比 `%`。角度符号与 `docs/data_interface.md` 一致（FE 正=背伸，RUD 正=尺偏）。

分层：

- **L1 描述性基线**：个人暴露的自适应分位数（“今天 vs 你平时”）。
- **L2 目标线**：相对基线的行为目标（默认降 20%），非医学阈值。
- **L3 症状联动**：n-of-1 关联与观察性“疼痛容量”估计；数据不足时返回 `not_evaluable`。

## 1. 配置字段 `config["personal_baseline"]`

来自 `config/thresholds.yaml`。

| 路径 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `percentiles` | list[int] | `[50, 90]` | 会话内对 \|FE\|、\|RUD\| 计算的分位数 |
| `tracked_metrics` | list[str] | 见下 | 基线追踪并做 EWMA 的指标名 |
| `enrollment.min_valid_minutes` | float | `1.0` | 入组最少有效分钟，否则 `rejected` |
| `enrollment.max_session_minutes` | float | `5.0` | 入组有效分钟上限；超过 → `rejected`（硬门禁） |
| `enrollment.min_valid_sample_pct` | float | `60.0` | 入组最低有效样本占比 |
| `adaptive.ewma_alpha` | float | `0.3` | 更新权重（越大越跟新会话） |
| `adaptive.established_minutes_min` | float | `30.0` | 累计有效分钟达此值 → `established` |
| `symptom.primary_metric` | str | `dose_rate_deg_s_per_min` | 疼痛容量估计用的主暴露指标 |
| `symptom.lag_days` | int | `1` | 暴露→疼痛滞后天数（今天暴露关联次日疼痛） |
| `symptom.min_paired_days` | int | `7` | 关联/耐受最少配对天数 |
| `symptom.min_group_days` | int | `3` | 耐受估计每组（高/低疼痛）最少天数 |
| `symptom.tolerance_percentile` | float | `75` | 非升高日暴露取此分位作为耐受线 |
| `symptom.bootstrap_iterations` | int | `2000` | 相关系数 CI 的自助采样次数 |
| `symptom.seed` | int | `20260829` | 自助采样种子（结果可复现） |
| `goal.default_target_reduction_pct` | float | `20.0` | L2 目标线默认下降百分比 |
| `evidence.control_effect` / `evidence.note` | str | — | 证据边界文案 |

`tracked_metrics` 默认四项：`abs_fe_deg_p90`、`abs_rud_deg_p90`、`dose_rate_deg_s_per_min`、`P_high_pct`。

## 2. 数据类型 `PersonalBaseline`

```python
@dataclass
class PersonalBaseline:
    participant_id: str
    metrics: dict[str, float | None]   # 每个 tracked_metric 一个运行估计
    observed_minutes: float            # 累计有效分钟
    session_count: int                 # 已并入的会话数
    status: str                        # provisional | established | rejected
    reasons: list[str]                 # 仅 rejected 时非空
    schema_version: str = "1.0"
    updated_at: str | None = None      # 调用方自带时间戳，模块不生成
```

- `.to_dict()` → JSON 友好 dict；`PersonalBaseline.from_dict(payload)` → 反序列化。
- `status` 语义：`provisional`=数据未攒够；`established`=累计有效分钟 ≥ `established_minutes_min`；`rejected`=入组质量不过关（`metrics` 全 `None`）。

## 3. 接口清单

### 3.1 `session_exposure_summary(timestamp_s, theta_fe_deg, theta_rud_deg, config, quality=None)`

单次会话摘要。

输入：

| 参数 | 类型 | 说明 |
|---|---|---|
| `timestamp_s` | np.ndarray | 严格递增时间戳（秒） |
| `theta_fe_deg` | np.ndarray | 屈伸角序列 |
| `theta_rud_deg` | np.ndarray | 桡尺偏角序列 |
| `config` | Mapping | 全局配置 |
| `quality` | np.ndarray \| None | 可选质量分（`>=0.2` 视为有效） |

输出：

```jsonc
{
  "valid_minutes": 2.0,
  "valid_sample_pct": 100.0,
  "metrics": {
    "abs_fe_deg_p50": 0.0, "abs_fe_deg_p90": 0.0,
    "abs_rud_deg_p50": 0.0, "abs_rud_deg_p90": 0.0,
    "dose_rate_deg_s_per_min": 0.0,   // = D_total / valid_minutes
    "P_high_pct": 0.0                 // 超黄区时间占比
  }
}
```

`metrics` 含 p50/p90 全部；基线只追踪 `tracked_metrics` 列出的项。无有效样本时对应值为 `null`。

### 3.2 `init_personal_baseline(participant_id, summary, config, updated_at=None) -> PersonalBaseline`

入组。`summary` 为 3.1 的输出。门禁（任一触发 → `status="rejected"`，`metrics` 全 `null`）：

- `valid_minutes < min_valid_minutes` → `insufficient_valid_minutes`
- `valid_minutes > max_session_minutes` → `enrollment_session_exceeds_max_minutes`
- `valid_sample_pct < min_valid_sample_pct` → `valid_sample_pct_below_minimum`

通过后按 `established_minutes_min` 决定 `provisional` / `established`。

### 3.3 `update_personal_baseline(baseline, summary, config, updated_at=None) -> PersonalBaseline`

自适应更新。对每个 tracked 指标做 EWMA：`new = (1-α)·prev + α·value`（`prev` 为 `None` 时直接取 `value`）。累加 `observed_minutes`、`session_count`，重算 `status`。

### 3.4 `relative_exposure(today_summary, baseline) -> dict`（L1）

每个 tracked 指标一条：

```jsonc
{ "abs_fe_deg_p90": { "today": 30.0, "baseline": 24.0, "ratio": 1.25, "pct_vs_baseline": 25.0 } }
```

`baseline` 为 `null`/0 或今日缺失时，`ratio` / `pct_vs_baseline` 为 `null`。

### 3.5 `goal_line(baseline, target_reduction_pct) -> dict`（L2）

`{metric: baseline*(1 - pct/100)}`（缺失为 `null`）。行为目标，非医学阈值。

### 3.6 `symptom_exposure_association(exposure_values, pain_values, config, lag_days=None) -> dict`（L3）

输入两条等长按天序列（每日暴露、每日 NRS）。配对逻辑：`lag>0` 时 `exposure[:-lag]` 对 `pain[lag:]`。

输出：

```jsonc
{
  "status": "evaluable" | "not_evaluable",
  "reasons": ["insufficient_paired_days" | "no_variation_in_series"],
  "n_pairs": 20, "lag_days": 1,
  "pearson_r": 0.62, "spearman_r": 0.58,
  "pearson_r_ci95": [0.31, 0.83],   // 自助法，确定性种子
  "interpretation": "单受试者关联，描述性、非因果、非疾病风险"
}
```

不足 `min_paired_days` 或序列无变异 → `not_evaluable`。

### 3.7 `estimate_exposure_tolerance(exposure_values, pain_values, config, lag_days=None) -> dict`（L3）

按个人疼痛中位数分“升高/非升高”两组，取非升高日暴露的 `tolerance_percentile` 作为耐受线。

```jsonc
{
  "status": "evaluable" | "not_evaluable",
  "reasons": [...],
  "primary_metric": "dose_rate_deg_s_per_min",
  "n_pairs": 20, "lag_days": 1,
  "tolerance_exposure": 13.5,          // 观察耐受线
  "elevated_median_exposure": 27.8,    // 升高日暴露中位数（对比）
  "non_elevated_day_count": 10,
  "elevated_day_count": 10,
  "interpretation": "观察性个人统计，非临床/安全阈值"
}
```

任一组 `< min_group_days` 或总配对 `< min_paired_days` → `not_evaluable`。

### 3.8 `advisory_suggestions(relative, association, tolerance, primary_metric, config, high_exposure_pct=25.0) -> list`

每条：

```jsonc
{ "code": "...", "message": "...", "control_effect": "none", "requires_human_action": true }
```

触发码（仅行为类）：

- `exposure_above_personal_baseline`：今日主指标高于基线 ≥ `high_exposure_pct`。
- `exposure_above_personal_tolerance`：今日超耐受线（需 tolerance `evaluable`）。
- `exposure_pain_association_observed`：`pearson_r ≥ 0.4` 且关联 `evaluable`。

### 3.9 `build_personal_report(baseline, today_summary, config, exposure_series=None, pain_series=None, target_reduction_pct=None) -> dict`

主入口，汇总 L1+L2+L3。未传 `exposure_series` / `pain_series` 时，L3 两块返回 `not_evaluable`，`reasons=["no_symptom_series_provided"]`。

输出 schema：

```jsonc
{
  "schema_version": "1.0",
  "participant_id": "...",
  "baseline_status": "provisional|established|rejected",
  "baseline": { /* PersonalBaseline.to_dict */ },
  "relative_exposure": { /* 3.4 */ },
  "goal_line": { "target_reduction_pct": 20.0, "targets": { /* 3.5 */ } },
  "symptom_association": { /* 3.6 */ },
  "exposure_tolerance": { /* 3.7 */ },
  "suggestions": [ /* 3.8 */ ],
  "control_effect": "none",
  "evidence_limits": {
    "control_authority": "none",
    "ml_used": false,
    "claims": "advisory personal exposure tracking and single-subject symptom association",
    "not_claimed": ["disease risk","diagnosis of tenosynovitis","clinical efficacy","safety guarantee","tissue strain"],
    "note": "..."
  }
}
```

### 3.10 持久化

- `save_personal_baseline(path, baseline) -> None`：写 UTF-8 JSON（`ensure_ascii=False`、`indent=2`、`allow_nan=False`，自动建父目录）。
- `load_personal_baseline(path) -> PersonalBaseline`：读回并反序列化。

## 4. 典型调用时序

```
每次佩戴采集 →
  session_exposure_summary()            # 本次会话摘要
  首次: init_personal_baseline()        # 入组（1~5 min）
  之后: update_personal_baseline()      # EWMA 修正
  save_personal_baseline() / load_personal_baseline()   # 跨会话持久化
每日回访记录 NRS（外部按天持久化为序列）→
  build_personal_report(baseline, today_summary, config, exposure_series, pain_series)
```

## 5. 后端 API 集成

个人基线已接入 `backend.py` 产物链，对外经 `api.py` 的会话结果暴露。

### 5.1 请求侧（metadata）

在 `POST /api/v1/analysis-jobs` 的 `metadata` 中新增可选字段：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `participant_id` | str | 无 | 个人基线的键；缺省则**不启用**个人基线（`personal_baseline=null`）。格式同 `session_id` |
| `options.personal_baseline_role` | str | `auto` | `auto`=有基线则更新、无则用本次会话自举（不受 5min 上限）；`enroll`=严格入组（1~5min 门禁）；`update`=按更新处理。`enroll`/`update` 需要 `participant_id` |
| `options.baseline_target_reduction_pct` | float | 配置默认(20) | L2 目标线下降百分比，取值 `[0,100)` |

会话疼痛值：若上传 `mechanical_file` 且含 `discomfort_nrs`，取该会话有效 NRS 的均值作为这一时间点的疼痛标量进入历史；无则该点 `pain=null`（不参与 L3 配对）。

### 5.2 存储位置

以 `participant_id` 为键，落在输出根下的专用目录（与 `session_id` 会话目录隔离）：

```
<output_root>/_baselines/<participant_id>/baseline.json   # PersonalBaseline
<output_root>/_baselines/<participant_id>/history.json     # 按会话累积的 {exposure, pain} 时间点
```

读改写在服务锁内串行，避免同一参与者并发会话竞争。低质量会话（有效样本率 < 入组门槛或有效分钟为 0）**不写入**基线与历史，避免污染 EWMA。

### 5.3 响应侧

`GET /api/v1/sessions/{session_id}` 的 `SessionResult` 新增 `personal_baseline` 字段（无 `participant_id` 时为 `null`）：

```jsonc
"personal_baseline": {
  "participant_id": "S01",
  "status": "provisional|established|rejected|not_established",
  "role": "auto|enroll|update",
  "update_applied": true,
  "session_count": 1,
  "observed_minutes": 2.0,
  "relative_exposure": { /* 3.4 */ },
  "goal_line": { /* 3.5 */ },
  "symptom_association": { /* 3.6 */ },
  "exposure_tolerance": { /* 3.7 */ },
  "suggestions": [ /* 3.8 */ ],
  "control_effect": "none",
  "artifact_url": "/api/v1/sessions/S01.../artifacts/personal_baseline.json"
}
```

完整报告作为审计产物 `personal_baseline.json` 写入会话目录，进入 `manifest.json` 的 `outputs` 哈希，并可经既有工件端点下载（已加入 `ARTIFACT_NAMES` 白名单）。

说明：`relative_exposure` 以“更新前的历史基线”为参照（“今天 vs 你以前的平时”），`personal_baseline.baseline` 反映更新后的当前状态；每次会话视为一个时间点，L3 的滞后配对按会话顺序进行。

## 6. 门禁与证据语义小结

- 入组：`1.0 <= valid_minutes <= 5.0` 且 `valid_sample_pct >= 60`，否则 `rejected`。
- L3 全部受 `min_paired_days`（默认 7）与 `min_group_days`（默认 3）门禁，数据不足一律 `not_evaluable`，不给凑数结论。
- 报告全程 `control_effect="none"`、`ml_used=false`；`evidence_limits.not_claimed` 硬编码疾病风险等边界。
