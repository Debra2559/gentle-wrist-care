"""Replaceable, non-controlling explanation providers.

The default provider is deterministic and local.  The optional HTTP provider uses
an OpenAI-compatible chat-completions contract, but it is disabled unless the
caller explicitly selects it and supplies endpoint/model environment variables.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Mapping, Protocol, Sequence


_ALLOWED_EVIDENCE = {"bench", "replay", "simulation", "human"}


@dataclass(frozen=True)
class ExplanationRequest:
    schema_version: str
    request_id: str
    task: str
    language: str
    model: str
    evidence_type: str
    facts: dict[str, object]
    constraints: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["constraints"] = list(self.constraints)
        return payload


@dataclass(frozen=True)
class ExplanationResponse:
    schema_version: str
    summary: str
    observations: tuple[str, ...]
    limitations: tuple[str, ...]
    next_steps: tuple[str, ...]
    safety_effect: str = "none"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("observations", "limitations", "next_steps"):
            payload[key] = list(payload[key])
        return payload


class ExplanationProvider(Protocol):
    provider_name: str
    model: str
    makes_external_call: bool

    def generate(self, request: ExplanationRequest) -> ExplanationResponse:
        ...


class TemplateExplanationProvider:
    provider_name = "local_template"
    makes_external_call = False

    def __init__(self, model: str = "shewrist-template-v1") -> None:
        self.model = model

    def generate(self, request: ExplanationRequest) -> ExplanationResponse:
        facts = request.facts
        metrics = facts.get("deterministic_metrics", {})
        ml = facts.get("ml_shadow", {})
        alert_count = int(metrics.get("alert_count", 0)) if isinstance(metrics, Mapping) else 0
        valid_pct = metrics.get("valid_sample_pct") if isinstance(metrics, Mapping) else None
        accepted = int(ml.get("accepted_window_count", 0)) if isinstance(ml, Mapping) else 0
        rejected = int(ml.get("rejected_window_count", 0)) if isinstance(ml, Mapping) else 0
        valid_text = "未知" if valid_pct is None else f"{float(valid_pct):.1f}%"
        return ExplanationResponse(
            schema_version="1.0",
            summary=f"本次离线回放有效样本率为 {valid_text}，确定性链记录 {alert_count} 个提醒事件。",
            observations=(
                f"影子动作模型接受 {accepted} 个窗口、拒识 {rejected} 个窗口。",
                "角度与压力提醒仅由确定性状态机生成，动作模型结果只作上下文描述。",
            ),
            limitations=(
                "结果来自当前证据类型，不能解释为疾病风险、疗效或人体安全结论。",
                "未接入目标硬件时，时间同步、安装滑移、压力和释放能力仍未获得硬件证据。",
            ),
            next_steps=(
                "复核低质量区间、拒识窗口和确定性提醒时间点。",
                "接入生产解释服务时只替换 provider、model 和 endpoint 配置，不修改算法主链。",
            ),
        )


class OpenAICompatibleExplanationProvider:
    provider_name = "openai_compatible"
    makes_external_call = True

    def __init__(self, endpoint: str, api_key: str, model: str, timeout_s: float = 30.0) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("explanation endpoint must be an HTTP(S) URL")
        if not api_key:
            raise ValueError("explanation API key is empty")
        if not model:
            raise ValueError("explanation model is empty")
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = float(timeout_s)

    def generate(self, request: ExplanationRequest) -> ExplanationResponse:
        system = (
            "You summarize structured wrist-exposure engineering facts. Return JSON only with keys "
            "schema_version, summary, observations, limitations, next_steps, safety_effect. "
            "Never diagnose disease, claim treatment benefit, or issue alarm/mechanical commands. "
            "safety_effect must be 'none'."
        )
        body = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(request.to_dict(), ensure_ascii=False)},
            ],
        }
        http_request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"explanation API request failed: {exc}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("explanation API returned an invalid chat-completions response") from exc
        return validate_explanation_response(parsed)


def _strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a list of strings")
    items = tuple(str(item).strip() for item in value if str(item).strip())
    if any(len(item) > 1000 for item in items):
        raise ValueError(f"{field} contains an overlong item")
    return items


def validate_explanation_response(payload: Mapping[str, object]) -> ExplanationResponse:
    if str(payload.get("safety_effect", "none")) != "none":
        raise ValueError("explanation provider attempted to claim safety authority")
    summary = str(payload.get("summary", "")).strip()
    if not summary or len(summary) > 2000:
        raise ValueError("explanation summary is empty or overlong")
    return ExplanationResponse(
        schema_version=str(payload.get("schema_version", "1.0")),
        summary=summary,
        observations=_strings(payload.get("observations", []), "observations"),
        limitations=_strings(payload.get("limitations", []), "limitations"),
        next_steps=_strings(payload.get("next_steps", []), "next_steps"),
        safety_effect="none",
    )


def build_explanation_request(
    analysis: Mapping[str, object],
    model: str,
    language: str = "zh-CN",
) -> ExplanationRequest:
    evidence_type = str(analysis.get("evidence_type", "replay"))
    if evidence_type not in _ALLOWED_EVIDENCE:
        raise ValueError("unsupported evidence_type")
    deterministic = analysis.get("deterministic_control", {})
    ml_shadow = analysis.get("ml_shadow", {})
    metrics = deterministic.get("metrics", {}) if isinstance(deterministic, Mapping) else {}
    alerts = deterministic.get("alerts", []) if isinstance(deterministic, Mapping) else []
    tokens = ml_shadow.get("tokens", []) if isinstance(ml_shadow, Mapping) else []
    safe_metrics = {
        key: metrics.get(key)
        for key in (
            "task_duration_s",
            "valid_sample_pct",
            "P_high_pct",
            "D_FE_deg_s",
            "D_RUD_deg_s",
            "L_max_s",
            "alert_count",
            "safety_stop_count",
        )
        if isinstance(metrics, Mapping) and key in metrics
    }
    safe_tokens = []
    if isinstance(tokens, Sequence):
        for token in tokens:
            if isinstance(token, Mapping):
                safe_tokens.append(
                    {
                        key: token.get(key)
                        for key in ("event_type", "duration_ms", "confidence", "mean_quality", "evidence_type")
                    }
                )
    facts = {
        "deterministic_metrics": safe_metrics,
        "alert_count": len(alerts) if isinstance(alerts, Sequence) else 0,
        "ml_shadow": {
            "operating_mode": ml_shadow.get("operating_mode") if isinstance(ml_shadow, Mapping) else "shadow",
            "accepted_window_count": ml_shadow.get("accepted_window_count", 0) if isinstance(ml_shadow, Mapping) else 0,
            "rejected_window_count": ml_shadow.get("rejected_window_count", 0) if isinstance(ml_shadow, Mapping) else 0,
            "tokens": safe_tokens,
        },
        "control_policy": analysis.get("control_policy", {}),
    }
    return ExplanationRequest(
        schema_version="1.0",
        request_id=str(analysis.get("session_id", "offline-analysis")),
        task="non_clinical_engineering_summary",
        language=language,
        model=model,
        evidence_type=evidence_type,
        facts=facts,
        constraints=(
            "Do not diagnose or estimate disease risk.",
            "Do not create, suppress, or modify alerts or mechanical actions.",
            "Describe rejected or low-quality evidence explicitly.",
            "Return safety_effect=none.",
        ),
    )


def provider_from_config(config: Mapping[str, object]) -> ExplanationProvider:
    provider = str(config.get("provider", "template"))
    model = str(config.get("model", "shewrist-template-v1"))
    if provider == "template":
        return TemplateExplanationProvider(model)
    if provider != "openai_compatible":
        raise ValueError(f"unsupported explanation provider: {provider}")
    if not bool(config.get("enabled", False)):
        raise ValueError("external explanation API is disabled")
    endpoint = os.environ.get(str(config.get("endpoint_env", "SHEWRIST_LLM_ENDPOINT")), "")
    api_key = os.environ.get(str(config.get("api_key_env", "SHEWRIST_LLM_API_KEY")), "")
    model = os.environ.get(str(config.get("model_env", "SHEWRIST_LLM_MODEL")), model)
    return OpenAICompatibleExplanationProvider(endpoint, api_key, model, float(config.get("timeout_s", 30.0)))


def explain_analysis(analysis: Mapping[str, object], config: Mapping[str, object]) -> dict[str, object]:
    provider = provider_from_config(config)
    request = build_explanation_request(analysis, provider.model, str(config.get("language", "zh-CN")))
    response = provider.generate(request)
    return {
        "schema_version": "1.0",
        "provider": provider.provider_name,
        "model": provider.model,
        "api_called": provider.makes_external_call,
        "request": request.to_dict(),
        "response": response.to_dict(),
        "control_authority": "none",
    }
