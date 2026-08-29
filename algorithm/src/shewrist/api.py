"""FastAPI transport for the SheWrist asynchronous offline-analysis backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from .backend import ALGORITHM_RELEASE, SCHEMA_VERSION, AnalysisService, BackendError, BackendSettings
from .api_models import (
    CalibrationResult,
    ErrorEnvelope,
    HealthResponse,
    JobResponse,
    SessionResult,
    TimelineResponse,
    TokensResponse,
)


DEFAULT_MAX_UPLOAD_BYTES = 256 * 1024 * 1024
ERROR_RESPONSES = {
    400: {"model": ErrorEnvelope, "description": "Invalid request or unsupported input"},
    404: {"model": ErrorEnvelope, "description": "Job, session, or artifact not found"},
    409: {"model": ErrorEnvelope, "description": "Idempotency, session, or result-state conflict"},
    413: {"model": ErrorEnvelope, "description": "Upload too large"},
    422: {"model": ErrorEnvelope, "description": "Validation or analysis failure"},
}


def _settings_from_environment() -> BackendSettings:
    project_root = Path(os.environ.get("SHEWRIST_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    defaults = BackendSettings.default(project_root)
    output_root = Path(os.environ.get("SHEWRIST_API_OUTPUT_ROOT", defaults.output_root)).resolve()
    return BackendSettings(
        project_root=defaults.project_root,
        output_root=output_root,
        algorithm_config=Path(os.environ.get("SHEWRIST_ALGORITHM_CONFIG", defaults.algorithm_config)).resolve(),
        ml_config=Path(os.environ.get("SHEWRIST_ML_CONFIG", defaults.ml_config)).resolve(),
        explanation_config=Path(os.environ.get("SHEWRIST_EXPLANATION_CONFIG", defaults.explanation_config)).resolve(),
        model_path=Path(os.environ.get("SHEWRIST_MODEL_PATH", defaults.model_path)).resolve(),
    )


async def _read_upload(upload: UploadFile, field: str, maximum: int) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > maximum:
            raise BackendError(
                "UPLOAD_TOO_LARGE",
                f"{field} exceeds the configured upload limit.",
                field=field,
                details={"limit_bytes": maximum},
                http_status=413,
            )
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(settings: Optional[BackendSettings] = None) -> FastAPI:
    service = AnalysisService(settings or _settings_from_environment())
    maximum_upload = int(os.environ.get("SHEWRIST_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES))
    app = FastAPI(
        title="SheWrist Backend API",
        version="1.0.0",
        description=(
            "Asynchronous offline wrist-exposure analysis. The deterministic exposure engine is the only alert path; "
            "the CNN-HMM and optional explanation provider have no control authority."
        ),
    )
    app.state.analysis_service = service

    @app.exception_handler(BackendError)
    async def backend_error_handler(_request: Request, exc: BackendError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.payload())

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details = []
        for item in exc.errors():
            details.append({
                "location": ".".join(str(value) for value in item.get("loc", [])),
                "message": item.get("msg"),
                "type": item.get("type"),
            })
        error = BackendError(
            "INVALID_SCHEMA",
            "Request validation failed.",
            details={"violations": details},
            http_status=422,
        )
        return JSONResponse(status_code=error.http_status, content=error.payload())

    @app.get("/healthz", response_model=HealthResponse, operation_id="healthCheck", tags=["system"])
    def health_check() -> dict[str, object]:
        return {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "algorithm_release": ALGORITHM_RELEASE,
            "mode": "offline_async",
        }

    @app.post(
        "/api/v1/analysis-jobs",
        status_code=202,
        response_model=JobResponse,
        responses=ERROR_RESPONSES,
        operation_id="createAnalysisJob",
        tags=["analysis jobs"],
        summary="Create an asynchronous offline-analysis job",
    )
    async def create_analysis_job(
        background_tasks: BackgroundTasks,
        metadata: str = Form(..., description="JSON metadata matching SheWrist Backend API v1.0."),
        data_file: UploadFile = File(..., description="UTF-8 CSV containing joint-state or dual-IMU task samples."),
        mechanical_file: Optional[UploadFile] = File(None, description="Optional UTF-8 raw FSR, calibrated pressure, or operator-state CSV."),
        calibration_file: Optional[UploadFile] = File(None, description="Optional UTF-8 dual-IMU CAL/validation CSV for raw_dual_imu input."),
        idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    ) -> dict[str, object]:
        try:
            metadata_payload = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise BackendError(
                "INVALID_SCHEMA",
                "metadata must be valid JSON.",
                field="metadata",
                details={"line": exc.lineno, "column": exc.colno},
            ) from exc
        data = await _read_upload(data_file, "data_file", maximum_upload)
        mechanical = None
        if mechanical_file is not None:
            mechanical = await _read_upload(mechanical_file, "mechanical_file", maximum_upload)
        calibration = None
        if calibration_file is not None:
            calibration = await _read_upload(calibration_file, "calibration_file", maximum_upload)
        payload = service.create_job(
            metadata_payload=metadata_payload,
            data=data,
            data_filename=data_file.filename or "input.csv",
            mechanical=mechanical,
            mechanical_filename=mechanical_file.filename if mechanical_file is not None else None,
            calibration=calibration,
            calibration_filename=calibration_file.filename if calibration_file is not None else None,
            idempotency_key=idempotency_key,
        )
        if not payload.get("idempotent_replay", False):
            background_tasks.add_task(service.run_job, str(payload["job_id"]))
        return payload

    @app.post(
        "/api/v1/calibrations",
        status_code=201,
        response_model=CalibrationResult,
        responses=ERROR_RESPONSES,
        operation_id="createCalibration",
        tags=["calibrations"],
        summary="Record and validate a reusable calibration profile",
    )
    async def create_calibration(
        metadata: str = Form(..., description="JSON calibration metadata (calibration_id, sensors, calibration.segments)."),
        calibration_file: UploadFile = File(..., description="UTF-8 dual-IMU CAL CSV covering the neutral and functional segments."),
    ) -> dict[str, object]:
        try:
            metadata_payload = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise BackendError(
                "INVALID_SCHEMA",
                "metadata must be valid JSON.",
                field="metadata",
                details={"line": exc.lineno, "column": exc.colno},
            ) from exc
        calibration = await _read_upload(calibration_file, "calibration_file", maximum_upload)
        return service.create_calibration(
            metadata_payload=metadata_payload,
            calibration=calibration,
            calibration_filename=calibration_file.filename or "calibration.csv",
        )

    @app.get(
        "/api/v1/calibrations/{calibration_id}",
        response_model=CalibrationResult,
        responses=ERROR_RESPONSES,
        operation_id="getCalibration",
        tags=["calibrations"],
        summary="Get a stored calibration profile",
    )
    def get_calibration(calibration_id: str) -> dict[str, object]:
        return service.get_calibration(calibration_id)

    @app.get(
        "/api/v1/analysis-jobs/{job_id}",
        response_model=JobResponse,
        responses=ERROR_RESPONSES,
        operation_id="getAnalysisJob",
        tags=["analysis jobs"],
        summary="Get asynchronous job status",
    )
    def get_analysis_job(job_id: str) -> dict[str, object]:
        return service.get_job(job_id)

    @app.get(
        "/api/v1/sessions/{session_id}",
        response_model=SessionResult,
        responses=ERROR_RESPONSES,
        operation_id="getSessionAnalysis",
        tags=["sessions"],
        summary="Get the normalized analysis result",
    )
    def get_session_analysis(session_id: str) -> dict[str, object]:
        return service.get_result(session_id)

    @app.get(
        "/api/v1/sessions/{session_id}/timeline",
        response_model=TimelineResponse,
        responses=ERROR_RESPONSES,
        operation_id="getSessionTimeline",
        tags=["sessions"],
        summary="Get a paginated chart-ready timeline",
    )
    def get_session_timeline(
        session_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(1000, ge=1, le=5000),
    ) -> dict[str, object]:
        return service.get_timeline(session_id, offset, limit)

    @app.get(
        "/api/v1/sessions/{session_id}/tokens",
        response_model=TokensResponse,
        responses=ERROR_RESPONSES,
        operation_id="getSessionTokens",
        tags=["sessions"],
        summary="Get non-controlling CNN-HMM inertial tokens",
    )
    def get_session_tokens(session_id: str) -> dict[str, object]:
        return service.get_tokens(session_id)

    @app.get(
        "/api/v1/sessions/{session_id}/artifacts/{name}",
        responses=ERROR_RESPONSES,
        operation_id="downloadSessionArtifact",
        tags=["sessions"],
        summary="Download an allow-listed auditable artifact",
    )
    def download_session_artifact(session_id: str, name: str) -> FileResponse:
        path = service.get_artifact(session_id, name)
        return FileResponse(path, filename=path.name)

    return app


app = create_app()