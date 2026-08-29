"""SheWrist 2.0 interpretable wrist-exposure algorithms."""

from .dataset_registry import (
    DatasetAdapterError,
    DatasetCapabilityError,
    DatasetRegistry,
    DatasetUnavailableError,
)
from .experts import (
    ExpertContract,
    ExpertPrediction,
    ReservedExpert,
    ShadowActivityExpert,
    ValidatedFusionWeights,
    fuse_expert_probabilities,
)
from .explanation import (
    ExplanationRequest,
    ExplanationResponse,
    OpenAICompatibleExplanationProvider,
    TemplateExplanationProvider,
    explain_analysis,
)
from .baseline import (
    PersonalBaseline,
    advisory_suggestions,
    build_personal_report,
    estimate_exposure_tolerance,
    goal_line,
    init_personal_baseline,
    load_personal_baseline,
    relative_exposure,
    save_personal_baseline,
    session_exposure_summary,
    symptom_exposure_association,
    update_personal_baseline,
)
from .exposure import ExposureEngine, classify_zone
from .faults import FaultSpec, inject_faults
from .kinematics import compute_wrist_kinematics
from .metrics import exposure_metrics, intervention_efficiency
from .ml import ShadowActivityPipeline
from .pipeline import analyze_with_shadow
from .replay import verify_chunked_replay
from .session import analyze_session, prepare_public_joint_state
from .tokens import InertialToken, build_inertial_tokens
from .validation import angle_error_metrics, paired_condition_comparison

__all__ = [
    "DatasetRegistry",
    "DatasetAdapterError",
    "DatasetCapabilityError",
    "DatasetUnavailableError",
    "ExpertContract",
    "ExpertPrediction",
    "ReservedExpert",
    "ShadowActivityExpert",
    "ValidatedFusionWeights",
    "fuse_expert_probabilities",
    "PersonalBaseline",
    "session_exposure_summary",
    "init_personal_baseline",
    "update_personal_baseline",
    "relative_exposure",
    "goal_line",
    "symptom_exposure_association",
    "estimate_exposure_tolerance",
    "advisory_suggestions",
    "build_personal_report",
    "save_personal_baseline",
    "load_personal_baseline",
    "ExposureEngine",
    "classify_zone",
    "compute_wrist_kinematics",
    "exposure_metrics",
    "intervention_efficiency",
    "ShadowActivityPipeline",
    "analyze_with_shadow",
    "InertialToken",
    "build_inertial_tokens",
    "angle_error_metrics",
    "paired_condition_comparison",
    "ExplanationRequest",
    "ExplanationResponse",
    "OpenAICompatibleExplanationProvider",
    "TemplateExplanationProvider",
    "explain_analysis",
    "FaultSpec",
    "inject_faults",
    "verify_chunked_replay",
    "analyze_session",
    "prepare_public_joint_state",
]

__version__ = "0.8.0"