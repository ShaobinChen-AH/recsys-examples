# HotState: Unified GPU Hot-State Control Plane
from .state_handle import (
    StateHandle,
    StateType,
    Placement,
    TransferState,
    Reconstructability,
    ScoredHandle,
)
from .demand_signal import DemandSignal
from .state_registry import StateRegistry
from .global_directory import GlobalDirectory
from .online_cost_model import (
    BatchCostObservation,
    OnlineSystemCostModel,
    RecursiveLeastSquares,
)
from .value_engine import ValueEngine
from .kv_adapter import KVAdapter
from .transfer_scheduler import (
    TransferExecutionResult,
    TransferRequest,
    TransferResource,
    TransferScheduler,
    TransferStatus,
)

# The state model and budget math are CPU-testable.  Keep CUDA/DynamicEmb
# adapters optional at package-import time so those tests do not require a GPU
# runtime merely to import ``modules.hotstate.state_handle``.
try:
    from .embedding_adapter import EmbeddingAdapter
    from .hot_set_manager import HotSetManager
    from .hotstate_controller import HotStateController
    from .admission_adapter import HotStateAdmissionStrategy
except ImportError:
    EmbeddingAdapter = None
    HotSetManager = None
    HotStateController = None
    HotStateAdmissionStrategy = None

try:
    from .training_controller import TrainingHotStateController
except ImportError:
    TrainingHotStateController = None
