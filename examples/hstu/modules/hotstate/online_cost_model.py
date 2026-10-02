"""Online, measurement-driven cost model for HotState decisions.

The model deliberately has no CUDA dependency.  Runtime adapters feed it
observations collected around real inference batches and real state movement.
It uses small recursive least-squares models so estimates update online without
retaining an unbounded training set or requiring NumPy/sklearn.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


MIB = 1024.0 * 1024.0


@dataclass
class BatchCostObservation:
    latency_ms: float
    history_tokens: int
    num_candidates: int
    embedding_requests: int
    embedding_misses: int
    kv_miss_tokens: int
    concurrent_transfer_bytes: int = 0
    transfer_queue_depth: int = 0
    hbm_pressure: float = 0.0
    host_memory_pressure: float = 0.0


class RecursiveLeastSquares:
    """Bounded non-negative RLS with exponential forgetting."""

    def __init__(
        self,
        feature_names: Sequence[str],
        priors: Sequence[float],
        *,
        forgetting: float = 0.985,
        initial_variance: float = 8.0,
        non_negative: bool = True,
    ) -> None:
        if len(feature_names) != len(priors):
            raise ValueError("feature_names and priors must have equal length")
        self.feature_names = tuple(feature_names)
        self.theta = [float(value) for value in priors]
        self.forgetting = float(forgetting)
        self.non_negative = bool(non_negative)
        size = len(self.theta)
        self.covariance = [
            [float(initial_variance) if row == col else 0.0 for col in range(size)]
            for row in range(size)
        ]
        self.samples = 0
        self.error_ewma_ms = 0.0

    def predict(self, features: Sequence[float]) -> float:
        return sum(weight * float(value) for weight, value in zip(self.theta, features))

    def update(self, features: Sequence[float], target: float) -> float:
        x = [float(value) for value in features]
        if len(x) != len(self.theta):
            raise ValueError("feature vector has the wrong length")
        prediction = self.predict(x)
        error = float(target) - prediction

        px = [
            sum(self.covariance[row][col] * x[col] for col in range(len(x)))
            for row in range(len(x))
        ]
        denominator = self.forgetting + sum(x[i] * px[i] for i in range(len(x)))
        if denominator <= 1e-12:
            return prediction
        gain = [value / denominator for value in px]
        for index in range(len(self.theta)):
            updated = self.theta[index] + gain[index] * error
            if self.non_negative:
                updated = max(0.0, updated)
            # A corrupt timing sample must not permanently destabilize policy.
            self.theta[index] = min(updated, 1_000_000.0)

        x_times_p = [
            sum(x[row] * self.covariance[row][col] for row in range(len(x)))
            for col in range(len(x))
        ]
        self.covariance = [
            [
                (
                    self.covariance[row][col]
                    - gain[row] * x_times_p[col]
                )
                / self.forgetting
                for col in range(len(x))
            ]
            for row in range(len(x))
        ]
        self.samples += 1
        alpha = 0.1
        self.error_ewma_ms = (
            abs(error)
            if self.samples == 1
            else (1.0 - alpha) * self.error_ewma_ms + alpha * abs(error)
        )
        return prediction

    def coefficients(self) -> Dict[str, float]:
        return dict(zip(self.feature_names, self.theta))


class OnlineSystemCostModel:
    """Learns inference, movement, congestion, and capacity costs online."""

    BATCH_FEATURES = (
        "intercept_ms",
        "history_k_tokens",
        "candidate_hundreds",
        "embedding_request_hundreds",
        "embedding_miss_hundreds",
        "kv_uncached_k_tokens",
        "concurrent_transfer_mib",
        "transfer_queue_depth",
        "hbm_pressure",
        "host_memory_pressure",
    )

    def __init__(self) -> None:
        # Priors only make the first few decisions safe.  Every coefficient is
        # subsequently updated by measured batches and exposed with confidence.
        self.batch_model = RecursiveLeastSquares(
            self.BATCH_FEATURES,
            # 150 ms / 100 misses preserves the previous 1.5 ms-per-miss
            # cold-start assumption.  It is a visible prior, not a permanent
            # constant, and rapidly yields to measured latency observations.
            priors=(1.0, 0.5, 0.1, 0.02, 150.0, 3.5, 0.04, 0.05, 0.1, 0.1),
            forgetting=0.99,
            initial_variance=4.0,
        )
        self.transfer_models: Dict[Tuple[str, str], RecursiveLeastSquares] = {}
        self.observed_latency_ewma_ms = 0.0
        self.observed_tail_ewma_ms = 0.0
        self.shadow_price_ms_per_byte = 0.0
        self.last_hbm_pressure = 0.0
        self.last_transfer_queue_depth = 0
        self._last_batch_features: Optional[List[float]] = None
        self.last_prediction_ms = 0.0
        self.last_observation_ms = 0.0

    @staticmethod
    def _batch_features(observation: BatchCostObservation) -> List[float]:
        return [
            1.0,
            max(0.0, float(observation.history_tokens) / 1000.0),
            max(0.0, float(observation.num_candidates) / 100.0),
            max(0.0, float(observation.embedding_requests) / 100.0),
            max(0.0, float(observation.embedding_misses) / 100.0),
            max(0.0, float(observation.kv_miss_tokens) / 1000.0),
            max(0.0, float(observation.concurrent_transfer_bytes) / MIB),
            max(0.0, float(observation.transfer_queue_depth)),
            min(1.0, max(0.0, float(observation.hbm_pressure))),
            min(1.0, max(0.0, float(observation.host_memory_pressure))),
        ]

    def observe_batch(self, observation: BatchCostObservation) -> None:
        if observation.latency_ms < 0.0:
            raise ValueError("latency_ms must be non-negative")
        features = self._batch_features(observation)
        self.last_prediction_ms = self.batch_model.predict(features)
        self.batch_model.update(features, float(observation.latency_ms))
        self.last_observation_ms = float(observation.latency_ms)
        self._last_batch_features = features
        self.last_hbm_pressure = features[-2]
        self.last_transfer_queue_depth = int(observation.transfer_queue_depth)
        alpha = 0.08
        if self.batch_model.samples == 1:
            self.observed_latency_ewma_ms = float(observation.latency_ms)
            self.observed_tail_ewma_ms = float(observation.latency_ms)
        else:
            self.observed_latency_ewma_ms = (
                (1.0 - alpha) * self.observed_latency_ewma_ms
                + alpha * float(observation.latency_ms)
            )
            # Asymmetric EWMA follows high latency quickly and decays slowly.
            tail_alpha = 0.20 if observation.latency_ms > self.observed_tail_ewma_ms else 0.02
            self.observed_tail_ewma_ms = (
                (1.0 - tail_alpha) * self.observed_tail_ewma_ms
                + tail_alpha * float(observation.latency_ms)
            )

    def _transfer_model(self, state_class: str, direction: str) -> RecursiveLeastSquares:
        key = (str(state_class), str(direction))
        model = self.transfer_models.get(key)
        if model is None:
            model = RecursiveLeastSquares(
                ("fixed_ms", "transfer_mib", "queue_depth"),
                priors=(0.0, 0.04, 0.0),
                forgetting=0.98,
                initial_variance=8.0,
            )
            self.transfer_models[key] = model
        return model

    def observe_transfer(
        self,
        *,
        state_class: str,
        direction: str,
        bytes_moved: int,
        elapsed_ms: float,
        queue_depth: int = 0,
    ) -> None:
        if bytes_moved <= 0 or elapsed_ms < 0.0:
            return
        model = self._transfer_model(state_class, direction)
        model.update(
            [1.0, float(bytes_moved) / MIB, max(0.0, float(queue_depth))],
            float(elapsed_ms),
        )

    def estimate_transfer_ms(
        self,
        *,
        state_class: str,
        direction: str,
        bytes_moved: int,
        queue_depth: Optional[int] = None,
    ) -> float:
        if bytes_moved <= 0:
            return 0.0
        model = self._transfer_model(state_class, direction)
        depth = self.last_transfer_queue_depth if queue_depth is None else int(queue_depth)
        return max(
            0.0,
            model.predict([1.0, float(bytes_moved) / MIB, max(0.0, float(depth))]),
        )

    def embedding_miss_ms(self, misses: int = 1) -> float:
        coefficient = self.batch_model.coefficients()["embedding_miss_hundreds"]
        return max(0.0, coefficient * max(0, int(misses)) / 100.0)

    def kv_recompute_ms(self, uncached_tokens: int) -> float:
        coefficient = self.batch_model.coefficients()["kv_uncached_k_tokens"]
        return max(0.0, coefficient * max(0, int(uncached_tokens)) / 1000.0)

    def kv_residency_miss_ms(self, bytes_moved: int, queue_depth: int = 1) -> float:
        """Estimated critical-path cost of recovering a non-HBM KV object."""
        coefficients = self.batch_model.coefficients()
        online_stall = (
            coefficients["concurrent_transfer_mib"] * max(0, bytes_moved) / MIB
            + coefficients["transfer_queue_depth"] * max(0, queue_depth)
        )
        explicit_transfer = self.estimate_transfer_ms(
            state_class="kv",
            direction="onload",
            bytes_moved=bytes_moved,
            queue_depth=queue_depth,
        )
        confidence = self.confidence
        return max(
            0.0,
            confidence * online_stall + (1.0 - confidence) * explicit_transfer,
        )

    def concurrent_transfer_penalty_ms(self, bytes_moved: int, queue_depth: int) -> float:
        coefficients = self.batch_model.coefficients()
        return max(
            0.0,
            coefficients["concurrent_transfer_mib"] * max(0, bytes_moved) / MIB
            + coefficients["transfer_queue_depth"] * max(0, queue_depth),
        )

    def observe_allocation(
        self,
        *,
        candidate_densities_ms_per_byte: Iterable[float],
        selected_densities_ms_per_byte: Iterable[float],
        used_bytes: int,
        budget_bytes: int,
    ) -> None:
        if budget_bytes <= 0:
            return
        pressure = min(1.0, max(0.0, float(used_bytes) / float(budget_bytes)))
        self.last_hbm_pressure = pressure
        candidates = sorted(
            (max(0.0, float(value)) for value in candidate_densities_ms_per_byte),
            reverse=True,
        )
        selected = [
            max(0.0, float(value)) for value in selected_densities_ms_per_byte
        ]
        if pressure >= 0.90 and selected:
            target = min(selected)
        elif candidates and pressure >= 0.75:
            target = candidates[min(len(candidates) - 1, max(0, int(len(candidates) * pressure) - 1))]
        else:
            target = 0.0
        alpha = 0.20
        self.shadow_price_ms_per_byte = (
            (1.0 - alpha) * self.shadow_price_ms_per_byte + alpha * target
        )

    def opportunity_cost_ms(self, footprint_bytes: int) -> float:
        return max(0.0, int(footprint_bytes) * self.shadow_price_ms_per_byte)

    @property
    def confidence(self) -> float:
        # Sixteen measured batches are enough to dominate cold-start priors;
        # confidence remains explicit rather than pretending calibration is done.
        return min(1.0, self.batch_model.samples / 16.0)

    def snapshot(self) -> dict:
        transfer_models = {}
        for (state_class, direction), model in sorted(self.transfer_models.items()):
            coefficients = model.coefficients()
            per_mib = coefficients["transfer_mib"]
            transfer_models[f"{state_class}:{direction}"] = {
                "samples": model.samples,
                "fixed_ms": coefficients["fixed_ms"],
                "ms_per_mib": per_mib,
                "effective_bandwidth_gib_per_s": (
                    1000.0 / per_mib / 1024.0 if per_mib > 0.0 else None
                ),
                "queue_penalty_ms": coefficients["queue_depth"],
                "error_ewma_ms": model.error_ewma_ms,
            }
        return {
            "model": "online_recursive_least_squares",
            "batch_samples": self.batch_model.samples,
            "confidence": self.confidence,
            "calibration_state": (
                "online" if self.confidence >= 1.0 else "cold_start_blend"
            ),
            "batch_coefficients": self.batch_model.coefficients(),
            "batch_error_ewma_ms": self.batch_model.error_ewma_ms,
            "observed_latency_ewma_ms": self.observed_latency_ewma_ms,
            "observed_tail_ewma_ms": self.observed_tail_ewma_ms,
            "last_prediction_ms": self.last_prediction_ms,
            "last_observation_ms": self.last_observation_ms,
            "last_absolute_error_ms": abs(
                self.last_observation_ms - self.last_prediction_ms
            ),
            "last_observation_features": (
                dict(zip(self.BATCH_FEATURES, self._last_batch_features))
                if self._last_batch_features is not None else {}
            ),
            "shadow_price_ms_per_byte": self.shadow_price_ms_per_byte,
            "last_hbm_pressure": self.last_hbm_pressure,
            "last_transfer_queue_depth": self.last_transfer_queue_depth,
            "transfer_models": transfer_models,
        }
