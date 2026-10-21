"""Configurable inference policy, independent of storage and CUDA execution."""

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from modules.hotstate.state_handle import Placement


@dataclass(frozen=True)
class PolicyConfig:
    """Experimental settings; defaults are priors, not calibrated claims.

    Set HOTSTATE_POLICY_CONFIG to a JSON file or pass a config to the controller.
    A custom ValueFunction can replace the formula without changing execution.
    """

    selection: str = "value_density"
    initial_embedding_ratio: float = 0.5
    static_embedding_ratio: float = 0.6
    benefit_weight: float = 1.0
    movement_weight: float = 1.0
    risk_weight: float = 1.0
    opportunity_weight: float = 1.0
    density_time_exponent: float = 0.0
    minimum_net_benefit_ms: float = 0.0
    reuse_interval_alpha: float = 0.2
    reuse_probability_alpha: float = 0.1
    embedding_recency_weight: float = 0.6
    reuse_horizon_epochs: float = 8.0
    bandwidth_fraction: float = 0.8
    reserve_bytes: int = 0
    kv_granularity: str = "segment"
    kv_min_primary_pages: int = 64
    kv_append_margin_pages_per_user: int = 2
    batch_priors: tuple = (1.0, 0.5, 0.1, 0.02, 150.0, 3.5, 0.04, 0.05, 0.1, 0.1)
    transfer_priors: tuple = (0.0, 0.04, 0.0)
    batch_forgetting: float = 0.99
    transfer_forgetting: float = 0.98
    confidence_samples: int = 16

    def __post_init__(self):
        if self.selection not in ("value_density", "static_partition"):
            raise ValueError("unknown HotState selection policy")
        if self.kv_granularity not in ("segment", "user"):
            raise ValueError("kv_granularity must be segment or user")
        for name in ("initial_embedding_ratio", "static_embedding_ratio",
                     "reuse_interval_alpha", "reuse_probability_alpha",
                     "embedding_recency_weight", "bandwidth_fraction"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("benefit_weight", "movement_weight", "risk_weight",
                     "opportunity_weight", "density_time_exponent",
                     "minimum_net_benefit_ms", "reserve_bytes"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.reuse_horizon_epochs <= 0 or self.confidence_samples <= 0:
            raise ValueError("reuse horizon and confidence sample count must be positive")
        if self.kv_min_primary_pages < 0 or self.kv_append_margin_pages_per_user < 0:
            raise ValueError("KV page reserve settings must be non-negative")
        for name in ("batch_forgetting", "transfer_forgetting"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        for name, size in (("batch_priors", 10), ("transfer_priors", 3)):
            values = tuple(getattr(self, name))
            if len(values) != size or any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError(f"{name} requires {size} finite non-negative values")
            object.__setattr__(self, name, values)

    def snapshot(self):
        """Return reproducible settings and a stable identity for trace pairing."""
        values = asdict(self)
        digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        return {"config_id": digest, "parameters": values}


def load_policy_config(path=None):
    """Load an explicitly supplied JSON configuration, or use default priors."""
    path = path or os.getenv("HOTSTATE_POLICY_CONFIG")
    return PolicyConfig(**json.loads(Path(path).read_text(encoding="utf-8"))) if path else PolicyConfig()


@dataclass(frozen=True)
class ValueInputs:
    """Measured/estimated components supplied to a replaceable value formula."""

    gross_benefit_ms: float
    movement_ms: float
    risk_ms: float
    opportunity_ms: float


class ValueFunction(Protocol):
    def __call__(self, inputs: ValueInputs) -> float:
        """Return the net expected benefit in milliseconds."""
        ...


class WeightedValue:
    """Default interpretable formula; all component weights are configurable."""

    def __init__(self, config):
        self.config = config

    def __call__(self, inputs):
        c = self.config
        return (c.benefit_weight * inputs.gross_benefit_ms
                - c.movement_weight * inputs.movement_ms
                - c.risk_weight * inputs.risk_ms
                - c.opportunity_weight * inputs.opportunity_ms)


@dataclass(frozen=True)
class InferenceBudget:
    """One state envelope with explicit physical capacities and overhead.

    Residency changes reuse existing pools. They do not transfer physical
    capacity between pools; future resize backends must report that separately.
    """

    state_bytes: int
    embedding_capacity_bytes: int
    kv_capacity_bytes: int
    physical_bytes: int
    fixed_overhead_bytes: int = 0
    reserve_bytes: int = 0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer byte count")
        if self.state_bytes == 0:
            raise ValueError("state_bytes must be positive")

    @classmethod
    def observe(cls, state_bytes, embedding, kv, reserve_bytes=0):
        """Measure pools once; failed introspection must not invent capacity."""
        capacity_fn = getattr(embedding, "hbm_value_capacity_bytes", None)
        embedding_capacity = (
            int(capacity_fn()) if callable(capacity_fn)
            else int(embedding.hbm_capacity_keys()) * int(embedding.row_size_bytes())
        )
        kv_capacity = int(kv.physical_kv_cache_bytes())
        physical_fn = getattr(embedding, "physical_hbm_bytes", None)
        kv_physical_fn = getattr(kv, "physical_hbm_bytes", None)
        physical = (
            int(physical_fn()) + int(kv_physical_fn())
            if callable(physical_fn) and callable(kv_physical_fn)
            else embedding_capacity + kv_capacity
        )
        return cls(
            state_bytes=int(state_bytes),
            embedding_capacity_bytes=embedding_capacity,
            kv_capacity_bytes=kv_capacity,
            physical_bytes=physical,
            fixed_overhead_bytes=max(0, physical - embedding_capacity - kv_capacity),
            reserve_bytes=int(reserve_bytes),
        )

    @property
    def residency_bytes(self):
        return max(0, min(self.embedding_capacity_bytes + self.kv_capacity_bytes,
                          self.state_bytes - self.fixed_overhead_bytes - self.reserve_bytes))

    def snapshot(self):
        return {**asdict(self), "residency_budget_bytes": self.residency_bytes,
                "physical_capacity_reallocation_supported": False,
                "physical_budget_valid": self.physical_bytes <= self.state_bytes,
                "allocation_scope": "residency_within_fixed_physical_pools"}


class InferencePolicy:
    """Select feasible state under a shared envelope and backend constraints."""

    def __init__(self, config):
        self.config = config

    def select(self, scored, budget, protected_users=()):
        protected = set(protected_users)
        by_key = {s.handle.logical_key: s for s in scored}
        selected = {}
        used = {"embedding": 0, "kv": 0}
        caps = {"embedding": budget.embedding_capacity_bytes, "kv": budget.kv_capacity_bytes}
        if self.config.selection == "static_partition":
            emb = int(budget.residency_bytes * self.config.static_embedding_ratio)
            caps = {"embedding": min(caps["embedding"], emb),
                    "kv": min(caps["kv"], budget.residency_bytes - emb)}

        def add_package(item, mandatory=False):
            package = {}
            cursor = item
            while cursor.handle.logical_key not in selected:
                h = cursor.handle
                if h.logical_key in package:
                    raise ValueError("cyclic KV retention dependency")
                package[h.logical_key] = h
                parent = h.metadata.get("retain_after")
                if parent is None:
                    break
                if parent not in by_key:
                    raise ValueError("missing KV retention dependency")
                cursor = by_key[parent]
            delta = {kind: sum(h.footprint_bytes for h in package.values()
                              if ("kv" if h.is_kv else "embedding") == kind) for kind in used}
            fits = (sum(used.values()) + sum(delta.values()) <= budget.residency_bytes
                    and all(used[k] + delta[k] <= caps[k] for k in used))
            if mandatory or fits:
                selected.update(package)
                for kind in used:
                    used[kind] += delta[kind]

        # Pin already-resident request state and unfinished writebacks. Hard
        # semantic constraints take precedence over an infeasible policy quota.
        for item in scored:
            h = item.handle
            if Placement.HBM in h.placement and (
                h.metadata.get("user_id") in protected or h.writeback_pending
                or h.metadata.get("pinned", False)
            ):
                add_package(item, mandatory=True)
        for item in sorted(scored, key=lambda s: s.score, reverse=True):
            h = item.handle
            if h.footprint_bytes <= 0 or not h.metadata.get("policy_candidate", True):
                continue
            if item.net_benefit_ms <= self.config.minimum_net_benefit_ms:
                continue
            add_package(item)
        return list(selected.values())
