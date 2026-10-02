"""Planning and validation helpers for the inference HBM state envelope.

The planner accounts only for memory owned by the two state subsystems:
DynamicEmb's device tier and the paged KV cache.  Model activations and CUDA
allocator fragmentation are intentionally outside this envelope; callers must
leave the returned safety margin available for those costs.
"""

from dataclasses import dataclass
from math import ceil


MIB = 1024 * 1024
GIB = 1024 * MIB
DEFAULT_SAFETY_MARGIN_BYTES = 64 * MIB
DEFAULT_ELEMENT_BYTES = 2  # inference KV values are bfloat16/fp16


def ceil_div(numerator: int, denominator: int) -> int:
    numerator = int(numerator)
    denominator = int(denominator)
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}")
    return (numerator + denominator - 1) // denominator


def embedding_row_bytes(
    embedding_dim: int,
    element_bytes: int = DEFAULT_ELEMENT_BYTES,
    optimizer_state_bytes: int = 0,
) -> int:
    """Return the logical bytes occupied by one embedding row."""
    if embedding_dim < 0 or element_bytes < 0 or optimizer_state_bytes < 0:
        raise ValueError("embedding dimensions and byte widths must be non-negative")
    return int(embedding_dim) * (int(element_bytes) + int(optimizer_state_bytes))


def kv_page_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_tokens_per_page: int,
    element_bytes: int = DEFAULT_ELEMENT_BYTES,
) -> int:
    """Return bytes for one page across every KV layer (K and V included)."""
    if min(
        int(num_layers),
        int(num_kv_heads),
        int(head_dim),
        int(num_tokens_per_page),
        int(element_bytes),
    ) <= 0:
        raise ValueError("KV dimensions and element_bytes must be positive")
    values = (
        int(num_layers)
        * 2
        * int(num_tokens_per_page)
        * int(num_kv_heads)
        * int(head_dim)
    )
    return values * int(element_bytes)


def kv_onload_page_overhead(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_tokens_per_page: int,
    num_onload_buffer_pages: int,
    element_bytes: int = DEFAULT_ELEMENT_BYTES,
) -> int:
    """Return the device bytes reserved for the KV onload pages."""
    if int(num_onload_buffer_pages) < 0:
        raise ValueError("num_onload_buffer_pages must be non-negative")
    return kv_page_bytes(
        num_layers,
        num_kv_heads,
        head_dim,
        num_tokens_per_page,
        element_bytes,
    ) * int(num_onload_buffer_pages)


def kv_copy_buffer_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_tokens_per_chunk: int,
    num_onload_buffer_chunks: int = 1,
    num_offload_buffer_chunks: int = 8,
    element_bytes: int = DEFAULT_ELEMENT_BYTES,
) -> int:
    """Return the CUDA copy-buffer bytes allocated by the KV manager.

    The C++ manager allocates one chunk buffer per onload/offload slot.  A
    chunk contains K and V for one layer, hence it does not have a
    ``num_layers`` multiplier.
    """
    if min(
        int(num_layers),
        int(num_kv_heads),
        int(head_dim),
        int(num_tokens_per_chunk),
        int(element_bytes),
    ) <= 0:
        raise ValueError("KV chunk dimensions and element_bytes must be positive")
    if int(num_onload_buffer_chunks) < 0 or int(num_offload_buffer_chunks) < 0:
        raise ValueError("KV buffer chunk counts must be non-negative")
    chunk_bytes = (
        int(num_tokens_per_chunk)
        * 2
        * int(num_kv_heads)
        * int(head_dim)
        * int(element_bytes)
    )
    return chunk_bytes * (int(num_onload_buffer_chunks) + int(num_offload_buffer_chunks))


def safety_margin_bytes(state_budget_bytes: int) -> int:
    """Return the fixed 10% (with a 64 MiB floor) safety reservation."""
    budget = int(state_budget_bytes)
    if budget < 0:
        raise ValueError(f"state budget must be non-negative, got {budget}")
    return max(DEFAULT_SAFETY_MARGIN_BYTES, ceil(budget * 0.10))


@dataclass(frozen=True)
class PhysicalBudgetPlan:
    state_budget_bytes: int
    embedding_hbm_bytes: int
    kv_budget_bytes: int
    safety_margin_bytes: int
    kv_page_bytes: int
    kv_onload_pages: int
    kv_onload_page_bytes: int
    kv_copy_buffer_bytes: int
    blocks_in_primary_pool: int

    @property
    def kv_primary_page_bytes(self) -> int:
        return self.blocks_in_primary_pool * self.kv_page_bytes

    @property
    def planned_state_bytes(self) -> int:
        return (
            self.embedding_hbm_bytes
            + self.kv_primary_page_bytes
            + self.kv_onload_page_bytes
            + self.kv_copy_buffer_bytes
            + self.safety_margin_bytes
        )

    def as_dict(self):
        return {
            "state_budget_bytes": self.state_budget_bytes,
            "embedding_hbm_bytes": self.embedding_hbm_bytes,
            "kv_budget_bytes": self.kv_budget_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "kv_page_bytes": self.kv_page_bytes,
            "kv_onload_pages": self.kv_onload_pages,
            "kv_onload_page_bytes": self.kv_onload_page_bytes,
            "kv_copy_buffer_bytes": self.kv_copy_buffer_bytes,
            "kv_primary_page_bytes": self.kv_primary_page_bytes,
            "blocks_in_primary_pool": self.blocks_in_primary_pool,
            "planned_state_bytes": self.planned_state_bytes,
        }


def max_blocks_in_primary_pool(
    kv_budget_bytes: int,
    kv_page_bytes_value: int,
    kv_onload_page_bytes: int = 0,
    kv_copy_buffer_bytes_value: int = 0,
) -> int:
    """Return the floor page count after fixed KV overheads are reserved."""
    page_bytes = int(kv_page_bytes_value)
    if page_bytes <= 0:
        raise ValueError(f"kv_page_bytes must be positive, got {page_bytes}")
    remaining = (
        int(kv_budget_bytes)
        - int(kv_onload_page_bytes)
        - int(kv_copy_buffer_bytes_value)
    )
    return max(0, remaining // page_bytes)


def plan_physical_budget(
    state_budget_bytes: int,
    embedding_hbm_ratio: float,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_tokens_per_page: int,
    max_batch_size: int,
    max_sequence_length: int,
    num_tokens_per_chunk: int,
    num_onload_buffer_chunks: int = 1,
    num_offload_buffer_chunks: int = 8,
    element_bytes: int = DEFAULT_ELEMENT_BYTES,
    min_primary_pages: int = 1,
) -> PhysicalBudgetPlan:
    """Split one state budget into DynamicEmb HBM and KV primary pages.

    ``max_sequence_length`` is the actual model sequence length passed to the
    KV manager, including the interleaved item/action representation.
    """
    state_budget = int(state_budget_bytes)
    ratio = float(embedding_hbm_ratio)
    if state_budget <= 0:
        raise ValueError("state_budget_bytes must be positive")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"embedding_hbm_ratio must be in [0, 1], got {ratio}")
    if max_batch_size <= 0 or max_sequence_length <= 0:
        raise ValueError("max_batch_size and max_sequence_length must be positive")
    if int(num_onload_buffer_chunks) < 0 or int(num_offload_buffer_chunks) < 0:
        raise ValueError("KV buffer chunk counts must be non-negative")
    if int(min_primary_pages) < 0:
        raise ValueError("min_primary_pages must be non-negative")

    margin = safety_margin_bytes(state_budget)
    usable = state_budget - margin
    if usable <= 0:
        raise ValueError(
            "state HBM budget must exceed its safety margin: "
            f"budget={state_budget}, safety_margin={margin}"
        )
    page_bytes = kv_page_bytes(
        num_layers,
        num_kv_heads,
        head_dim,
        num_tokens_per_page,
        element_bytes,
    )
    onload_pages = ceil_div(
        int(max_batch_size) * int(max_sequence_length), int(num_tokens_per_page)
    )
    onload_bytes = kv_onload_page_overhead(
        num_layers,
        num_kv_heads,
        head_dim,
        num_tokens_per_page,
        onload_pages,
        element_bytes,
    )
    copy_bytes = kv_copy_buffer_bytes(
        num_layers,
        num_kv_heads,
        head_dim,
        num_tokens_per_chunk,
        num_onload_buffer_chunks,
        num_offload_buffer_chunks,
        element_bytes,
    )
    embedding_bytes = int(usable * ratio)
    kv_budget = usable - embedding_bytes
    blocks = max_blocks_in_primary_pool(
        kv_budget,
        page_bytes,
        onload_bytes,
        copy_bytes,
    )
    if blocks < int(min_primary_pages):
        raise ValueError(
            "state HBM budget cannot fit the requested KV fixed overhead and "
            f"{min_primary_pages} primary page(s): budget={state_budget}, "
            f"embedding={embedding_bytes}, onload={onload_bytes}, copy={copy_bytes}, "
            f"page={page_bytes}"
        )

    plan = PhysicalBudgetPlan(
        state_budget_bytes=state_budget,
        embedding_hbm_bytes=embedding_bytes,
        kv_budget_bytes=kv_budget,
        safety_margin_bytes=margin,
        kv_page_bytes=page_bytes,
        kv_onload_pages=onload_pages,
        kv_onload_page_bytes=onload_bytes,
        kv_copy_buffer_bytes=copy_bytes,
        blocks_in_primary_pool=blocks,
    )
    if plan.planned_state_bytes > state_budget:
        raise AssertionError("physical budget planner exceeded its requested envelope")
    return plan


# Explicit aliases make the accounting vocabulary easy to discover from tests
# and call sites without changing the canonical helper names above.
kv_onload_page_bytes = kv_onload_page_overhead
kv_copy_buffers_bytes = kv_copy_buffer_bytes
