from typing import Iterable, Optional

import torch
from dynamicemb.types import AdmissionStrategy


class HotStateAdmissionStrategy(AdmissionStrategy):
    """Mutable DynamicEmb admission policy controlled by HotState."""

    def __init__(self, admit_all_when_empty: bool = True):
        self._enabled = True
        self._admit_all_when_empty = admit_all_when_empty
        self._allowed_keys_cpu = torch.empty(0, dtype=torch.int64)

        self.num_admit_calls = 0
        self.num_accepted = 0
        self.num_rejected = 0
        self.last_policy_size = 0

    def update_policy(
        self,
        item_indices: Optional[Iterable[int]] = None,
        max_admitted_keys: Optional[int] = None,
        enabled: bool = True,
        admit_all_when_empty: Optional[bool] = None,
    ) -> None:
        self._enabled = enabled
        if admit_all_when_empty is not None:
            self._admit_all_when_empty = admit_all_when_empty

        if item_indices is None:
            self._allowed_keys_cpu = torch.empty(0, dtype=torch.int64)
            self.last_policy_size = 0
            return

        selected_keys = []
        seen_keys = set()
        for raw_key in item_indices:
            key = int(raw_key)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected_keys.append(key)
            if max_admitted_keys is not None and len(selected_keys) >= max_admitted_keys:
                break

        self._allowed_keys_cpu = torch.tensor(selected_keys, dtype=torch.int64)
        self.last_policy_size = len(selected_keys)

    def admit(self, keys: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        self.num_admit_calls += 1

        if keys.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=keys.device)

        if not self._enabled:
            admit_mask = torch.ones(keys.shape, dtype=torch.bool, device=keys.device)
        elif self._allowed_keys_cpu.numel() == 0:
            admit_mask = torch.full(
                keys.shape,
                bool(self._admit_all_when_empty),
                dtype=torch.bool,
                device=keys.device,
            )
        else:
            allowed_keys = self._allowed_keys_cpu.to(
                device=keys.device,
                non_blocking=True,
            )
            if hasattr(torch, "isin"):
                admit_mask = torch.isin(keys, allowed_keys)
            else:
                admit_mask = (keys.view(-1, 1) == allowed_keys.view(1, -1)).any(dim=1)

        accepted = int(admit_mask.sum().item())
        self.num_accepted += accepted
        self.num_rejected += int(keys.numel()) - accepted
        return admit_mask

    def initialize_non_admitted_embeddings(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
    ) -> bool:
        if indices.numel() == 0:
            return False
        buffer.index_fill_(0, indices, 0.0)
        return True
