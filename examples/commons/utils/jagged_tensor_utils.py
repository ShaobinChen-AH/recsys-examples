from typing import Dict

import torch
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


def embeddings_to_jt_dict(
    embeddings: torch.Tensor,
    features: KeyedJaggedTensor,
) -> Dict[str, JaggedTensor]:
    """
    Preserve per-key metadata from the input KJT so TorchRec does not
    recompute length_per_key from CUDA lengths and trigger a D2H sync.
    """

    embeddings_kjt = KeyedJaggedTensor(
        keys=features.keys(),
        values=embeddings,
        lengths=features.lengths(),
        offsets=features.offsets(),
        stride=features.stride(),
        stride_per_key=features.stride_per_key(),
        length_per_key=features.length_per_key(),
        offset_per_key=features.offset_per_key(),
    )
    return embeddings_kjt.to_dict()

