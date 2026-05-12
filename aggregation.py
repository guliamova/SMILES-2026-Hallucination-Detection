"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).
"""

from __future__ import annotations

import torch


SELECTED_LAYERS: tuple[int, ...] = (14, 18, 22)

_RESP_FRAC = 0.25
_RESP_MIN = 8
_RESP_MAX = 48


def _window_length(n_real: int) -> int:
    window = max(_RESP_MIN, min(_RESP_MAX, int(round(_RESP_FRAC * n_real))))
    return min(window, n_real)


def _response_indices(attention_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return the sequence-dimension indices of the response window, on ``device``."""
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    n_real = int(real_positions.numel())
    if n_real == 0:
        return torch.zeros(1, dtype=torch.long, device=device)

    idx = real_positions[-_window_length(n_real):]
    return idx.to(device)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Multi-layer mean-pool over the response window + last-token of deepest layer."""
    device = hidden_states.device
    resp_idx = _response_indices(attention_mask, device)
    n_layers = hidden_states.size(0)

    pooled_layers: list[torch.Tensor] = []
    deepest_layer_tensor: torch.Tensor | None = None
    for layer_idx in SELECTED_LAYERS:
        li = max(0, min(layer_idx, n_layers - 1))
        layer = hidden_states[li]
        resp_tokens = layer.index_select(0, resp_idx)
        pooled = resp_tokens.mean(dim=0)
        pooled_layers.append(pooled)
        deepest_layer_tensor = layer

    assert deepest_layer_tensor is not None
    last_pos = int(resp_idx[-1].item())
    pooled_layers.append(deepest_layer_tensor[last_pos])

    return torch.cat(pooled_layers, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-layer L2 norms + adjacent-layer cosine drift + log response length."""
    device = hidden_states.device
    n_layers = hidden_states.size(0)
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)

    if real_positions.numel() == 0:
        return torch.zeros(n_layers + (n_layers - 1) + 1, device=device)

    last_pos = int(real_positions[-1].item())
    last_vecs = hidden_states[:, last_pos, :]

    norms = last_vecs.norm(p=2, dim=-1)
    a = last_vecs[:-1]
    b = last_vecs[1:]
    cos_drift = torch.nn.functional.cosine_similarity(a, b, dim=-1)

    window_len = float(_window_length(int(real_positions.numel())))
    log_window = torch.tensor(
        [torch.log(torch.tensor(window_len + 1.0)).item()],
        device=device,
    )

    return torch.cat([norms, cos_drift, log_window], dim=0).float()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask).float()

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask).float()
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
