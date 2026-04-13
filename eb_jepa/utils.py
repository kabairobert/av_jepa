import torch


def flatten_spatio_temporal(x: torch.Tensor):
    """Flatten [B, C, T, H, W] -> [B*T*H*W, C].

    Returns (flat, meta) where meta is (b, c, t, h, w) for unflattening.
    """
    b, c, t, h, w = x.shape
    flat = x.permute(0, 2, 3, 4, 1).reshape(-1, c)
    return flat, (b, c, t, h, w)


def unflatten_spatio_temporal(flat: torch.Tensor, b: int, t: int, h: int, w: int):
    """Unflatten [B*T*H*W, C] -> [B, C, T, H, W].

    Note: caller must supply `b, t, h, w`. The channel dim is inferred.
    """
    return flat.view(b, t, h, w, -1).permute(0, 4, 1, 2, 3)
