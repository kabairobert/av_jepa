import torch

from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml
from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig


def init_data(env_name, cfg_data=None, **kwargs):
    """Initialize data loaders for the specified environment.

    Args:
        env_name: Name of the environment (currently only "two_rooms" is supported).
        cfg_data: Configuration dictionary for the dataset.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Tuple of (train_loader, val_loader, config).
    """
    if env_name != "two_rooms":
        raise ValueError(f"Unknown env: {env_name}. Only 'two_rooms' is supported.")

    config = update_config_from_yaml(WallDatasetConfig, cfg_data)
    dset = WallDataset(config=config)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=cfg_data.get("num_workers"),
        pin_memory=cfg_data.get("pin_mem"),
        drop_last=True,
        persistent_workers=cfg_data.get("persistent_workers")
        and cfg_data.get("num_workers") > 0,
    )
    val_dset = WallDataset(config=config)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=4,
        shuffle=False,
        num_workers=cfg_data.get("num_workers"),
        pin_memory=cfg_data.get("pin_mem"),
        drop_last=True,
        persistent_workers=cfg_data.get("persistent_workers")
        and cfg_data.get("num_workers") > 0,
    )
    return loader, val_loader, config
