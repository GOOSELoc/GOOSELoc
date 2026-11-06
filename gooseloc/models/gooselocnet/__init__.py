"""GOOSELocNet, our multimodal baseline (needs ``pip install gooseloc[model]``)."""

from gooseloc.models.gooselocnet.inference import (
    dataset_kwargs,
    load_checkpoint,
    predict_batch,
    save_checkpoint,
)
from gooseloc.models.gooselocnet.losses import total_loss
from gooseloc.models.gooselocnet.model import (
    VARIANTS,
    GOOSELocNet,
    GOOSELocNetConfig,
    heading_from_logits,
    patch_to_xy,
)

__all__ = [
    "GOOSELocNet", "GOOSELocNetConfig", "VARIANTS", "dataset_kwargs", "heading_from_logits",
    "load_checkpoint", "patch_to_xy", "predict_batch", "save_checkpoint", "total_loss",
]
