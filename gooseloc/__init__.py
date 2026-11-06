"""GOOSELoc: ground-to-aerial localization in unstructured off-road terrain.

Quick start::

    from gooseloc import GOOSELoc, evaluate

    ds = GOOSELoc("data/GOOSELoc", split="test", setting="imu_prior")
    sample = ds[0]
    ...
    result = evaluate("my_predictions.csv", split="test")
"""

from gooseloc.data.dataset import GOOSELoc
from gooseloc.data.metadata import load_scenes, select
from gooseloc.eval.benchmark import evaluate

__version__ = "1.0.0"
__all__ = ["GOOSELoc", "evaluate", "load_scenes", "select", "__version__"]
