from .base import MappingStrategy
from .mode_a import AutoSyllableStrategy
from .mode_b import SingleSampleStrategy

STRATEGIES = {
    "a": AutoSyllableStrategy,
    "b": SingleSampleStrategy,
}

__all__ = ["MappingStrategy", "AutoSyllableStrategy", "SingleSampleStrategy", "STRATEGIES"]
