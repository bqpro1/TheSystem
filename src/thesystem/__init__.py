"""TheSystem: one frozen decoder, decision and generation modes."""
from .inference import TheSystem, load_model

__all__ = ["TheSystem", "load_model"]
__version__ = "0.1.0rc4"
