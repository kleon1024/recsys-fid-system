"""Small, inspectable recommendation lab built around packed feature IDs."""

from .fid import FidCodec, FidVersion
from .schema import DEFAULT_SCHEMA, FeatureRegistry

__all__ = ["DEFAULT_SCHEMA", "FeatureRegistry", "FidCodec", "FidVersion"]
