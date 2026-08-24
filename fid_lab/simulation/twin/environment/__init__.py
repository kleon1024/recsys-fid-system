"""Hidden environment state and response mechanisms; never a platform input."""

from .latent import LatentCatalogState, LatentUserState
from .runtime import UserEnvironment

__all__ = ("LatentCatalogState", "LatentUserState", "UserEnvironment")
