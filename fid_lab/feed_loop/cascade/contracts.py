"""Single vocabulary authority for every Feed cascade runtime."""

RECALL_ROUTES = (
    "ann",
    "graph",
    "geo",
    "fresh",
    "long_tail",
    "popular",
    "post_search",
    "retarget",
)
BASE_RECALL_ROUTES = RECALL_ROUTES[:6]
COARSE_MODELS = ("quality_only", "lr_v1", "dcnv2_distilled")


def validate_routes(routes):
    unknown = set(routes) - set(RECALL_ROUTES)
    if unknown:
        raise ValueError(f"unsupported recall routes: {sorted(unknown)}")


def validate_coarse_model(model):
    if model not in COARSE_MODELS:
        raise ValueError(f"unsupported coarse model: {model}")
