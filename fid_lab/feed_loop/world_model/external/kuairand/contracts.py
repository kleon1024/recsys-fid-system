"""Single authority for external KuaiRand sequence semantics."""

from __future__ import annotations


SOURCE_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "user_features_pure.csv",
    "video_features_basic_pure.csv",
)
RANDOMIZED_SOURCE_FILES = (
    "log_standard_4_08_to_4_21_1k.csv",
    "log_standard_4_22_to_5_08_1k.csv",
    "log_random_4_22_to_5_08_1k.csv",
    "user_features_1k.csv",
    "video_features_basic_1k.csv",
)
FEEDBACK_NAMES = (
    "is_click",
    "long_view",
    "is_like",
    "is_comment",
    "is_forward",
    "is_follow",
    "is_hate",
)
SPARSE_NAMES = (
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "video_type",
    "upload_type",
    "music_type",
)
DENSE_NAMES = (
    "duration_log_norm",
    "hour_sin",
    "hour_cos",
    "aspect_ratio",
    "account_age_log_norm",
    "follow_count_log_norm",
    "fan_count_log_norm",
    "friend_count_log_norm",
    "low_activity",
    "live_streamer",
    "video_author",
)
TRAIN_END_DATE = 20220418
VALIDATION_END_DATE = 20220421
DEFAULT_SEQUENCE_LENGTH = 64
RANDOM_ITEM_POOL_SIZE = 7_388
RANDOMIZED_SPLIT_RATES = {
    "train": 0.25,
    "validation": 0.20,
    "standard_test": 0.05,
}
HASH_VOCABULARIES = (
    1_002,
    262_145,
    262_145,
    8_193,
    4,
    33,
    1_025,
)
