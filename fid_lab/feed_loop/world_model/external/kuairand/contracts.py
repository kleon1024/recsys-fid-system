"""Single authority for external KuaiRand sequence semantics."""

from __future__ import annotations


SOURCE_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "user_features_pure.csv",
    "video_features_basic_pure.csv",
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
