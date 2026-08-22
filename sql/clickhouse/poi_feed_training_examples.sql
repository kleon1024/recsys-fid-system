-- Point-in-time POI Feed examples. Table names are public simulator contracts.
WITH
    300 AS allowed_lateness_seconds,
    604800 AS attribution_window_seconds,
    decisions AS
    (
        SELECT
            request_id,
            viewer_id,
            author_id,
            video_id,
            poi_id,
            impression_time,
            feature_fids,
            dense_features,
            sequence_features,
            recall_route,
            sampling_probability,
            teacher_score,
            teacher_rank,
            exposed,
            pixel_observable,
            recall_score,
            coarse_score,
            fine_score,
            value_score,
            feature_version,
            model_version,
            index_version
        FROM recommendation_decision_log
        WHERE poi_id != 0
    ),
    actions AS
    (
        SELECT
            request_id,
            video_id,
            poi_id,
            action,
            min(event_time) AS event_time,
            min(received_at) AS received_at
        FROM viewer_action_log
        GROUP BY request_id, video_id, poi_id, action, event_id
    ),
    commerce AS
    (
        SELECT
            request_id,
            video_id,
            poi_id,
            max(action = 'submit') AS submit,
            max(action = 'order') AS placed_order,
            max(action = 'payment') AS payment
        FROM commerce_event_log
        GROUP BY request_id, video_id, poi_id
    )
SELECT
    d.request_id,
    d.viewer_id,
    d.author_id,
    d.video_id,
    d.poi_id,
    d.impression_time,
    d.feature_fids,
    d.dense_features,
    d.sequence_features,
    d.recall_route,
    d.sampling_probability,
    d.teacher_score,
    d.teacher_rank,
    d.exposed,
    maxIf(a.action = 'long_view', a.event_time <= d.impression_time + 300) AS long_view,
    maxIf(a.action = 'anchor_click', a.event_time <= d.impression_time + 600) AS anchor_click,
    maxIf(a.action = 'detail', a.event_time <= d.impression_time + 1800) AS detail,
    maxIf(a.action = 'favorite', a.event_time <= d.impression_time + 86400) AS favorite,
    any(c.submit) AS submit,
    any(c.placed_order) AS placed_order,
    any(c.payment) AS payment,
    now() >= d.impression_time + attribution_window_seconds + allowed_lateness_seconds AS commerce_label_mature,
    d.pixel_observable AS pixel_label_observable,
    [d.recall_score, d.coarse_score, d.fine_score, d.value_score] AS served_scores,
    [d.feature_version, d.model_version, d.index_version] AS version_manifest
FROM decisions AS d
LEFT JOIN actions AS a USING (request_id, video_id, poi_id)
LEFT JOIN commerce AS c USING (request_id, video_id, poi_id)
GROUP BY ALL;
