-- Funnel closure, cascade opportunity, and offline-online replay diagnostics.
SELECT
    toDate(impression_time) AS event_date,
    count() AS main_candidates,
    countIf(poi_id != 0) AS poi_candidates,
    countIf(poi_id != 0 AND eligible) AS eligible_candidates,
    countIf(poi_id != 0 AND coarse_rank <= 600) AS coarse_pass,
    countIf(poi_id != 0 AND fine_rank <= 200) AS fine_pass,
    countIf(poi_id != 0 AND exposed) AS exposures,
    countIf(exposed AND long_view) AS long_views,
    countIf(exposed AND anchor_click) AS anchor_clicks,
    countIf(exposed AND payment) AS payments,
    countIf(exposed AND pixel_conversion > 0) AS attributed_conversions,
    coarse_pass / greatest(poi_candidates, 1) AS coarse_pass_rate,
    fine_pass / greatest(coarse_pass, 1) AS fine_pass_rate,
    countIf(teacher_rank <= 200 AND coarse_rank <= 600)
        / greatest(countIf(teacher_rank <= 200), 1) AS teacher_topk_preservation
FROM recommendation_funnel_daily
GROUP BY event_date
ORDER BY event_date;

SELECT
    feature_version,
    model_version,
    index_version,
    count() AS replayed,
    quantileExact(0.50)(abs(offline_score - online_score)) AS score_delta_p50,
    quantileExact(0.99)(abs(offline_score - online_score)) AS score_delta_p99,
    max(abs(offline_score - online_score)) AS score_delta_max,
    countIf(offline_fids != online_fids) AS fid_mismatches,
    countIf(offline_candidates != online_candidates) AS candidate_mismatches
FROM feature_replay_log
GROUP BY feature_version, model_version, index_version;
