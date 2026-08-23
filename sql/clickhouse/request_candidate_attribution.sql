-- Request-level stage attribution. Unobserved labels remain masked, never zero-filled.
WITH request_stage AS
(
    SELECT
        request_id,
        countIf(is_corpus_oracle) AS oracle_recalled,
        countIf(is_corpus_oracle AND coarse_pass) AS oracle_after_coarse,
        countIf(is_corpus_oracle AND exposed_position = 1) AS oracle_exposed,
        maxIf(fine_rank, is_corpus_oracle) AS oracle_fine_rank,
        maxIf(mix_rank, is_corpus_oracle) AS oracle_mix_rank
    FROM recommendation_candidate_decision_log
    GROUP BY request_id
)
SELECT
    multiIf(
        oracle_recalled = 0, 'recall_miss',
        oracle_after_coarse = 0, 'coarse_miss',
        oracle_exposed = 1, 'served_oracle',
        oracle_fine_rank != oracle_mix_rank, 'mix_rank_miss',
        'fine_rank_miss'
    ) AS failure_stage,
    count() AS requests,
    requests / sum(requests) OVER () AS request_share
FROM request_stage
GROUP BY failure_stage
ORDER BY requests DESC;

-- Mature unified-LT outcomes by the stage that determined exposure opportunity.
SELECT
    d.filter_reason,
    count() AS candidates,
    countIf(l.label_masks['long_view']) AS mature_long_view_candidates,
    avgIf(l.labels['long_view'], l.label_masks['long_view']) AS long_view_rate,
    sum(if(mapContains(l.exchanged_lt_components, 'total'),
           l.exchanged_lt_components['total'], 0.0)) AS exchanged_lt
FROM recommendation_candidate_decision_log AS d
INNER JOIN recommendation_mature_label_log AS l
    USING (request_id, candidate_id, poi_id)
GROUP BY d.filter_reason
ORDER BY candidates DESC;
