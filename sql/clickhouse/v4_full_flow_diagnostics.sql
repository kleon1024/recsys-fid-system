-- Exact request reconstruction across every platform stage.
WITH route_items AS
(
    SELECT
        request_id,
        item_id,
        arrayStringConcat(arraySort(groupUniqArray(route_name)), ',') AS route_names,
        min(route_rank) AS best_route_rank,
        max(route_score) AS best_route_score
    FROM v4_route_candidate_log
    GROUP BY request_id, item_id
)
SELECT
    r.request_id,
    r.user_id,
    r.surface,
    r.event_time,
    r.experiment_cell,
    c.item_id,
    routes.route_names,
    c.recall_rank,
    c.recall_score,
    c.coarse_pass,
    c.coarse_rank,
    c.coarse_score,
    c.fine_pass,
    c.fine_rank,
    c.fine_score,
    c.exposed,
    c.exposed_position,
    c.drop_stage
FROM v4_request_log AS r
INNER JOIN v4_candidate_decision_log AS c USING (request_id)
LEFT JOIN route_items AS routes USING (request_id, item_id)
WHERE r.request_id = {request_id:UInt64}
ORDER BY c.recall_rank;

-- Route volume and unique item coverage. Pairwise overlap is computed from the
-- same request/item grain instead of inferred from merged route bits.
SELECT
    route_name,
    count() AS candidates,
    uniqExact(request_id) AS requests,
    uniqExact(item_id) AS unique_items,
    avg(route_score) AS mean_route_score
FROM v4_route_candidate_log
GROUP BY route_name
ORDER BY candidates DESC;

SELECT
    left.route_name AS left_route,
    right.route_name AS right_route,
    uniqExact((left.request_id, left.item_id)) AS overlapping_candidates
FROM v4_route_candidate_log AS left
INNER JOIN v4_route_candidate_log AS right
    ON left.request_id = right.request_id
   AND left.item_id = right.item_id
   AND left.route_name < right.route_name
GROUP BY left_route, right_route
ORDER BY overlapping_candidates DESC;

-- Marginal unique coverage by route, at request/item grain.
WITH overlap AS
(
    SELECT
        request_id,
        item_id,
        uniqExact(route_name) AS route_count
    FROM v4_route_candidate_log
    GROUP BY request_id, item_id
)
SELECT
    route.route_name,
    count() AS candidates,
    countIf(overlap.route_count = 1) AS marginal_unique_candidates
FROM v4_route_candidate_log AS route
INNER JOIN overlap USING (request_id, item_id)
GROUP BY route.route_name
ORDER BY marginal_unique_candidates DESC;

-- Stage pass/drop distribution.
SELECT
    drop_stage,
    count() AS candidates,
    uniqExact(request_id) AS requests,
    candidates / sum(candidates) OVER () AS candidate_share
FROM v4_candidate_decision_log
GROUP BY drop_stage
ORDER BY candidates DESC;

-- Observable country/region/content/age slices. Content lifecycle is added by
-- P1; age is retained in ticks here rather than guessed as a wall-clock unit.
SELECT
    request.user_country,
    request.user_region,
    candidate.content_kind,
    candidate.country AS item_country,
    candidate.region AS item_region,
    intDiv(candidate.content_age, 96) AS content_age_days,
    candidate.drop_stage,
    count() AS candidates
FROM v4_candidate_decision_log AS candidate
INNER JOIN v4_request_log AS request USING (request_id)
GROUP BY ALL
ORDER BY candidates DESC;

-- Mature, censored and inapplicable labels remain distinct.
SELECT
    task_name,
    censor_reason,
    count() AS labels,
    countIf(label_mask) AS observed_labels,
    avgIf(label_value, label_mask) AS observed_rate
FROM v4_mature_label_log
GROUP BY task_name, censor_reason
ORDER BY task_name, censor_reason;

-- Recall positives come from the recall example authority. A missing candidate
-- is diagnosable only when that positive has an identified external/randomized
-- source; ordinary unexposed Feed items are never labeled by hidden truth.
SELECT example.request_id, example.item_id
FROM v4_training_example_log AS example
LEFT JOIN v4_candidate_decision_log AS candidate USING (request_id, item_id)
WHERE example.authority = 'recall'
  AND example.role = 'positive'
  AND candidate.item_id IS NULL;

-- Events whose request closure was lost.
SELECT event.*
FROM v4_event_log AS event
LEFT JOIN v4_request_log AS request USING (request_id)
WHERE event.request_id >= 0
  AND event.user_id >= 0
  AND request.request_id IS NULL;

-- Checkpoint age and failed/fallback snapshots.
SELECT
    lane,
    model_name,
    checkpoint_version,
    max(data_watermark) AS latest_data_watermark,
    max(created_time - v4_checkpoint_log.data_watermark) AS checkpoint_age,
    any(validation_status) AS validation_status,
    any(publish_state) AS publish_state,
    any(fallback_version) AS fallback_version
FROM v4_checkpoint_log
GROUP BY lane, model_name, checkpoint_version
ORDER BY checkpoint_age DESC;
