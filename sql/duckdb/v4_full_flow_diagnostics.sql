CREATE OR REPLACE TEMP VIEW v4_route_items AS
SELECT
    request_id,
    item_id,
    string_agg(DISTINCT route_name, ',' ORDER BY route_name) AS route_names,
    min(route_rank) AS best_route_rank,
    max(route_score) AS best_route_score
FROM v4_route_candidate_log
GROUP BY request_id, item_id;

CREATE OR REPLACE TEMP VIEW v4_request_case AS
SELECT
    r.request_id,
    r.user_id,
    r.surface,
    r.event_time,
    r.experiment_cell,
    r.recall_version_id,
    r.coarse_version_id,
    r.fine_version_id,
    r.mix_version_id,
    c.item_id,
    c.recall_rank,
    c.recall_score,
    routes.route_names,
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
JOIN v4_candidate_decision_log AS c USING (request_id)
LEFT JOIN v4_route_items AS routes USING (request_id, item_id);

CREATE OR REPLACE TEMP VIEW v4_stage_distribution AS
SELECT
    drop_stage,
    count(*) AS candidates,
    count(DISTINCT request_id) AS requests,
    candidates / sum(candidates) OVER () AS candidate_share
FROM v4_candidate_decision_log
GROUP BY drop_stage;

CREATE OR REPLACE TEMP VIEW v4_route_distribution AS
SELECT
    route_name,
    count(*) AS candidates,
    count(DISTINCT request_id) AS requests,
    count(DISTINCT item_id) AS unique_items,
    avg(route_score) AS mean_route_score
FROM v4_route_candidate_log
GROUP BY route_name;

CREATE OR REPLACE TEMP VIEW v4_route_marginal_coverage AS
WITH overlap AS
(
    SELECT
        request_id,
        item_id,
        count(DISTINCT route_name) AS route_count
    FROM v4_route_candidate_log
    GROUP BY request_id, item_id
)
SELECT
    route.route_name,
    count(*) AS candidates,
    count(*) FILTER (WHERE overlap.route_count = 1) AS marginal_unique_candidates
FROM v4_route_candidate_log AS route
JOIN overlap USING (request_id, item_id)
GROUP BY route.route_name;

CREATE OR REPLACE TEMP VIEW v4_candidate_slices AS
SELECT
    request.user_country,
    request.user_region,
    candidate.content_kind,
    candidate.country AS item_country,
    candidate.region AS item_region,
    floor(candidate.content_age / 96) AS content_age_days,
    candidate.drop_stage,
    count(*) AS candidates
FROM v4_candidate_decision_log AS candidate
JOIN v4_request_log AS request USING (request_id)
GROUP BY ALL;

CREATE OR REPLACE TEMP VIEW v4_label_maturity AS
SELECT
    task_name,
    censor_reason,
    count(*) AS labels,
    count(*) FILTER (WHERE label_mask) AS observed_labels,
    avg(label_value) FILTER (WHERE label_mask) AS observed_rate
FROM v4_mature_label_log
GROUP BY task_name, censor_reason;

CREATE OR REPLACE TEMP VIEW v4_recall_miss AS
SELECT example.request_id, example.item_id
FROM v4_training_example_log AS example
LEFT JOIN v4_candidate_decision_log AS candidate
    USING (request_id, item_id)
WHERE example.authority = 'recall'
  AND example.role = 'positive'
  AND candidate.item_id IS NULL;

CREATE OR REPLACE TEMP VIEW v4_orphan_events AS
SELECT event.*
FROM v4_event_log AS event
LEFT JOIN v4_request_log AS request USING (request_id)
WHERE event.request_id >= 0 AND request.request_id IS NULL;

CREATE OR REPLACE TEMP VIEW v4_checkpoint_health AS
SELECT
    *,
    validation_status != 'pass'
        OR publish_state NOT IN ('active', 'candidate', 'shadow')
        OR fallback_version != '' AS unhealthy
FROM v4_checkpoint_log;
