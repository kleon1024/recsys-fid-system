-- Seven-day multi-touch attribution with a 24-hour exponential half-life.
WITH
    604800 AS attribution_window_seconds,
    86400.0 AS half_life_seconds,
    deduplicated_pixels AS
    (
        SELECT
            event_id,
            argMax(conversion_id, received_at) AS conversion_id,
            argMax(identity, received_at) AS identity,
            argMax(merchant_id, received_at) AS merchant_id,
            argMax(event_time, received_at) AS conversion_time,
            argMax(click_id, received_at) AS exact_click_id
        FROM pixel_event_log
        GROUP BY event_id
    ),
    eligible AS
    (
        SELECT
            p.event_id,
            p.conversion_id,
            c.click_id,
            c.request_id,
            c.video_id,
            c.poi_id,
            exp(-log(2) * dateDiff('second', c.event_time, p.conversion_time) / half_life_seconds) AS raw_weight
        FROM deduplicated_pixels AS p
        INNER JOIN outbound_click_log AS c
            ON (p.exact_click_id != '' AND c.click_id = p.exact_click_id)
            OR
            (
                p.exact_click_id = ''
                AND p.identity != ''
                AND c.identity = p.identity
                AND c.merchant_id = p.merchant_id
                AND c.event_time <= p.conversion_time
                AND c.event_time >= p.conversion_time - attribution_window_seconds
            )
    )
SELECT
    event_id,
    conversion_id,
    click_id,
    request_id,
    video_id,
    poi_id,
    raw_weight / sum(raw_weight) OVER (PARTITION BY event_id) AS fractional_label
FROM eligible;
