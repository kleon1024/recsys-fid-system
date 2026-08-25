# Commerce transaction and inventory Launch Reviews

This is synthetic digital-twin evidence, not company production evidence.

本文是合成数字孪生证据，不代表任何公司的生产指标。

## Workflow correction / 交易链修复

The old world generated orders directly from `DETAIL`, so its CVR sample space
was wrong. Commerce now requires:

```text
impression → click → detail → add_cart → order → payment → refund
```

Only product details can create a cart. `order_id` begins at cart time and is
preserved through payment and refund. Order scheduling uses point-in-time hidden
inventory and counts pending orders as reservations, so concurrent requests
cannot sell more than the available units. Local keeps its separate detail-to-
submit skeleton and is not relabeled as Commerce.

旧 world 从 `DETAIL` 直接生成订单，CVR 样本空间错误。现在 Commerce 必须经过购物车；
`order_id` 从购物车开始贯穿订单、支付和退款；未交付订单计入 reservation，避免并发
超卖。Local 仍保留独立链路，不借用 Commerce label。

## Infrastructure failures found / 全链路故障

1. Late payment and Pixel events were appended by ingest order to an event-time
   sequence. At tick 340 this finally violated chronological request history.
   The projection now merges retained and late events by event time, and exposes
   an explicit rebuild from delivered event-log partitions.
2. Failed launches wrote request partitions before the world checkpoint CAS.
   Thirteen orphan partitions were found and removed from the training manifest.
   Launches now write to attempt staging, atomically publish the partition set,
   and reconcile anything beyond the factual branch head after interruption.

第一，延迟支付和 Pixel 曾按到达顺序写入 event-time 序列，世界演进到 tick 340 后才
触发倒序校验。现在序列会合并 late event，并支持从已交付日志重建。第二，失败实验
曾先写 request partition、后提交 checkpoint，留下 13 个 orphan。现在每次 attempt
先写 staging，成功后整体发布；中断后按 factual head 自动清理。

## C-LR-001: zero-inventory filter

The policy filtered only products whose inventory was exactly zero. Across
ticks 340-372, control exposed no such product. The experiment ended
`NO_SUPPORT`; observed metric differences were not interpreted as treatment
effects.

库存等于零的过滤实验没有任何 control support，因此以 `NO_SUPPORT` 结束，不把小样本
波动解释成收益。

## C-LR-002: low-inventory route filter

The next policy excluded product inventory at or below 0.28 inside the
`commerce_intent` route. Control had ten low-inventory product exposures, but
treatment still had nine because `retarget` reintroduced them after the route-
local filter. The version was rejected for full-chain inconsistency; it was not
allowed to accumulate more traffic.

第二版在 `commerce_intent` 内过滤库存不高于 0.28 的商品，但 `retarget` 在后面把商品
重新送回。Control 有 10 次低库存曝光，Treatment 仍有 9 次，因此版本因全链路不一致
被 REJECT，而不是继续拉长实验。

## C-LR-003: global inventory eligibility

The corrected treatment applies inventory eligibility to every Commerce route
before reciprocal-rank fusion. Its launch is pending completion of the one-time
observable history rebuild. No result is claimed until the factual checkpoint,
request stream and report agree.

修正版在 RRF 合并前对所有 Commerce route 统一执行库存门禁。当前等待一次性的可观测
历史重建完成；在 factual checkpoint、request stream 和报告一致前不宣称实验结果。
