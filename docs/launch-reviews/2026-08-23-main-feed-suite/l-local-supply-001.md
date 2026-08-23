# L-LOCAL-SUPPLY-001 — Personalized POI posting supply

Status: `pass_to_cluster_switchback`, not a viewer-level rollout.

## Change

Rank posting-page POIs with author and history context. Published videos mutate
a copied supply catalog and enter ordinary Fresh and Local recall before Feed
distribution.

## Posting funnel

| Metric | Control | Treatment |
|---|---:|---:|
| Posting-page entry | 5.33% | 5.33% |
| POI selection | 1.67% | 2.67% |
| Submit | 0.67% | 2.00% |
| Publish | 0.67% | 1.67% |
| Published videos | 2 | 5 |
| Mean published quality | 0.470 | 0.726 |
| Local supply Value Tree | 3.289 | 12.703 |

Entry is unchanged. The synthetic increment comes from candidate ranking and
downstream funnel quality, not additional posting-page traffic.

## Paired-world distribution effect

| Metric | Relative effect |
|---|---:|
| Feed stay | +0.98% |
| Long views | +1.75% |
| Quality long views | +2.22% |
| POI anchor clicks | +11.67% |
| Local Value Tree | +9.83% |
| Platform LT container | +0.61% |

## Experiment unit and decision

Publishing supply changes the catalog seen by other viewers, so viewer SUTVA is
violated and UID randomization is invalid. These paired worlds justify only the
next experiment: city-time switchback or author cluster with contamination and
platform-metric monitoring. Five treatment videos are not rollout evidence.

投稿供给会改变其他用户看到的候选集，因此不能用普通 UID A/B。当前结果只允许进入
城市时段 switchback 或作者 cluster 实验，不能直接放量。
