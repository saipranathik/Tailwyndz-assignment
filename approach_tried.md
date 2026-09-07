# Approaches Tried and Dropped

## 1. High-activity and repetition rules

### Approach
We first tested whether automated accounts could be identified using simple activity and repetition thresholds, such as high posting frequency, repeated-text rate, and combinations of post volume with repetition.

### Result
These rules produced very large numbers of false positives.

For example, the rule:

- posts >= 10 AND repeated-text rate >= 20%

produced:

- Predicted automated posts: 556,558
- Precision: 6.4%
- Recall: 40.7%
- Error rate: 47.5%

A posting-frequency threshold of >= 5 produced a precision of only 7.3%.

### Decision
Dropped as a standalone detector. High activity and repeated content are common during a live event and therefore cannot reliably distinguish automated traffic from genuine high-volume human activity.

---

## 2. Author/profile metadata rules

### Approach
We tested account-level signals including posting frequency, repeated-content rate, profile completeness, and other available author metadata.

### Result
The signals provided useful supporting clues but did not provide sufficient separation on their own.

Examples from the experiments:

| Signal | Predicted | Precision | Recall | Error |
|---|---:|---:|---:|---:|
| posting_frequency >= 5 | 97,414 | 7.3% | 8.1% | 14.2% |
| posting_frequency >= 10 | 26,273 | 7.3% | 2.2% | 9.1% |
| no picture | 149,270 | 7.2% | 12.2% | 17.9% |
| no bio | 253,892 | 7.2% | 20.9% | 25.3% |

We also tested posting-interval regularity. The distributions overlapped substantially between human and automated accounts, so interval regularity was not used as a standalone decision rule.

### Decision
Dropped as the primary detector. Profile and activity metadata are useful as supporting evidence but are too weak to classify automated traffic reliably by themselves.

---

## 3. Coordinated repeated-text detection

### Approach
We tested whether repeated or near-identical text posted by multiple authors within a short time window could identify coordinated automated activity.

We evaluated windows from 5 to 60 seconds and required repeated text to appear across multiple authors.

### Result
The detector showed a strong precision/recall trade-off.

| Rule | Precision | Recall | Error |
|---|---:|---:|---:|
| 5s + 3 authors | 3.3% | 31.9% | 73.1% |
| 10s + 3 authors | 5.4% | 61.4% | 81.3% |
| 30s + 3 authors | 7.8% | 95.7% | 82.4% |
| 60s + 3 authors | 8.1% | 99.7% | 82.8% |

At wider windows, recall became very high, but the detector flagged enormous numbers of legitimate posts as coordinated activity.

### Decision
Dropped as a standalone detector. Common phrases, slogans, event reactions, and repeated legitimate content make short-window text coordination too noisy for direct classification.

---

## Final automated-traffic approach

The final detector uses transparent content-based signals rather than any single activity, profile, or coordination rule.

It combines signals for:

- promotional/spam-like language,
- negative amplification language,
- automated monitoring/summary language.

On the frozen final dataset, the final detector achieved:

- True automated posts: 87,868
- Predicted automated posts: 85,344
- Precision: 97.0%
- Recall: 94.2%
- Error rate: 0.632%

The final approach was selected because it provided substantially better precision while retaining high recall, without relying on a single brittle account-level or coordination heuristic.