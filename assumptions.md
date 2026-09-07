# Assumptions and Limitations

## Data and timestamps

- Timestamps are parsed into UTC before analysis.
- The event window is defined as 2026-04-18 10:00 UTC to 2026-04-19 00:00 UTC.
- The sentiment baseline uses the 14-day period immediately preceding the event.
- Rows with timestamps that cannot be parsed are excluded from time-based analysis. In the final dataset, all timestamps were successfully parsed.
- Late-arriving and out-of-order records are retained rather than silently discarded.

## Relevance

- Kestrel is treated as relevant only when the post contains evidence that it refers to the event/brand rather than an unrelated use of the word.
- Ambiguous Kestrel mentions are not treated as confirmed relevant posts for VSS.
- The relevance classifier is rule-based and therefore can make errors, particularly for short or context-dependent posts.

## Automated traffic

- Automated traffic is detected at the post level in the final evaluation.
- The final detector uses transparent content signals for promotional/spam language, negative amplification language, and automated monitoring/summary language.
- Author/profile metadata and coordination signals are treated as supporting evidence or experimental approaches rather than standalone production rules.
- Posts with missing text are harder to classify using content-based signals and represent an important limitation.
- The detector does not claim to identify every automated account perfectly.

## Sentiment

- Sentiment is scored as positive, neutral, or negative using a transparent rule-based approach.
- The approach includes multilingual vocabulary, selected Hindi/Devanagari and Hinglish terms, Portuguese/German terms, emojis, negation, and selected sarcasm patterns.
- Sarcasm, mixed-language expressions, very short posts, and context-dependent language remain difficult cases.
- Hindi negation is a known weak area. For example, some constructions where "not" follows the sentiment word can be misclassified.
- Sentiment is used for aggregate event monitoring rather than as a claim of perfect individual-post classification.

## VSS

- VSS is calculated as current 15-minute net sentiment minus the same-market 14-day baseline net sentiment.
- Only posts classified as human, relevant, and assigned to a known market are used.
- A window with fewer than 220 qualifying posts produces `NO SIGNAL`.
- A VSS breach occurs at VSS <= -14.
- Escalation requires three consecutive 15-minute breach windows for the same market.
- Consecutive windows must be exactly 15 minutes apart; missing windows do not count toward the three-window requirement.
- Raw post volume is not used as the sentiment signal itself.

## Incident evaluation

- Incident evaluation uses the separately stored ground truth after predictions have been generated.
- Ground truth is not used as an input to the production detection rules.
- The evaluation covers the planted queue-crush, sound-failure, sponsor-announcement, payment-outage, and positive-decoy scenarios.

## Known limitations

- The final dataset is synthetic and therefore performance on it should not be interpreted as production performance on real social-media data.
- The relevance classifier does not have a complete independent manually labelled test set, so relevance precision/recall is not claimed.
- Missing location/geo information prevents some posts from contributing to market-level VSS.
- Coordination detection was tested but rejected as a standalone detector because it produced very low precision.
- The final system is designed as a decision-support signal for a command centre, not as an autonomous incident-response system.