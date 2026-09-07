# Tailwyndz Propel Lateral Drive 2026 — Assessment 4

## Forty Minutes Into The Set

This project analyzes a synthetic social-media dataset from a live event and builds a lightweight command-centre monitoring pipeline.

The pipeline addresses:

1. Data-quality auditing
2. Kestrel/event relevance filtering
3. Automated-traffic detection
4. Multilingual sentiment scoring
5. Verified-Sentiment Shift (VSS) calculation
6. Sustained negative-signal detection
7. Evaluation against planted incident ground truth

---

## Project Structure

```text
Tailwyndz Assignment/
├── data/
│   ├── final/
│   ├── ground_truth/
│   ├── processed/
│   └── raw/
│
├── docs/
│   └── memo/
│       └── memo.pdf
│
├── outputs/
│   ├── bot_evaluation.json
│   ├── data_quality_audit.json
│   ├── incident_evaluation.csv
│   ├── relevance_summary.json
│   ├── scored_sample.csv
│   ├── sentiment_summary.json
│   └── vss_15min.csv
│
├── presentation/
│   └── Forty-Minutes-Into-The-Set.pptx
│
├── src/
│   ├── analyze_data.py
│   ├── content_bot_test.py
│   ├── coordination_test.py
│   ├── final_generate.py
│   ├── generate_data.py
│   └── signal_test.py
│
├── tests/
│   └── test_vss.py
│
├── .gitignore
├── approach_tried.md
├── assumptions.md
├── README.md
└── requirements.txt
```

> **Note:** `data/ground_truth/` contains private evaluation data and is excluded from version control. It is used locally for evaluation only.

---

## Environment

Python 3 is required.

Install the pinned dependencies with:

```bash
pip install -r requirements.txt
```

---

## AI Usage

AI tools were used as development assistance during this assessment.

- **ChatGPT** — used for code debugging, identifying implementation issues, suggesting test cases, reviewing analysis logic, and helping structure documentation and presentation/memo content.
- AI assistance was used as a support tool rather than as a substitute for the analysis. Final implementation decisions, thresholds, evaluation methodology, results, assumptions, and interpretations were reviewed against the assessment requirements and the generated dataset.
- No paid AI APIs or proprietary AI services were used in the analysis pipeline.
