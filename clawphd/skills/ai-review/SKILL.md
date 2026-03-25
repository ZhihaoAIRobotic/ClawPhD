---
name: ai-review
description: "AI paper reviewer. Use when the user says 'review my paper', 'help me review this paper', '审稿', 'give me feedback on my paper', 'check my manuscript', 'evaluate this paper for NeurIPS/ICLR/EuroSys'. Accepts PDF files and produces structured narrative reviews with venue-specific dimensional scores and Accept/Reject recommendation."
metadata: {"clawphd":{"emoji":"📝"}}
---

## When to use

- "Review my paper at /path/to/paper.pdf"
- "Help me prepare my NeurIPS submission — review this draft"
- "给我审一下这篇论文"
- "Check if this paper is ready for ICLR submission"
- "Give me peer-review style feedback on this manuscript"
- "Evaluate this paper for EuroSys, tell me the scores"

## Tool

`paper_review(pdf_path, venue="NeurIPS", mode="sot", num_reflections=1)`

## Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| pdf_path | string | yes | Absolute path to the paper PDF |
| venue | string | no | Target conference: NeurIPS / ICLR / ICML / EuroSys / OSDI / SOSP / CVPR / ICCV / General (default: General) |
| mode | string | no | `sot` = Strength-of-Thought multi-pass (default, highest quality); `pure` = single-pass (faster); `vlm` = send page images to vision model (evaluates figures/typesetting) |
| num_reflections | integer | no | Self-reflection iterations (default: 1) |

## Return value

```json
{
  "status": "ok",
  "output_path": "/home/user/.clawphd/workspace/outputs/paper_review/my_paper/review.md",
  "venue": "NeurIPS",
  "decision": "Accept",
  "overall_score": 7,
  "scores": {
    "Originality": 3,
    "Quality": 3,
    "Clarity": 4,
    "Significance": 3,
    "Soundness": 3,
    "Presentation": 4,
    "Contribution": 3,
    "overall": 7,
    "confidence": 4,
    "decision": "Accept",
    "score_rationale": "Solid contribution with strong experiments and clear presentation, minor novelty concerns."
  },
  "warnings": []
}
```

## Scoring rubric

Scores follow real conference review forms:

| Venue | Dimensions | Overall scale |
|-------|-----------|---------------|
| NeurIPS / ICLR / ICML | Originality, Quality, Clarity, Significance, Soundness, Presentation, Contribution (each 1–4) | 1–10 |
| EuroSys / OSDI / SOSP | Novelty, Systems_Contribution, Evaluation_Rigor, Clarity (each 1–4) | 1–10 |
| CVPR / ICCV | Technical_Novelty, Soundness, Experiments, Presentation, Impact (each 1–4) | 1–6 |

Rubric source: [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) (12.5k stars) — NeurIPS official review form.

## Example conversation

User: 帮我把这篇论文按照 NeurIPS 标准审一下：/home/user/papers/my_draft.pdf

Agent: [calls paper_review with pdf_path="/home/user/papers/my_draft.pdf", venue="NeurIPS", mode="sot"]
→ Review saved to ~/.clawphd/workspace/outputs/paper_review/my_draft/review.md
→ Decision: Accept | Overall: 7/10
→ Originality: 3/4 | Soundness: 3/4 | Contribution: 3/4

## Output layout

```
outputs/paper_review/<paper_stem>/
├── review.md    # narrative review (6 sections) + score table
└── meta.json    # venue, mode, scores, elapsed time
```

## review.md structure

```
# Paper Review: <title>
> Venue: NeurIPS | Mode: sot | Generated: 2026-03-22

## Synopsis
## Summary of Review
## Strengths
## Weaknesses
## Suggestions for Improvement
## References

---

## Scores (NeurIPS)
| Dimension | Score |
...
**Decision: Accept**
```
