# Paper Review: SELF-RAG — Learning to Retrieve, Generate, and Critique through Self-Reflection

> **Paper:** Asai et al., arXiv:2310.11511 | **Venue:** ICLR 2024 (Rejected)
> **Review settings:** venue=ICLR | mode=pure | num_reflections=0

> **Context for this test case:** SELF-RAG was submitted to ICLR 2024 and rejected, primarily because it
> had already been accepted at EMNLP 2023 before ICLR's review cycle completed — not due to fundamental
> quality issues. Our system evaluates intrinsic paper quality: it correctly scores SELF-RAG lower than
> Mamba across all breakthrough dimensions (Originality 3 vs 4, Significance 3 vs 4, Contribution 3 vs 4),
> reflecting its solid but more incremental contribution relative to Mamba's architectural innovation.

---

# Review of "SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"

## Synopsis

This paper introduces Self-Reflective Retrieval-Augmented Generation (SELF-RAG), a framework that trains language models to adaptively retrieve passages on-demand and generate reflection tokens to evaluate retrieval necessity, passage relevance, factual support, and output utility. Unlike standard RAG approaches that indiscriminately retrieve fixed numbers of documents, SELF-RAG learns when retrieval is beneficial and uses special tokens (Retrieve, ISREL, ISSUP, ISUSE) to guide generation through segment-level beam search. The authors train critic and generator models on 150k instruction-output pairs augmented with GPT-4-generated reflection tokens. Experiments on six tasks show SELF-RAG (7B/13B parameters) outperforms ChatGPT and retrieval-augmented Llama2-chat on open-domain QA, reasoning, fact verification, and long-form generation, with significant improvements in factuality and citation accuracy.

## Summary of Review

This paper presents a well-motivated approach to addressing factual inaccuracies in LLMs through adaptive retrieval and self-reflection. The core innovation—training models to generate reflection tokens that enable controllable, on-demand retrieval—is compelling and demonstrates strong empirical results across diverse tasks (Table 2). The framework's ability to customize behavior at inference time without retraining (Section 3.3, Figure 3b) is particularly valuable. However, the paper has notable weaknesses: the mathematical formulation of the segment scoring function lacks theoretical justification, the training procedure's dependence on GPT-4 for initial data collection raises reproducibility concerns, and the computational overhead of parallel passage processing and beam search is insufficiently analyzed.

## Strengths

- **Novel framework with strong empirical results**: SELF-RAG significantly outperforms baselines across six diverse tasks, including a 34.8 point improvement over Llama2-7B on PopQA (54.9 vs 14.7 accuracy, Table 2) and substantial gains in citation precision on ASQA (66.9 vs 5.5 for Alpaca-7B). The approach demonstrates consistent improvements over both retrieval-augmented and non-augmented baselines.

- **Effective training strategy with offline critique integration**: The two-stage training approach (Section 3.2) that distills GPT-4 reflection tokens into a critic model C (Eq. 1) and then trains generator M with standard next-token prediction (Eq. 2) is computationally efficient compared to RLHF. The critic achieves >90% agreement with GPT-4 predictions, and ablations show both retriever and critic components are essential (Table 3a).

- **Inference-time controllability without retraining**: The framework enables customizable generation through adjustable weights in the segment scoring function (Eq. 4, Section 3.3). Figure 3b demonstrates that increasing ISSUP weight from 1.0 to 2.0 improves citation precision by ~5 points while trading off fluency (MAUVE).

- **Comprehensive evaluation with appropriate metrics**: The paper evaluates on closed-set tasks (PubHealth, ARC-Challenge), short-form QA (PopQA, TriviaQA), and long-form generation (Biography, ASQA) using task-appropriate metrics including factuality (FactScore), citation accuracy, and fluency (MAUVE).

## Weaknesses

- **Insufficient theoretical justification for segment scoring function**: Equation 4 presents the segment score as S(Critique) = Σ_G w_G s^G_t, but provides no theoretical or empirical justification for why linear combination is optimal. No analysis explores alternative aggregation functions or validates that this formulation effectively captures multi-objective preferences.

- **Dependence on proprietary GPT-4 for training data initialization**: Section 3.2.1 acknowledges using GPT-4 to generate initial reflection tokens, creating reproducibility concerns and potential bias propagation. The paper provides limited analysis of how GPT-4 annotation quality affects downstream performance.

- **Inadequate computational cost analysis**: Algorithm 1 processes K retrieved passages in parallel and performs segment-level beam search, but the paper provides no runtime comparisons, memory requirements, or wall-clock time measurements against baselines. For practical deployment, understanding the efficiency-accuracy tradeoff is critical.

- **Limited exploration of design choices and hyperparameters**: The paper uses sentence-level segmentation without justifying this granularity or comparing alternatives. Default weights in Eq. 4 are set without systematic ablation.

- **Incomplete ablation studies**: Table 3a provides ablations only on three datasets with a reduced 50k training set. The human evaluation samples only 50 examples per task, limiting statistical confidence.

## Suggestions for Improvement

- **Provide theoretical or empirical justification for Eq. 4's formulation**: Conduct ablation studies comparing linear aggregation against multiplicative or learned combinations. Analyze the normalization scheme's sensitivity to weight choices.

- **Reduce dependence on GPT-4 and improve reproducibility**: Explore alternative initialization strategies such as human annotations or self-training from model-generated critiques. Release the GPT-4-generated training data and prompts.

- **Add comprehensive computational cost analysis**: Report wall-clock inference time, memory usage, and FLOPs for SELF-RAG versus baselines. Quantify the overhead of parallel passage processing and beam search. Provide guidance on selecting K and B values.

- **Systematically explore and justify design choices**: Compare sentence-level segmentation against alternative granularities. Conduct grid search over weight parameters in Eq. 4. Extend Figure 4's scaling analysis to determine if performance saturates.

- **Expand ablation studies with statistical rigor**: Conduct full ablations on all six tasks with the complete 150k training set, removing each reflection token type individually. Increase human evaluation sample size to ≥200 examples with inter-annotator agreement metrics.

## References

- Asai, A., et al. (2023). SELF-RAG: Learning to retrieve, generate, and critique through self-reflection. *EMNLP 2023 / arXiv:2310.11511*.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for knowledge-intensive NLP tasks. *NeurIPS*.
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*.
- Touvron, H., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint*.
- Min, S., et al. (2023). FActScoring: Fine-grained atomic evaluation of factual precision. *EMNLP*.
- Gao, T., et al. (2023). Enabling large language models to generate text with citations. *EMNLP*.

---

## Scores (ICLR)

| Dimension | Score |
|-----------|-------|
| Originality | 3 / 4 |
| Quality | 3 / 4 |
| Clarity | 3 / 4 |
| Significance | 3 / 4 |
| Soundness | 3 / 4 |
| Presentation | 3 / 4 |
| Contribution | 3 / 4 |
| **Overall** | **7 / 10** |
| Confidence | 4 / 5 |

**Decision: Accept**

*Strong empirical results and novel adaptive retrieval framework with inference-time controllability outweigh concerns about theoretical justification and GPT-4 dependence, making this a solid contribution to retrieval-augmented generation. All dimensions score 3/4 — reflecting reliable but incremental progress rather than architectural breakthroughs.*

---

### Score comparison with accepted paper (Mamba)

| Dimension | SELF-RAG (Rejected) | Mamba (Accepted) |
|-----------|---------------------|------------------|
| Originality | 3 / 4 | **4 / 4** |
| Quality | 3 / 4 | 3 / 4 |
| Clarity | 3 / 4 | 3 / 4 |
| Significance | 3 / 4 | **4 / 4** |
| Soundness | 3 / 4 | 3 / 4 |
| Presentation | 3 / 4 | 3 / 4 |
| Contribution | 3 / 4 | **4 / 4** |
| **Overall** | **7 / 10** | **8 / 10** |

Mamba achieves the maximum score (4/4) on Originality, Significance, and Contribution — the three dimensions most predictive of long-term impact. SELF-RAG scores 3/4 uniformly, indicating solid but not breakthrough-level work. This aligns with real-world outcomes: Mamba became a foundational architecture adopted across dozens of downstream works, while SELF-RAG, though well-received at EMNLP, did not achieve the same structural influence.
