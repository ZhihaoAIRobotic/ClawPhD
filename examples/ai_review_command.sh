#!/bin/bash
# Prerequisites: clawphd installed, VLM provider (OpenRouter) configured in ~/.clawphd/config.json

clawphd agent -m "请帮我用 NeurIPS 的审稿标准审阅这篇论文，使用 SoT 模式进行深度分析：$(pwd)/paper.pdf"
