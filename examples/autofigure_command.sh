#!/bin/bash
# AutoFigure example: convert a raster figure image to an editable SVG
#
# Prerequisites:
#   1. Set FAL API key in ~/.clawphd/config.json:
#        { "tools": { "autofigure": { "falApiKey": "your-key" } } }
#      or export FAL_KEY="your-key"
#   2. pip install torch torchvision transformers pillow httpx lxml cairosvg
#   3. A VLM provider must be configured (openrouter / bianxie / gemini)
#
# Usage:
#   bash examples/autofigure_command.sh
#   OR with a custom image:
#   IMAGE_PATH=/path/to/figure.png bash examples/autofigure_command.sh

IMAGE_PATH="/home/ubuntu/research/ClawPhD/examples/test_gs.jpg"

clawphd agent -m "我换了新模型，再次尝试将下面这张论文方法图转换为可编辑的 drawio 文件。

图片路径：${IMAGE_PATH}
输出目录：./output/autofigure5

完成后告诉我最终 drawio 的路径。如果有报错没成功也告诉我，不要跳过 generate_drawio_template 步骤。
"
