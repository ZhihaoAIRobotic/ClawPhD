"""Example: Generate academic diagrams via clawphd's PaperBanana tools.

Prerequisites
-------------
Option A  (Google direct — VLM + image-gen):
    pip install google-genai Pillow
    export GOOGLE_API_KEY="your-key-here"

Option B  (Replicate for everything):
    pip install replicate Pillow
    export REPLICATE_API_TOKEN="r8_..."

Option C  (mix — Replicate image-gen, Google VLM):
    pip install replicate google-genai Pillow
    export REPLICATE_API_TOKEN="r8_..."
    export GOOGLE_API_KEY="your-key-here"

All options require a clawphd LLM key in ~/.clawphd/config.json.

Usage
-----
    python examples/paperbanana_diagram.py                                    # all Google
    python examples/paperbanana_diagram.py --backend replicate                # all Replicate
    python examples/paperbanana_diagram.py --vlm-backend replicate            # Replicate VLM only
    python examples/paperbanana_diagram.py --backend replicate --mode plot    # plot via Replicate VLM
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# ── Ensure the repo root is importable ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# 1.  Create VLM / Image-gen providers  (clawphd-native Gemini wrappers)
# ---------------------------------------------------------------------------

def _build_providers(backend: str = "gemini", vlm_backend: str | None = None):
    """Return (vlm_provider, image_gen_provider, reference_store).

    Args:
        backend: ``"gemini"`` (default) or ``"replicate"`` for *both* VLM
            and image generation unless ``vlm_backend`` overrides VLM.
        vlm_backend: Override VLM backend independently (``"gemini"`` or
            ``"replicate"``).  Defaults to the value of ``backend``.
    """
    from clawphd.agent.tools.paperbanana_providers import (
        GeminiVLM,
        GeminiImageGen,
        ReplicateVLM,
        ReplicateImageGen,
        ReferenceStore,
    )

    vlm_backend = vlm_backend or backend

    google_key = os.environ.get("GOOGLE_API_KEY")
    replicate_token = os.environ.get("REPLICATE_API_TOKEN")

    # ── VLM provider ───────────────────────────────────────────────────────
    if vlm_backend == "replicate":
        if not replicate_token:
            print("ERROR: Set REPLICATE_API_TOKEN  (https://replicate.com/account/api-tokens)")
            sys.exit(1)
        vlm = ReplicateVLM(
            api_token=replicate_token,
            model="google/gemini-2.5-flash",
        )
        print("  VLM backend:   Replicate (google/gemini-2.5-flash)")
    else:
        if not google_key:
            print("ERROR: Set GOOGLE_API_KEY  (free: https://makersuite.google.com/app/apikey)")
            sys.exit(1)
        vlm = GeminiVLM(api_key=google_key, model="gemini-2.0-flash")
        print("  VLM backend:   Google Gemini (direct)")

    # ── Image generation provider ──────────────────────────────────────────
    if backend == "replicate":
        if not replicate_token:
            print("ERROR: Set REPLICATE_API_TOKEN  (https://replicate.com/account/api-tokens)")
            sys.exit(1)
        image_gen = ReplicateImageGen(
            api_token=replicate_token,
            model="google/gemini-2.5-flash-image",
        )
        print("  Image backend: Replicate (google/gemini-2.5-flash-image)")
    else:
        if not google_key:
            print("ERROR: Set GOOGLE_API_KEY  (free: https://makersuite.google.com/app/apikey)")
            sys.exit(1)
        image_gen = GeminiImageGen(
            api_key=google_key,
            model="gemini-2.5-flash-preview-04-17",
        )
        print("  Image backend: Google Gemini (direct)")

    # ── Reference store (optional) ─────────────────────────────────────────
    reference_store = None
    ref_path = Path(__file__).resolve().parent.parent / "paperbanana" / "data" / "reference_sets"
    if (ref_path / "index.json").exists():
        reference_store = ReferenceStore(ref_path)
        print(f"  Reference store: {reference_store.count} examples")
    else:
        print(f"  Reference store: not found at {ref_path}")

    return vlm, image_gen, reference_store


# ---------------------------------------------------------------------------
# 2.  Build a clawphd AgentLoop with PaperBanana tools enabled
# ---------------------------------------------------------------------------

def _build_agent(vlm, image_gen, reference_store):
    """Return an AgentLoop wired with diagram tools."""
    from clawphd.config.loader import load_config
    from clawphd.bus.queue import MessageBus
    from clawphd.providers.litellm_provider import LiteLLMProvider
    from clawphd.agent.loop import AgentLoop

    config = load_config()

    api_key = config.get_api_key()
    api_base = config.get_api_base()
    model = config.agents.defaults.model

    if not api_key and not model.startswith("bedrock/"):
        print("ERROR: No LLM API key in ~/.clawphd/config.json")
        sys.exit(1)

    bus = MessageBus()
    provider = LiteLLMProvider(api_key=api_key, api_base=api_base, default_model=model)

    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=model,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        restrict_to_workspace=False,
        # ── PaperBanana providers ──
        vlm_provider=vlm,
        image_gen_provider=image_gen,
        reference_store=reference_store,
    )

    return agent


# ---------------------------------------------------------------------------
# 3.  Example prompts
# ---------------------------------------------------------------------------

DIAGRAM_PROMPT = """\
Generate a methodology diagram for the following paper section.  please read the diagram-gen skill first for better diagram

## Methodology
Our framework, PaperBanana, automates the generation of publication-quality
academic illustrations through a multi-agent pipeline. The system takes as
input a methodology section (S) and a figure caption (C).

Phase 1 – Linear Planning:
1. The Retriever agent selects the top-10 most relevant reference examples
   from a curated set of high-quality diagrams.
2. The Planner agent uses in-context learning from these examples to
   generate a detailed textual description (P) of the target diagram.
3. The Stylist agent refines the description to optimise visual aesthetics (P*).

Phase 2 – Iterative Refinement:
4. The Visualiser agent renders the description into an image using a
   text-to-image generation model.
5. The Critic agent evaluates the image on faithfulness, conciseness,
   readability, and aesthetics, providing targeted revision feedback.
6. Steps 4-5 repeat for up to 3 iterations until quality is satisfactory.

## Caption
Overview of the PaperBanana multi-agent framework for automated academic
illustration generation.
"""

PLOT_PROMPT = """\
IMPORTANT: Please read the diagram-gen skill first for better diagram.

Generate a statistical plot comparing LLM benchmark performance. 

## Source context
Table 1: Performance comparison of different models on three benchmarks.

| Model     | MMLU  | HellaSwag | ARC-C |
|-----------|-------|-----------|-------|
| GPT-4o    | 88.7  | 95.3      | 96.4  |
| Claude 3  | 86.8  | 93.7      | 93.5  |
| Gemini    | 85.0  | 87.8      | 89.8  |
| Llama 3   | 79.2  | 82.0      | 83.4  |
| Mistral   | 75.3  | 81.4      | 78.6  |

## Caption
Performance comparison of frontier LLMs across three benchmarks (MMLU,
HellaSwag, ARC-Challenge).

## Raw data (JSON)
""" + json.dumps({
    "models": ["GPT-4o", "Claude 3", "Gemini", "Llama 3", "Mistral"],
    "MMLU": [88.7, 86.8, 85.0, 79.2, 75.3],
    "HellaSwag": [95.3, 93.7, 87.8, 82.0, 81.4],
    "ARC-C": [96.4, 93.5, 89.8, 83.4, 78.6],
}, indent=2) + """

Use `diagram_type="statistical_plot"` and pass the raw data.
"""


# ---------------------------------------------------------------------------
# 4.  Main
# ---------------------------------------------------------------------------

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="ClawPhD PaperBanana example")
    parser.add_argument(
        "--mode", choices=["diagram", "plot"], default="diagram",
        help="What to generate (default: diagram)",
    )
    parser.add_argument(
        "--backend", choices=["gemini", "replicate"], default="replicate",
        help="Backend for both VLM and image-gen (default: gemini)",
    )
    parser.add_argument(
        "--vlm-backend", choices=["gemini", "replicate"], default="replicate",
        help="Override VLM backend independently (default: same as --backend)",
    )
    args = parser.parse_args()

    prompt = DIAGRAM_PROMPT if args.mode == "diagram" else PLOT_PROMPT

    print("=" * 60)
    print(f"  ClawPhD × PaperBanana  —  mode: {args.mode}  backend: {args.backend}")
    print("=" * 60)

    # Build providers
    print("\n[1/3] Creating providers …")
    vlm, image_gen, ref_store = _build_providers(
        backend=args.backend, vlm_backend=args.vlm_backend
    )

    # Build agent
    print("[2/3] Wiring clawphd AgentLoop …")
    agent = _build_agent(vlm, image_gen, ref_store)

    # Run
    print(f"[3/3] Sending request to agent (mode={args.mode}) …\n")
    response = await agent.process_direct(
        content=prompt,
        session_key="example:paperbanana",
    )

    print("\n" + "─" * 60)
    print("Agent response:\n")
    print(response)
    print("─" * 60)


if __name__ == "__main__":
    asyncio.run(main())
