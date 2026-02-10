"""Lightweight VLM / image-gen / reference-store providers for diagram tools.

These are self-contained — no dependency on the ``paperbanana`` package.

All classes are duck-typed to match what the tools in
``clawphd.agent.tools.paperbanana`` expect:

* **GeminiVLM**          → ``async generate(prompt, …) -> str``
* **ReplicateVLM**       → ``async generate(prompt, …) -> str``
* **GeminiImageGen**     → ``async generate(prompt, …) -> PIL.Image``
* **ReplicateImageGen**  → ``async generate(prompt, …) -> PIL.Image``
* **ReferenceStore**     → ``get_all() -> list[ReferenceExample]``
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Data model for reference examples
# ---------------------------------------------------------------------------

@dataclass
class ReferenceExample:
    """A single curated reference diagram."""

    id: str
    source_context: str
    caption: str
    image_path: str
    category: str | None = None


# ---------------------------------------------------------------------------
# ReferenceStore  (file-based, JSON index)
# ---------------------------------------------------------------------------

class ReferenceStore:
    """Manages a curated set of academic reference diagrams.

    Expects a directory containing an ``index.json`` with the format::

        {
          "examples": [
            {"id": "ref_001", "source_context": "…", "caption": "…",
             "image_path": "images/ref_001.png", "category": "flow"}
          ]
        }

    Image paths inside the JSON are resolved relative to the store directory.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._examples: list[ReferenceExample] = []
        self._loaded = False

    # -- public API ----------------------------------------------------------

    def get_all(self) -> list[ReferenceExample]:
        """Return every reference example."""
        self._load()
        return self._examples

    def get_by_category(self, category: str) -> list[ReferenceExample]:
        """Filter examples by category tag."""
        self._load()
        return [e for e in self._examples if e.category == category]

    def get_by_id(self, example_id: str) -> ReferenceExample | None:
        """Lookup a single example by ID."""
        self._load()
        return next((e for e in self._examples if e.id == example_id), None)

    @property
    def count(self) -> int:
        self._load()
        return len(self._examples)

    # -- internal ------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        index_file = self.path / "index.json"
        if not index_file.exists():
            logger.warning(f"No reference index at {self.path}")
            self._loaded = True
            return
        with open(index_file) as f:
            data = json.load(f)
        for item in data.get("examples", []):
            img = item.get("image_path", "")
            if img and not Path(img).is_absolute():
                img = str(self.path / img)
            self._examples.append(
                ReferenceExample(
                    id=item["id"],
                    source_context=item["source_context"],
                    caption=item["caption"],
                    image_path=img,
                    category=item.get("category"),
                )
            )
        logger.info(f"Loaded {len(self._examples)} reference examples")
        self._loaded = True


# ---------------------------------------------------------------------------
# GeminiVLM  (text / vision-language generation)
# ---------------------------------------------------------------------------

class GeminiVLM:
    """Gemini-based VLM provider for text generation and image understanding.

    Requires ``pip install google-genai Pillow``.

    Args:
        api_key: Google API key (free tier: https://makersuite.google.com/app/apikey).
        model: Gemini model name (default ``gemini-2.0-flash``).
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._api_key = api_key
        self._model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai is required. Install with: pip install google-genai"
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def generate(
        self,
        prompt: str,
        images: list[Any] | None = None,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text, optionally conditioned on images."""
        from google.genai import types

        client = self._get_client()

        contents: list[Any] = []
        if images:
            for img in images:
                b64 = _image_to_base64(img)
                contents.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(b64),
                        mime_type="image/png",
                    )
                )
        contents.append(prompt)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_prompt:
            config.system_instruction = system_prompt
        if response_format == "json":
            config.response_mime_type = "application/json"

        response = client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return response.text


# ---------------------------------------------------------------------------
# ReplicateVLM  (text / vision-language generation via Replicate)
# ---------------------------------------------------------------------------

class ReplicateVLM:
    """VLM provider via the Replicate API using streaming.

    Requires ``pip install replicate``.

    Works with any text/vision model on Replicate.  The default is
    ``google/gemini-2.5-flash`` but any model accepting ``prompt`` +
    ``images`` inputs will work.

    Args:
        api_token: Replicate API token (``REPLICATE_API_TOKEN`` env var
            is also picked up automatically by the ``replicate`` package).
        model: Replicate model identifier in ``owner/name`` format.
    """

    def __init__(
        self,
        api_token: str | None = None,
        model: str = "google/gemini-2.5-flash",
    ):
        self._api_token = api_token
        self._model = model

    async def generate(
        self,
        prompt: str,
        images: list[Any] | None = None,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text, optionally conditioned on images, via Replicate streaming.

        Images are uploaded as base64 data URIs.  The synchronous
        ``replicate.stream`` iterator is offloaded to a thread so the
        event loop stays free.
        """
        import asyncio

        try:
            import replicate as _replicate
        except ImportError:
            raise ImportError(
                "replicate is required. Install with: pip install replicate"
            )

        # Build input dict matching the Replicate model schema
        input_params: dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        if system_prompt:
            input_params["prompt"] = f"{system_prompt}\n\n{prompt}"

        # Convert PIL images → Replicate ``images`` field format:
        #   plain data-URI strings: ["data:image/jpeg;base64,…"]
        #   (Replicate rejects objects like {"value": "…"} – it wants strings)
        if images:
            image_uris: list[str] = []
            for img in images:
                # Use JPEG to keep the base64 payload small (~10× smaller
                # than PNG for photo-like content)
                b64 = _image_to_base64(img, fmt="JPEG", max_dim=1024)
                image_uris.append(f"data:image/jpeg;base64,{b64}")
            input_params["images"] = image_uris
        else:
            input_params["images"] = []

        # Required fields for Replicate Gemini models
        input_params.setdefault("videos", [])
        input_params.setdefault("dynamic_thinking", False)
        input_params.setdefault("top_p", 0.95)

        if response_format == "json":
            input_params["prompt"] += (
                "\n\nIMPORTANT: Respond with strict JSON only, no markdown fences."
            )

        def _stream() -> str:
            """Synchronous Replicate streaming call (offloaded to a thread)."""
            chunks: list[str] = []
            if self._api_token:
                client = _replicate.Client(api_token=self._api_token)
                for event in client.stream(self._model, input=input_params):
                    chunks.append(str(event))
            else:
                # Falls back to REPLICATE_API_TOKEN env var
                for event in _replicate.stream(self._model, input=input_params):
                    chunks.append(str(event))
            return "".join(chunks)

        return await asyncio.to_thread(_stream)


# ---------------------------------------------------------------------------
# GeminiImageGen  (text → image)
# ---------------------------------------------------------------------------

class GeminiImageGen:
    """Gemini-based image generation provider.

    Requires ``pip install google-genai Pillow``.

    Args:
        api_key: Google API key.
        model: Image generation model (default ``gemini-2.5-flash-preview-04-17``).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-preview-04-17",
    ):
        self._api_key = api_key
        self._model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai is required. Install with: pip install google-genai"
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a PIL Image from a text prompt."""
        from PIL import Image
        from google.genai import types

        client = self._get_client()

        if negative_prompt:
            prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

        ratio = width / height
        if ratio > 1.5:
            aspect = "16:9"
        elif ratio > 1.2:
            aspect = "3:2"
        elif ratio < 0.67:
            aspect = "9:16"
        elif ratio < 0.83:
            aspect = "2:3"
        else:
            aspect = "1:1"

        max_dim = max(width, height)
        size = "1K" if max_dim <= 1024 else ("2K" if max_dim <= 2048 else "4K")

        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect,
                image_size=size,
            ),
        )

        response = client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )

        # Extract image from response
        parts = None
        if getattr(response, "candidates", None):
            parts = response.candidates[0].content.parts
        else:
            parts = getattr(response, "parts", None)

        if not parts:
            raise ValueError("Gemini image response had no content parts.")

        for part in parts:
            if hasattr(part, "as_image"):
                try:
                    return part.as_image()
                except Exception:
                    pass
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                data = inline.data
                image_bytes = base64.b64decode(data) if isinstance(data, str) else data
                return Image.open(BytesIO(image_bytes))

        raise ValueError("Gemini image response did not contain image data.")


# ---------------------------------------------------------------------------
# ReplicateImageGen  (text → image via Replicate API)
# ---------------------------------------------------------------------------

class ReplicateImageGen:
    """Image generation via the Replicate API.

    Requires ``pip install replicate Pillow``.

    Works with any image model hosted on Replicate.  The default is
    ``google/gemini-2.5-flash-image`` but you can swap it for any model
    that accepts a ``prompt`` input and returns an image file.

    Args:
        api_token: Replicate API token (``REPLICATE_API_TOKEN`` env var
            is also picked up automatically by the ``replicate`` package).
        model: Replicate model identifier in ``owner/name`` format.
        output_format: Image format returned by the model (jpg, png, webp).
    """

    def __init__(
        self,
        api_token: str | None = None,
        model: str = "google/gemini-2.5-flash-image",
        output_format: str = "png",
    ):
        self._api_token = api_token
        self._model = model
        self._output_format = output_format

    async def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a PIL Image from a text prompt via Replicate.

        The call is blocking (``replicate.run`` is synchronous), so it is
        wrapped in ``asyncio.to_thread`` to avoid blocking the event loop.
        """
        import asyncio
        from PIL import Image

        try:
            import replicate
        except ImportError:
            raise ImportError(
                "replicate is required. Install with: pip install replicate"
            )

        # Compute aspect ratio string for models that accept it
        ratio = width / height
        if ratio > 1.5:
            aspect = "16:9"
        elif ratio > 1.2:
            aspect = "3:2"
        elif ratio < 0.67:
            aspect = "9:16"
        elif ratio < 0.83:
            aspect = "2:3"
        else:
            aspect = "1:1"

        if negative_prompt:
            prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

        input_params: dict[str, Any] = {
            "prompt": prompt,
            "image_input": [],          # required — empty for text-to-image
            "aspect_ratio": aspect,
            "output_format": self._output_format,
        }

        logger.info(
            "Calling Replicate image generation",
            model=self._model,
            aspect=aspect,
            prompt_len=len(prompt),
        )

        def _run() -> Any:
            """Synchronous Replicate call (offloaded to a thread)."""
            if self._api_token:
                client = replicate.Client(api_token=self._api_token)
                return client.run(self._model, input=input_params)
            # Falls back to REPLICATE_API_TOKEN env var
            return replicate.run(self._model, input=input_params)

        output = await asyncio.to_thread(_run)

        # ``replicate.run`` for image models returns a FileOutput object
        # with a ``.read()`` method that yields raw bytes.
        logger.debug("Replicate output type: {}", type(output).__name__)
        image_bytes = output.read()
        return Image.open(BytesIO(image_bytes))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _image_to_base64(
    image: Any,
    fmt: str = "PNG",
    max_dim: int | None = None,
    quality: int = 85,
) -> str:
    """Convert a PIL Image to a base64-encoded string.

    Args:
        image: PIL Image object.
        fmt: Image format — ``"PNG"`` (lossless) or ``"JPEG"`` (smaller).
        max_dim: If set, downscale the longest edge to this value (preserves
                 aspect ratio).  Useful to keep base64 payloads manageable.
        quality: JPEG quality (1–95).  Ignored for PNG.
    """
    if max_dim and max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        from PIL import Image as _PILImage  # noqa: N811

        image = image.resize(new_size, _PILImage.LANCZOS)

    # JPEG doesn't support alpha → convert RGBA to RGB
    if fmt.upper() == "JPEG" and image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")

    buf = BytesIO()
    save_kwargs: dict[str, Any] = {"format": fmt}
    if fmt.upper() == "JPEG":
        save_kwargs["quality"] = quality
    image.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
