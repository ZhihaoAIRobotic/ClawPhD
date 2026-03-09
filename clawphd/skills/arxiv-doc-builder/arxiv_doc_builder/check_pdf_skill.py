#!/usr/bin/env python3
"""
Check if pdf skill is available in the environment.

Searches workspace skills directories for a pdf skill with SKILL.md.
"""

from pathlib import Path

SKILL_SEARCH_PATHS = [
    Path("skills/pdf"),
    Path.home() / ".clawphd/skills/pdf",
]


def check_pdf_skill_available() -> bool:
    """Check if pdf skill is available."""
    return any((loc / "SKILL.md").exists() for loc in SKILL_SEARCH_PATHS)


def get_pdf_skill_path() -> Path | None:
    """Get the path to pdf skill if available."""
    for loc in SKILL_SEARCH_PATHS:
        if (loc / "SKILL.md").exists():
            return loc
    return None


if __name__ == "__main__":
    if check_pdf_skill_available():
        skill_path = get_pdf_skill_path()
        print(f"✓ pdf skill found at: {skill_path}")
        exit(0)
    else:
        print("✗ pdf skill not found")
        exit(1)
