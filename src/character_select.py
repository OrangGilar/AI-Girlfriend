"""Interactive character and user-gender selection for the Persona Chatbot.

Call select_character() and select_user_gender() at the top of the notebook
before loading the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_GENDER_OPTIONS: list[tuple[str, str]] = [
    ("male",    "Male   (he/him)"),
    ("female",  "Female (she/her)"),
    ("neutral", "Prefer not to say"),
]


def load_characters(
    config_path: str | Path = "config/characters.json",
) -> list[dict[str, Any]]:
    """Return the character list from characters.json."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return json.load(f)["characters"]


def _display_menu(characters: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 58)
    print("      ✦  PERSONA CHATBOT  —  CHARACTER SELECT  ✦")
    print("=" * 58)

    female_chars = [c for c in characters if c.get("gender") == "female"]
    male_chars   = [c for c in characters if c.get("gender") == "male"]

    if female_chars:
        print("\n  ── Female Characters ────────────────────────────────")
        for i, c in enumerate(female_chars, 1):
            dere = c.get("dere_type", "")
            print(f"  [{i}]  {c['display_name']:<34}  {dere}")

    offset = len(female_chars)
    if male_chars:
        print("\n  ── Male Characters ──────────────────────────────────")
        for j, c in enumerate(male_chars, offset + 1):
            dere = c.get("dere_type", "")
            print(f"  [{j}]  {c['display_name']:<34}  {dere}")

    print("\n" + "=" * 58)


def select_character(
    config_path: str | Path = "config/characters.json",
) -> dict[str, Any]:
    """Prompt the user to pick a character; return its config dict."""
    characters = load_characters(config_path)
    _display_menu(characters)

    while True:
        try:
            raw = input(f"  Enter number [1–{len(characters)}]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(characters):
                chosen = characters[idx]
                dere   = chosen.get("dere_type", "")
                print(f"\n  ✓  Selected  →  {chosen['display_name']}  [{dere}]\n")
                return chosen
            print(f"  Please enter a number between 1 and {len(characters)}.")
        except (ValueError, EOFError):
            print("  Invalid input — please enter a number.")


def select_user_gender() -> str:
    """Prompt the user to state their gender; return 'male', 'female', or 'neutral'."""
    print("=" * 58)
    print("  Your gender influences how your character interacts.")
    print("=" * 58)
    for i, (_, label) in enumerate(_GENDER_OPTIONS, 1):
        print(f"  [{i}]  {label}")
    print()

    while True:
        try:
            raw = input(f"  Enter number [1–{len(_GENDER_OPTIONS)}]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(_GENDER_OPTIONS):
                key, label = _GENDER_OPTIONS[idx]
                print(f"\n  ✓  Set to  →  {label}\n")
                return key
            print(f"  Please enter a number between 1 and {len(_GENDER_OPTIONS)}.")
        except (ValueError, EOFError):
            print("  Invalid input — please enter a number.")