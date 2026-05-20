"""Persona prompt manager.

Loads the persona definition from ``config/config.json`` and builds the
context prefix that is prepended to every conversation turn before the
model generates a response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Persona:
    name: str
    age: int
    personality: str
    speech_style: str
    background: str
    dere_type: str = ""
    character_gender: str = "female"
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    greeting_neutral: str = ""
    greeting_male: str = ""
    greeting_female: str = ""
    dialogue_file: str = "data/anime_dialogues.txt"
    user_gender: str = "neutral"

    @property
    def greeting(self) -> str:
        if self.user_gender == "male":
            return self.greeting_male or self.greeting_neutral
        if self.user_gender == "female":
            return self.greeting_female or self.greeting_neutral
        return self.greeting_neutral

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config/config.json",
        character_id: str = "mai",
        user_gender: str = "neutral",
    ) -> "Persona":
        path = Path(config_path)

        char_path = path.parent / "characters.json"
        load_path = char_path if char_path.exists() else path

        with load_path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        if "characters" in data:
            chars = data["characters"]
            char = next((c for c in chars if c["id"] == character_id), chars[0])
        else:
            char = data["persona"]
            char.setdefault("id", "mai")
            char.setdefault("dere_type", "Tsundere")
            char.setdefault("gender", "female")
            char.setdefault("greeting_neutral", char.get("greeting", ""))
            char.setdefault("greeting_male", char.get("greeting", ""))
            char.setdefault("greeting_female", char.get("greeting", ""))
            char.setdefault("dialogue_file", "data/anime_dialogues.txt")

        raw_name = char.get("display_name", char.get("name", "Unknown"))
        name = raw_name.split(" (")[0].strip()

        return cls(
            name=name,
            age=char.get("age", 17),
            personality=char["personality"],
            speech_style=char["speech_style"],
            background=char.get("background", ""),
            dere_type=char.get("dere_type", ""),
            character_gender=char.get("gender", char.get("character_gender", "female")),
            likes=char.get("likes", []),
            dislikes=char.get("dislikes", []),
            greeting_neutral=char.get("greeting_neutral", char.get("greeting", "")),
            greeting_male=char.get("greeting_male", char.get("greeting", "")),
            greeting_female=char.get("greeting_female", char.get("greeting", "")),
            dialogue_file=char.get("dialogue_file", "data/anime_dialogues.txt"),
            user_gender=user_gender,
        )

    def build_prefix(self, memory_summary: str = "") -> str:
        likes = ", ".join(self.likes) if self.likes else "—"
        dislikes = ", ".join(self.dislikes) if self.dislikes else "—"

        if self.user_gender == "male":
            gender_note = (
                "The person you are talking to is male. "
                "If your character is female and romantically interested, "
                "reflect that subtly through your tone — protective, teasing, "
                "or quietly affectionate toward a boy. "
                "If your character is male, treat him as a close male companion "
                "with a respectful, casual dynamic."
            )
        elif self.user_gender == "female":
            gender_note = (
                "The person you are talking to is female. "
                "If your character is female, treat her with warm sisterly or "
                "friendly closeness. "
                "If your character is male and romantically interested, reflect "
                "that through quiet protectiveness and understated care toward a girl."
            )
        else:
            gender_note = ""

        lines = [
            f"You are {self.name}, a {self.age}-year-old anime character ({self.dere_type}). "
            f"Stay fully in character at all times. Never break the fourth wall.",
            f"Personality: {self.personality}",
            f"Speech style: {self.speech_style}",
            f"Background: {self.background}",
            f"Likes: {likes}",
            f"Dislikes: {dislikes}",
        ]
        if gender_note:
            lines.append(gender_note)
        lines.append(
            "Reply in 1-3 short sentences. "
            "Do not narrate actions in asterisks. "
            "Do not describe yourself in the third person."
        )
        if memory_summary:
            lines.append(f"Recent context: {memory_summary}")

        return "\n".join(lines)

    def format_turn(self, user_input: str, memory_summary: str = "") -> str:
        prefix = self.build_prefix(memory_summary=memory_summary)
        return f"{prefix}\nUser: {user_input}\n{self.name}:"


def load_persona(
    config_path: str | Path = "config/config.json",
    character_id: str = "mai",
    user_gender: str = "neutral",
) -> Persona:
    return Persona.from_config(config_path, character_id=character_id, user_gender=user_gender)