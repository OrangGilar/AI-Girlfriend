"""Evaluation metrics for the persona chatbot.

Provides BLEU score against reference replies, model perplexity on a
text corpus, and a side-by-side comparison between persona-conditioned
and baseline (no-persona) generation.
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
except ImportError:
    sentence_bleu   = None
    corpus_bleu     = None
    SmoothingFunction = None


@dataclass
class DialoguePair:
    user: str
    reference: str


def load_dialogues(path: str | Path) -> list[DialoguePair]:
    text = Path(path).read_text(encoding="utf-8")
    pairs: list[DialoguePair] = []
    user_line: Optional[str] = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("USER:"):
            user_line = line.split(":", 1)[1].strip()
        elif ":" in line and user_line is not None:
            reply = line.split(":", 1)[1].strip()
            pairs.append(DialoguePair(user=user_line, reference=reply))
            user_line = None

    return pairs


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def compute_bleu(
    hypotheses: list[str],
    references: list[str],
    smoothing: bool = True,
) -> float:
    if corpus_bleu is None or SmoothingFunction is None:
        raise ImportError("nltk is required. Run `pip install nltk`.")
    if not hypotheses:
        return 0.0
    hyps  = [_tokenize(h) for h in hypotheses]
    refs  = [[_tokenize(r)] for r in references]
    smth  = SmoothingFunction().method1 if smoothing else None
    return float(corpus_bleu(refs, hyps, smoothing_function=smth))


def compute_perplexity(
    text: str,
    model,
    tokenizer,
    device: Optional[str] = None,
    stride: int = 512,
    max_length: Optional[int] = None,
) -> float:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    encodings  = tokenizer(text, return_tensors="pt")
    input_ids  = encodings.input_ids.to(device)

    if max_length is None:
        max_length = getattr(model.config, "n_positions", 1024)

    nlls: list[torch.Tensor] = []
    prev_end = 0
    seq_len  = input_ids.size(1)
    if seq_len < 2:
        return float("nan")

    for begin in range(0, seq_len, stride):
        end      = min(begin + max_length, seq_len)
        trg_len  = end - prev_end
        ids      = input_ids[:, begin:end]
        tgt      = ids.clone()
        tgt[:, :-trg_len] = -100

        with torch.no_grad():
            out = model(ids, labels=tgt)
            nlls.append(out.loss * trg_len)

        prev_end = end
        if end == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return float(ppl.item())


def compare_persona_vs_baseline(
    chatbot,
    pairs: list[DialoguePair],
    smoothing: bool = True,
    verbose: bool = False,
) -> dict:
    persona_outputs:  list[str] = []
    baseline_outputs: list[str] = []
    references:       list[str] = []
    rows:             list[dict] = []

    for pair in pairs:
        chatbot.reset_conversation()
        persona_reply  = chatbot.generate(pair.user, use_persona=True,  record=False)

        chatbot.reset_conversation()
        baseline_reply = chatbot.generate(pair.user, use_persona=False, record=False)

        persona_outputs.append(persona_reply)
        baseline_outputs.append(baseline_reply)
        references.append(pair.reference)
        rows.append({
            "user":      pair.user,
            "reference": pair.reference,
            "persona":   persona_reply,
            "baseline":  baseline_reply,
        })

        if verbose:
            print(f"USER:      {pair.user}")
            print(f"REFERENCE: {pair.reference}")
            print(f"PERSONA:   {persona_reply}")
            print(f"BASELINE:  {baseline_reply}")
            print("-" * 60)

    return {
        "bleu_persona":  compute_bleu(persona_outputs,  references, smoothing),
        "bleu_baseline": compute_bleu(baseline_outputs, references, smoothing),
        "delta":         compute_bleu(persona_outputs,  references, smoothing)
                       - compute_bleu(baseline_outputs, references, smoothing),
        "rows": rows,
    }


def format_comparison_table(result: dict, max_rows: int = 10) -> str:
    lines = [
        f"BLEU (persona):  {result['bleu_persona']:.4f}",
        f"BLEU (baseline): {result['bleu_baseline']:.4f}",
        f"Δ:               {result['delta']:+.4f}",
        "",
        "| # | User | Reference | Persona | Baseline |",
        "|---|------|-----------|---------|----------|",
    ]

    def clip(s: str, n: int = 40) -> str:
        s = s.replace("|", "/").replace("\n", " ")
        return s if len(s) <= n else s[:n - 1] + "…"

    for i, row in enumerate(result["rows"][:max_rows], 1):
        lines.append(
            f"| {i} | {clip(row['user'])} | {clip(row['reference'])} "
            f"| {clip(row['persona'])} | {clip(row['baseline'])} |"
        )
    return "\n".join(lines)