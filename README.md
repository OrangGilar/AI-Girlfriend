# Persona Chatbot Multi-Character Anime Companion

A Transformer-based persona chatbot supporting **6 unique characters** with
distinct dere personalities, gender-aware interactions, rolling memory, and
quantized inference on Google Colab's free T4 GPU.

---

## Characters

| # | Name | Type | Gender |
|---|------|------|--------|
| 1 | Mai Sakurajima (桜島麻衣) | Tsundere | Female |
| 2 | Yuki Amane (天音雪) | Kuudere | Female |
| 3 | Hana Mizuki (水木花) | Dandere | Female |
| 4 | Sora Himari (陽鞠空) | Deredere | Female |
| 5 | Rei Shirogane (白銀零) | Himedere | Female |
| 6 | Kaito Ryusei (流星海斗) | Kuudere | Male |

---

## Architecture

Testing-Kanojo/
├── config/
│   ├── config.json          # Model & memory hyperparameters
│   └── characters.json      # All character definitions
├── src/
│   ├── persona.py           # Persona loader & gender-aware system prompt
│   ├── character_select.py  # Interactive character & gender selection
│   ├── model.py             # Transformer chatbot (Qwen 2.5-7B, 4-bit)
│   ├── memory.py            # Rolling summarization memory
│   └── evaluation.py        # BLEU, perplexity, persona vs baseline
├── data/
│   ├── anime_dialogues.txt  # Mai reference dialogues (30 pairs)
│   ├── dialogues_yuki.txt   # Yuki reference dialogues (20 pairs)
│   ├── dialogues_hana.txt   # Hana reference dialogues (20 pairs)
│   ├── dialogues_sora.txt   # Sora reference dialogues (20 pairs)
│   ├── dialogues_rei.txt    # Rei reference dialogues (20 pairs)
│   └── dialogues_kaito.txt  # Kaito reference dialogues (20 pairs)
└── AI_Girlfriend.ipynb      # Main notebook

---

## Quick Start (Google Colab)

1. Open `AI_Girlfriend.ipynb` in Google Colab.
2. Set Runtime → **T4 GPU** (Runtime → Change runtime type).
3. Click **Runtime → Run all**.
4. In Cell 0-A, **choose a character** and **your gender** when prompted.
5. The model loads automatically (~3-4 minutes on first run).
6. Chat in Section 3.

---

## Key Features

### Multi-Character Selection
Run `select_character()` at startup to pick from 6 characters across 5 dere
archetypes. Each character has a unique personality, speech style, background,
and 20 reference dialogue pairs for evaluation.

### Gender-Aware Interactions
`select_user_gender()` at startup sets whether the user is male, female, or
neutral. The system prompt is adjusted accordingly — romantic undertones,
terms of address, and tone all shift based on the combination of user gender
and character gender.

### Hot-Swap Characters
The optional Character Switch cell lets you swap personas mid-session without
reloading the model. Memory is cleared on switch.

### Rolling Memory
`ConversationMemory` uses a sliding window of the last `window_size` turns
(default: 10). When the window fills, older turns are condensed into a summary
using `distilbart-cnn-12-6`, which is injected back into the system prompt.

### Evaluation
- **BLEU-4** (corpus-level, smoothed): persona-conditioned vs. baseline
- **Perplexity**: sliding-window on each character's reference dialogue file
- **Side-by-side table**: qualitative comparison across 10 sample turns

---

## Hyperparameters (`config/config.json`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| model | Qwen/Qwen2.5-7B-Instruct | ~7B params, instruction-tuned |
| quantization | 4bit | NF4, double quant, fp16 compute |
| temperature | 0.8 | Controls randomness |
| top_p | 0.9 | Nucleus sampling |
| top_k | 50 | Top-k filtering |
| repetition_penalty | 1.1 | Reduces looping |
| max_new_tokens | 120 | Max reply length |
| memory window | 10 turns | Before summarization triggers |
| seed | 42 | Reproducibility |

---

## Reproducibility

All random states are seeded at `42` via:

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
```

`reset_conversation()` re-seeds the generator to ensure deterministic replay.

---

## Limitations & Ethics

- **Persona accuracy**: BLEU scores are low (~0.003) because BLEU is a poor
  metric for open-domain dialogue; qualitative reading of the comparison table
  is more informative.
- **Character bias**: All female characters default to heteronormative
  romantic framing toward male users. This reflects the source material's
  conventions and may not represent all users.
- **Model bias**: Qwen 2.5-7B was trained on broad web data; it may
  occasionally produce responses inconsistent with a character's defined
  personality, particularly for edge-case or emotionally ambiguous prompts.
- **Perplexity inflation**: Stylized anime dialogue diverges from the model's
  training distribution (general web text), so perplexity is expected to be
  higher than on standard benchmarks.
- **Dataset size**: Each character has 20–30 reference pairs — sufficient for
  comparative evaluation but too small for fine-tuning.
- **Content handling**: The persona system prompt includes a dislike for
  "perverted comments"; however, the underlying LLM is the final gatekeeper
  and may not always enforce this in out-of-distribution inputs.

---

## Adding a New Character

1. Add an entry to `config/characters.json` following the existing schema.
2. Create `data/dialogues_<id>.txt` with at least 15 USER/CHARACTER pairs.
3. The character will appear automatically in `select_character()` on next run.
