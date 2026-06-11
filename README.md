# AI Game Recommender

A content-based Steam game recommender with a natural-language front end.
Describe what you want in plain English — including references to other games —
and an LLM translates your intent into tags, which drive a nearest-neighbor
search over ~30k games.

## Example

```
> Like Counter-Strike but slower and more tactical

CTU: Counter Terrorism Unit
Full Spectrum Warrior
Breach & Clear
Full Spectrum Warrior: Ten Hammers
TACTICAL
Tom Clancy's Rainbow Six® 3 Gold
UFO: Aftershock
Special Tactics
Brigade E5: New Jagged Union
Elementals Reborn
```

The model has no idea what "Counter-Strike" or "slower" means as data — the LLM
does that reasoning, mapping the description to tags like `tactical` and
`realistic`, and the retrieval layer finds games closest to that tag profile.

## How it works

```
user notes ──▶ LLM (tag parser) ──▶ want/avoid tags
                                          │
                                          ▼
                              raw query vector (over tag vocab)
                                          │
                                   TF-IDF transform
                                          │
                                          ▼
                          NearestNeighbors (cosine)  ◀── prebuilt game index
                                          │
                                          ▼
                                  nearest games
```

1. **LLM tag parser** — the user's free-text notes (and any referenced games)
   are converted into `want` / `avoid` tags. The LLM is constrained to choose
   only from the dataset's real tag vocabulary, and its output is validated
   against that vocabulary before use.
2. **Query vector** — the chosen tags become a vector over the full tag space.
3. **TF-IDF + cosine KNN** — games and queries are TF-IDF weighted so rare,
   distinctive tags outweigh ubiquitous ones, then compared by cosine
   similarity (direction, not magnitude) to find the nearest games.

This is **content-based retrieval with LLM query understanding** — not a trained
model. The "intelligence" lives in the feature representation and the LLM parser.

## Design decisions

- **Cosine, not Euclidean** — tag values are vote counts, so popular games have
  large vectors regardless of character. Cosine compares direction (the tag
  _mix_) and ignores magnitude (popularity).
- **TF-IDF weighting** — a tag like `action` appears on most games and carries
  little signal; rare tags like `tactical` are highly distinguishing. TF-IDF
  down-weights common tags so similarity is driven by what's distinctive.
- **LLM constrained to a fixed vocabulary** — the parser may only emit tags that
  exist as columns, and outputs are filtered against the vocabulary. Small local
  models drift, so this guardrail is load-bearing.
- **Provider-agnostic LLM** — talks to any OpenAI-compatible endpoint via env
  vars, so it runs against local Ollama or a hosted model with no code change.

## Setup

### MUST HAVE A `.env` file. AN EXAMPLE `.env` IS AS FOLLOWS:

```
LLM_BASE_URL="http://localhost:11434/v1/"
LLM_API_KEY="ollama"
LLM_MODEL="llama3.2:latest"
KNN_PATH="./knn.joblib"
```

### Setup Instructions

```bash
pip install -r requirements.txt
# place steamspy_tag_data.csv and steam.csv in data/
python main.py
```

Uses [Ollama](https://ollama.com) by default (`ollama pull llama3.2`), but any
OpenAI-compatible API works.

## Data

Steam metadata and SteamSpy tag data (game × tag vote counts). Bundled in `data/`.

## Limitations

- Cosine discards magnitude to remove popularity bias — but magnitude also
  encodes how much tag data a game has, so obscure games with a "pure" tag
  profile can rank above well-known matches. A minimum-vote floor would address
  this.
- `avoid` tags are parsed but not yet applied as a result filter.
- Evaluation is qualitative (example queries), as content recommenders need user
  interaction data to score rigorously.
