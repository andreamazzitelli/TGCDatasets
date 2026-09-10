# Label taxonomy

## Model 1 — TCG Identifier

Single-label, multi-class. One folder per label under
`data/tcg_identifier/train/`.

Core TCGs (highest hobby volume / most likely to actually appear in your
photos):

| Label folder          | Game                                   |
|------------------------|-----------------------------------------|
| `pokemon`               | Pokémon TCG                             |
| `magic_the_gathering`   | Magic: The Gathering                    |
| `yugioh`                 | Yu-Gi-Oh!                               |
| `lorcana`                | Disney Lorcana                          |
| `one_piece`              | One Piece Card Game                     |
| `digimon`                | Digimon Card Game                       |
| `dragon_ball_super`      | Dragon Ball Super Card Game / Fusion World |
| `flesh_and_blood`        | Flesh and Blood                         |
| `star_wars_unlimited`    | Star Wars: Unlimited                    |
| `weiss_schwarz`          | Weiß Schwarz                            |

Two extra classes you should include even though they're not "a TCG":

| Label folder | Purpose |
|---|---|
| `other_tcg` | Catch-all for games you support but haven't given a dedicated class yet (Union Arena, Grand Archive, MetaZoo, Final Fantasy TCG, etc.). Prevents the model from being forced into a wrong core label. |
| `not_a_card` | Negative class: hands, tables, packaging, other collectibles, blurry/empty frames. Without this, a real camera pointed at "not a card" will still confidently pick one of the TCG labels — Create ML classifiers always emit *some* label. |

**Adding a new TCG later:** add a folder + images and retrain. Because this
is single-label multi-class (not multi-label), every existing image's label
stays correct — you're just adding a class, not restructuring existing data.

## Model 2 — Grading Status

Single-label, multi-class, **TCG-agnostic** — collect graded examples across
all TCGs into the same label folder, since what the model is actually
learning is slab/label design, not card content.

| Label folder | Meaning |
|---|---|
| `raw` | Ungraded — bare card, sleeved, or in a toploader/penny sleeve (not a hard graded slab) |
| `psa` | PSA (Professional Sports Authenticator) — red-bordered label, most common by volume |
| `bgs` | Beckett Grading Services — black label (standard); include gold ("Pristine 10") and silver-label variants in this same folder for coverage |
| `cgc` | CGC Trading Cards — blue label |
| `sgc` | SGC (Sportscard Guaranty Corporation) — white/black-bordered label |
| `other_graded` | Smaller/newer graders (TAG, Arena Club, ACE Grading, etc.) — catch-all so an unfamiliar slab doesn't get forced into PSA/BGS/CGC/SGC |

If you only care about *graded vs. not*, collapse this to two labels
(`raw` / `graded`) — simpler, needs less data, but you lose which company
graded it. The 6-class version above is recommended if you want that detail
and are willing to source more images (see `docs/data_collection.md` for
volume guidance per class).

### Why not fold grading into Model 1's classes?

`pokemon` × `{raw, psa, bgs, cgc, sgc, other_graded}` alone is already 60
cells (10 core TCGs × 6 grading states), most of which you will struggle to
find enough real photos for (e.g. "SGC-graded Star Wars: Unlimited" is a
thin market). A combined single model also can't be extended TCG-by-TCG or
grader-by-grader independently — adding one new TCG means resourcing 6 new
classes, not 1. The two-model split lets each classifier be trained,
evaluated, and improved independently, and lets the grading model benefit
from graded-card photos of *any* TCG (including sports cards, if you have
access to them) since slab design transfers across content.
