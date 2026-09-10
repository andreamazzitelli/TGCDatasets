# Create ML workflow

Targets the **Create ML** app (Applications → Create ML, or via Xcode →
Open Developer Tool) on macOS, using the **Image Classification** template.
You'll build two separate projects/models, one per folder tree below.

## 1. Folder layout Create ML expects

Create ML's image classifier ingests a folder of subfolders, one subfolder
per label, subfolder name = label name:

```
data/tcg_identifier/train/
  pokemon/            *.jpg / *.png / *.heic ...
  magic_the_gathering/
  yugioh/
  lorcana/
  one_piece/
  digimon/
  dragon_ball_super/
  flesh_and_blood/
  star_wars_unlimited/
  weiss_schwarz/
  other_tcg/
  not_a_card/

data/tcg_identifier/test/      # optional, same subfolder structure
  ...
```

and identically shaped for grading:

```
data/grading_status/train/
  raw/
  psa/
  bgs/
  cgc/
  sgc/
  other_graded/

data/grading_status/test/      # optional
  ...
```

`scripts/scaffold_dataset.sh` creates exactly this tree (empty, ready for
you to drop images into).

- **`train/` only, no `test/`:** Create ML automatically holds out ~20% of
  `train/` as a validation set during training. Fine for early iterations.
- **With `test/`:** after training, Create ML reports accuracy against a
  held-out set that never influenced training/validation — do this once
  you have enough data to spare, it's a much more honest accuracy number.
  Keep `test/` images disjoint from `train/` (don't photograph the same
  physical card from a slightly different angle for both — that leaks).

## 2. Building the model

1. Open Create ML → **New Document** → **Image Classification**.
2. Drag `data/tcg_identifier/train` onto the Training Data well (and
   `data/tcg_identifier/test` onto Testing Data, if you built one).
3. **Augmentations** (right-hand panel) — Create ML applies these on the fly
   during training, multiplying effective data without new photos:
   - `Crop`, `Rotate`, `Blur`, `Expose` (brightness), `Noise`: generally safe
     and useful for both models.
   - `Flip` (horizontal): fine for the TCG Identifier model. **Turn this off
     for the Grading Status model** — grading labels have fixed text/layout,
     and a flipped slab isn't something the model will ever see for real.
   - Start with a moderate combination (e.g. Crop + Rotate + Expose +
     Noise) rather than all of them at max — over-aggressive augmentation on
     a small dataset can hurt more than help. Iterate.
4. **Train.** Watch the training/validation accuracy curve — if validation
   accuracy plateaus far below training accuracy, that's overfitting
   (usually means: not enough data variety, or classes too imbalanced —
   go back to `data_collection.md`).
5. Repeat steps 1–4 for `data/grading_status`.

## 3. Evaluating

- Check the **per-class** precision/recall in the Testing tab, not just
  overall accuracy — with imbalanced or small classes, overall accuracy can
  look fine while one class (e.g. `sgc`, `other_tcg`) is being predicted
  almost randomly.
- Confusion between specific classes tells you where to add data next:
  - TCG model confusing two games → add more varied angles/lighting for
    those two specifically, not the whole dataset.
  - Grading model confusing `bgs` with `cgc` → both are common
    look-alikes at a glance (colored border + slab); add close-up label
    shots so the model has legible detail to key on, not just overall slab
    silhouette.

## 4. Export & use

- Export as an `.mlmodel` (File → the model's output, or drag from the
  Output tab) — Create ML models are ready to drop straight into an Xcode
  project via Vision/Core ML (`VNCoreMLModel`).
- Keep both `.mlmodel` files (TCG Identifier, Grading Status) out of this
  repo unless you want them version-controlled — they're binary build
  artifacts of the data here, not the dataset itself. If you do want them
  tracked, use Git LFS rather than committing them directly (Core ML models
  are often tens of MB+).
- At inference: run the same input image through both models and combine
  the two labels (e.g. `pokemon` + `psa` → "Pokémon, PSA-graded"). If the
  Grading Status model says `raw`, you don't need a grading company label at
  all — that's the expected/normal case, not a missing prediction.

## 5. Iterating

Treat this as a loop, not a one-shot build: train → check per-class
metrics → source more data for the weak classes → retrain. Given the class
list here is deliberately extensible (`other_tcg`, `other_graded`), you can
start with fewer images per class, ship a v1, and keep improving specific
classes over time without restructuring the dataset.
