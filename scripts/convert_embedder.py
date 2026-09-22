#!/usr/bin/env python3
"""Convert the `facebook/dinov2-small` checkpoint to a Core ML `.mlpackage`
so `TCGPortfolioTracker` can run the *same* embedding space on-device that
`scripts/build_index.py` used to build the prototype index — see
docs/tcg_detection_pipeline.md §3.3.

The exported Core ML model's output IS the final 384-dim L2-normalized
embedding — CLS-token extraction and normalization are baked into the traced
graph, mirroring exactly what `scripts/_remote_embed.py` /
`TCG_ETL/transform/embeddings.py` do at inference (forward pass ->
`last_hidden_state[:, 0, :]` -> L2-normalize). Nothing is left for the app
to reimplement downstream.

Pixel normalization (rescale 1/255 + ImageNet mean/std, as reported by this
exact checkpoint's `AutoImageProcessor`) is also baked in, via `ct.ImageType`
per-channel `scale`/`bias`, so the Core ML model accepts a raw 224x224 RGB
image directly. Resize (shortest-edge 256) + center-crop (224x224) is
*not* baked in — Core ML's `ImageType` has no resize/crop step of its own,
so the app is expected to hand Core ML an already 224x224 image (standard
practice for fixed-input-shape vision models; e.g. via Vision/CoreImage
before prediction).

Usage:

    python scripts/convert_embedder.py \
        --out data/index/dinov2_small_embedder.mlpackage \
        --verify-images data/tcg_identifier/test/pokemon/bwp-BW07.png \
                         data/tcg_identifier/test/yugioh/<some file>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
INDEX_OUT_DIR = DATA_ROOT / "index"

DINOV2_MODEL_NAME = "facebook/dinov2-small"
EMBEDDING_DIMENSION = 384
INPUT_SIZE = 224  # verified against this checkpoint's AutoImageProcessor at conversion time, not assumed


def _freeze_position_embeddings_for_input_size(model: "Any", input_size: int) -> None:
    """`facebook/dinov2-small` is pretrained at image_size=518 (37x37 patch
    grid) and interpolates its position embeddings at runtime (bicubic) to
    whatever input size is actually given — here always 224x224 (16x16
    patches), since our AutoImageProcessor crops to a fixed size.

    `torch.jit.trace` records that runtime `F.interpolate(..., mode="bicubic")`
    call as an `upsample_bicubic2d` op, which coremltools's PyTorch converter
    does not implement, so conversion fails.

    Since the input size is fixed for this export, there is nothing dynamic
    to trace: we precompute the interpolated position embeddings exactly
    once, using the model's own `interpolate_pos_encoding` (same bicubic
    math, same numbers), and bake that fixed 16x16-grid tensor in as the
    model's `position_embeddings` parameter. `Dinov2Embeddings.forward`'s own
    interpolation call then takes its early-return path (`num_patches ==
    num_positions and height == width`), so no interpolate op is traced at
    all — this changes nothing numerically, it only moves the interpolation
    from trace-time to conversion-time.
    """
    import torch

    embeddings_module = model.embeddings
    patch_size = model.config.patch_size
    num_patches_per_side = input_size // patch_size
    dummy_embeddings = torch.zeros(1, num_patches_per_side * num_patches_per_side + 1, model.config.hidden_size)
    with torch.no_grad():
        new_pos_embed = embeddings_module.interpolate_pos_encoding(dummy_embeddings, input_size, input_size)
        embeddings_module.position_embeddings = torch.nn.Parameter(new_pos_embed)


def _wrap_model(model: "Any", input_size: int, mean: list[float], std: list[float]) -> tuple["Any", "Any"]:
    """Build the traced module: mean/std normalize -> forward pass -> CLS
    token -> L2-normalize. Input is expected in [0, 1] range (Core ML's
    ImageType does the 1/255 rescale before this).

    Returns (wrapper_module, example_input).
    """
    import torch

    _freeze_position_embeddings_for_input_size(model, input_size)

    class Dinov2Embedder(torch.nn.Module):
        """Forward pass -> CLS token -> L2-normalize, with the per-channel
        mean/std normalize step baked in as explicit tensor ops (see
        `_wrap_model`'s docstring for why this isn't done via
        `ct.ImageType`'s scale/bias instead)."""

        def __init__(self, dinov2: torch.nn.Module, mean: list[float], std: list[float]) -> None:
            super().__init__()
            self.dinov2 = dinov2
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        def forward(self, pixel_values_0_1: "torch.Tensor") -> "torch.Tensor":
            normalized = (pixel_values_0_1 - self.mean) / self.std
            out = self.dinov2(pixel_values=normalized)
            cls_token = out.last_hidden_state[:, 0, :]
            return torch.nn.functional.normalize(cls_token, p=2, dim=1)

    wrapper = Dinov2Embedder(model, mean, std).eval()
    example_input = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    return wrapper, example_input


def cmd_convert(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {DINOV2_MODEL_NAME} + its AutoImageProcessor...", flush=True)
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_NAME)
    model = AutoModel.from_pretrained(DINOV2_MODEL_NAME)
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    hidden = getattr(model.config, "hidden_size", None)
    if hidden != EMBEDDING_DIMENSION:
        sys.exit(f"unexpected hidden size {hidden}, expected {EMBEDDING_DIMENSION}")

    # --- confirm (not assume) the processor's actual preprocessing knobs ---
    proc_cfg = processor.to_dict()
    crop_size = proc_cfg.get("crop_size") or {}
    input_size = crop_size.get("height") or proc_cfg.get("size", {}).get("shortest_edge")
    if not input_size:
        sys.exit(f"could not determine processor's output image size from config: {proc_cfg}")
    input_size = int(input_size)
    if input_size != INPUT_SIZE:
        print(
            f"WARNING: processor's confirmed crop size is {input_size}x{input_size}, "
            f"not the expected {INPUT_SIZE}x{INPUT_SIZE} — using the confirmed value.",
            file=sys.stderr,
        )
    image_mean = proc_cfg.get("image_mean")
    image_std = proc_cfg.get("image_std")
    rescale_factor = proc_cfg.get("rescale_factor", 1 / 255)
    if not image_mean or not image_std:
        sys.exit(f"processor config has no image_mean/image_std: {proc_cfg}")
    print(f"confirmed input size: {input_size}x{input_size}")
    print(f"confirmed image_mean: {image_mean}")
    print(f"confirmed image_std: {image_std}")
    print(f"confirmed rescale_factor: {rescale_factor}")

    # Core ML ImageType applies: pixel_out = pixel_in * scale + bias, where
    # `scale`/`bias` as *lists* (one value per channel) is documented as
    # supported by coremltools's ImageType, but combined with a channel-first
    # PyTorch input this specific coremltools version (9.0) hits a broadcast
    # bug in its `insert_image_preprocessing_op` backend pass (shape
    # (1,3,H,W) vs (1,1,1,3) — a channel-last bias tensor built for a
    # channel-first graph). Worked around by only using ImageType's *scalar*
    # rescale (1/255, same for every channel — always safe) and baking the
    # per-channel mean/std normalize into the traced graph itself instead
    # (see `_wrap_model`) — numerically identical to AutoImageProcessor's
    # `(pixel/255 - mean) / std`, just split across two components instead of
    # one.
    scale = rescale_factor
    bias = 0.0
    print(f"ImageType scale (scalar 1/255 rescale only): {scale}")
    print("per-channel mean/std normalize baked into the traced graph instead of ImageType bias (see comment above)")

    wrapper, example_input = _wrap_model(model, input_size, image_mean, image_std)

    print("tracing with torch.jit.trace...", flush=True)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example_input, strict=False)

    print("sanity-checking traced module against the eager module...", flush=True)
    with torch.inference_mode():
        eager_out = wrapper(example_input)
        traced_out = traced(example_input)
    import torch.nn.functional as F

    trace_cos = F.cosine_similarity(eager_out, traced_out).item()
    print(f"  eager vs traced cosine similarity (zeros input): {trace_cos:.6f}")
    if trace_cos < 0.999:
        sys.exit("traced module diverges from eager module — aborting before conversion")

    import coremltools as ct

    print("converting to Core ML (mlprogram)...", flush=True)
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, input_size, input_size),
                scale=scale,
                bias=bias,
                color_layout=ct.colorlayout.RGB,
                channel_first=True,
            )
        ],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )
    print(f"converted in {time.time() - t0:.1f}s", flush=True)

    mlmodel.short_description = (
        "DINOv2-small (facebook/dinov2-small) image embedder: forward pass -> "
        "CLS token (last_hidden_state[:, 0, :]) -> L2-normalize. Input must "
        "already be resized/center-cropped to the fixed input size (see "
        "conversion manifest) before prediction; pixel rescale (1/255, via "
        "ImageType) and per-channel mean/std normalize (baked into the "
        "graph) are both handled internally — feed a raw 0-255 RGB image."
    )
    mlmodel.author = "TGCDatasets/scripts/convert_embedder.py"
    mlmodel.version = args.checkpoint_revision or "unknown"

    mlmodel.save(str(out_path))
    print(f"wrote {out_path}")

    # --- version manifest sidecar ---
    import torch as _torch
    import transformers as _transformers

    manifest = {
        "model_name": DINOV2_MODEL_NAME,
        "checkpoint_revision": args.checkpoint_revision,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "input_size": input_size,
        "image_mean": image_mean,
        "image_std": image_std,
        "rescale_factor": rescale_factor,
        "coreml_image_type_scale": scale,
        "coreml_image_type_bias": bias,
        "convert_to": "mlprogram",
        "minimum_deployment_target": "iOS16",
        "torch_version": _torch.__version__,
        "transformers_version": _transformers.__version__,
        "coremltools_version": ct.__version__,
        "output_path": args.out,
        "build_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "postprocessing": "last_hidden_state[:, 0, :] (CLS token), L2-normalized (p=2, dim=1) — baked into the exported graph",
        "preprocessing_not_baked_in": "resize shortest-edge + center-crop to input_size x input_size (caller's responsibility before prediction)",
    }
    manifest_path = Path(args.manifest_out).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")

    if args.verify_images:
        _verify(processor, model, mlmodel, args.verify_images, input_size)


def _verify(processor: "Any", model: "Any", mlmodel: "Any", image_paths: list[str], input_size: int) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    print("\n--- verification: PyTorch/transformers path vs Core ML path ---", flush=True)
    sims = []
    for path_str in image_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"  ! skipping missing verify image: {path}", file=sys.stderr)
            continue
        with Image.open(path) as im:
            rgb = im.convert("RGB").copy()

        # Reference path: exactly what scripts/_remote_embed.py does.
        with torch.inference_mode():
            inputs = processor(images=[rgb], return_tensors="pt")
            ref = model(**inputs).last_hidden_state[:, 0, :]
            ref = F.normalize(ref.float(), p=2, dim=1).numpy()[0]

        # Core ML path: feed the same 224x224 center-cropped RGB image (the
        # resize/crop AutoImageProcessor already applied is reused here via
        # `inputs`'s pixel_values, converted back to a PIL image, so both
        # paths see pixel-identical input and only the normalize+model step
        # differs between backends).
        pixel_values = inputs["pixel_values"][0]  # already resized/cropped/normalized, CHW float
        # Undo AutoImageProcessor's normalize+rescale to reconstruct the raw
        # 0-255 uint8 224x224 image Core ML's ImageType expects (since the
        # baked-in scale/bias will redo rescale+normalize identically).
        mean = torch.tensor(processor.image_mean).view(3, 1, 1)
        std = torch.tensor(processor.image_std).view(3, 1, 1)
        raw = ((pixel_values * std + mean) * 255.0).clamp(0, 255).round().byte()
        raw_np = raw.permute(1, 2, 0).numpy()  # HWC uint8
        pil_input = Image.fromarray(raw_np, mode="RGB")

        pred = mlmodel.predict({"image": pil_input})
        coreml_vec = np.asarray(pred["embedding"]).reshape(-1)

        cos = float(np.dot(ref, coreml_vec) / (np.linalg.norm(ref) * np.linalg.norm(coreml_vec) + 1e-12))
        sims.append(cos)
        print(f"  {path.name}: cosine similarity = {cos:.6f}")

    if sims:
        avg = sum(sims) / len(sims)
        print(f"\naverage cosine similarity over {len(sims)} image(s): {avg:.6f}")
        if avg < 0.999:
            print("WARNING: cosine similarity is lower than expected for a faithful conversion — investigate before trusting this .mlpackage", file=sys.stderr)
        else:
            print("conversion verified: Core ML output matches the PyTorch/transformers reference path.")
    else:
        print("no verify images found — conversion NOT numerically verified", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        default=str(INDEX_OUT_DIR / "dinov2_small_embedder.mlpackage"),
        help="output .mlpackage path (default: data/index/dinov2_small_embedder.mlpackage)",
    )
    parser.add_argument(
        "--manifest-out",
        default=str(INDEX_OUT_DIR / "embedder_conversion_manifest.json"),
        help="output sidecar JSON path (default: data/index/embedder_conversion_manifest.json)",
    )
    parser.add_argument(
        "--checkpoint-revision",
        default="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        help="HF revision/commit hash of facebook/dinov2-small to record in the manifest "
        "(default: the revision resolved for this repo's cached checkpoint at authoring time)",
    )
    parser.add_argument(
        "--verify-images",
        nargs="*",
        default=[],
        help="one or more local image paths to numerically verify the conversion against "
        "(PyTorch/transformers reference vs Core ML prediction, cosine similarity)",
    )
    parser.set_defaults(func=cmd_convert)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
