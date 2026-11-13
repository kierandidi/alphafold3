# OpenFold3 → AlphaFold3 Weight Porting: Code Changes and Implications

This document records every change made to the AlphaFold3 (JAX/Haiku) codebase
to run OpenFold3 (PyTorch) weights, and what each change implies for anyone trying to
reproduce or re-implement AlphaFold3 from the released source.

**Repositories**
- AF3 source: [sokrypton/alphafold3](https://github.com/sokrypton/alphafold3) (fork of google-deepmind/alphafold3)
- OF3 source: [aqlaboratory/openfold3](https://github.com/aqlaboratory/openfold)
- OF3 weights: `s3://openfold/staging/of3-p2-155k.pt` (public, no sign required)

---

## Background

Google DeepMind released the AlphaFold3 source code but not the model weights
(commercial use restricted). AlQuraishi Lab released OpenFold3 with public weights
for the same architecture. Since both implement the same algorithm, the weights are
portable — but not without changes.

All OF3-specific branches in AF3 are gated on `global_config.of3_weights = True`,
set automatically via `--of3_weights` at the command line. Standard AF3 behavior
is unchanged when the flag is absent.

---

## Model Architecture Changes

These changes live in `src/alphafold3/model/`. They cannot be handled in the weight
converter alone because they reflect genuine differences in how OF3 and AF3 implement
or apply the same algorithm.

| File | Change | Why |
|---|---|---|
| `model_config.py` | Added `GlobalConfig.of3_weights: bool = False` | Master switch for all OF3 branches |
| `model.py` | Element one-hot index shifted by −1 | OF3 featurizes elements as `GetAtomicNumber() - 1` (0-indexed); AF3 uses `GetAtomicNum()` (1-indexed) |
| `network/atom_cross_attention.py` | `queries_single_cond *= queries_mask` | Masks padded query atoms from single conditioning, consistent with OF3's `atom_pair_mask` behavior |
| `network/atom_cross_attention.py` | `keys_ref_space_uid` sourced from `queries_ref_space_uid` | Matches OF3's uid lookup convention |
| `network/atom_cross_attention.py` | `offsets_valid &= keys_mask` | OF3 multiplies uid-match by `atom_pair_mask`; AF3 did not mask padded key atoms — `ref_space_uid=0` of padded atoms collided with token 0, distorting N-terminal backbone (CA–C was ~0.85 Å instead of 1.52 Å) |
| `network/evoformer.py` | Bond matrix symmetrized in `_embed_bonds` | OF3's `create_token_bonds` sets both `[i,j]` and `[j,i]` per bond; AF3 featurizes only one direction — ring bonds broken without this (ATP ribose C–C ~2.0 Å instead of ~1.5 Å) |
| `network/modules.py` | Pair bias axes swapped for column attention (`transpose=True`) | OF3 computes column attention by transposing the sequence input before `linear_z`, producing `Linear(z[k,q])` as the bias; AF3 computes `Linear(z[q,k])`. The weights were trained with OF3's convention so the bias must be swapped |
| `network/diffusion_transformer.py` | Per-block pair LayerNorm + linear projection branch | OF3's `AttentionPairBias` contains its own `layer_norm_z` + `linear_z` per block; AF3 originally used a single shared LN before the entire block stack |
| `network/diffusion_head.py` | Fourier `w`/`b` loaded as Haiku params instead of AF3's hardcoded constants | OF3 stores these as `register_buffer` (saved to `state_dict`); we convert them directly from the checkpoint so AF3's hardcoded JAX constants are never used |
| `network/noise_level_embeddings.py` | Optional `weight`/`bias` args added to `noise_embeddings()` | Allows passing the converted Fourier values from `diffusion_head` instead of falling back to AF3's hardcoded values |

---

## Weight Converter

`convert_of3_weights.py` + `src/alphafold3/model/of3_weight_converter.py`

Handles the systematic PyTorch → JAX/Haiku translation:

- **Parameter renames**: `weight` → `weights`, `LayerNorm.weight` → `scale`, `LayerNorm.bias` → `offset`
- **Linear transpositions**: PyTorch stores `(out, in)`; AF3 stores `(in, out)` for most projections
- **Attention head reshapes**: Q/K stored as `(H, D, in)` (transpose_weights=True); V as `(in, H, D)`
- **SwiGLU concatenation**: OF3 has separate `linear_a` (gate) and `linear_b`; AF3 concatenates them as `[gate | linear]` for a single fused projection
- **Layer stack aggregation**: OF3 has per-block parameter dicts (`blocks.0.*, blocks.1.*`); AF3's `hk.experimental.layer_stack` expects a leading stacked axis

**Critical bug fixed during conversion**: TriangularMultiplication incoming `a`/`b` projection weights were swapped.

---

## Implications for AlphaFold3 Reproduction

### 1. Atom cross-attention padding (N-terminal backbone distortion)

The `offsets_valid` bug in `atom_cross_attention.py` affects anyone using AF3 with padded atom sequences. AF3 pads atom arrays to a fixed size; padded slots are zero-filled, giving them `ref_space_uid=0`. Since the first real residue also has `ref_space_uid=0`, every padded key atom was treated as a valid neighbor of residue 0. This produced severe backbone geometry errors at the N-terminus in all test cases (CA–C ~0.85 Å, C–O ~0.95 Å vs. ideals of 1.52 Å and 1.23 Å).

**This is a bug in the released AF3 source**, not specific to OF3 weights. Fix:
```python
offsets_valid = (
    queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
) & keys_mask[:, None, :].astype(jnp.bool_)
```

### 2. Bond matrix asymmetry

AF3's featurizer provides bonds in one direction only (lower-index → higher-index from the CCD bond table). OF3 trained with a symmetric bond matrix. Whether this matters for AF3's own weights is unknown — but ring-topology ligands (saccharides, ATP ribose) are significantly affected: ATP ribose C–C bonds were ~2.0 Å (broken) vs. ~1.5 Å (correct) without symmetrization.

Anyone training AF3 from scratch should verify which convention is used in their featurizer and ensure consistency with the bond embedding weights.

### 3. Column attention pair bias (OF3 deviation from paper)

OF3's `TriangleAttention` (`starting=False`) transposes the pair representation **before** applying `linear_z`:

```python
# OF3 — column attention
x = x.transpose(-2, -3)           # z → z.T
triangle_bias[h, q, k] = Linear(z.T[q, k]) = Linear(z[k, q])
```

AF3's Algorithm 15 specifies the pair bias between query q and key k as `Linear(z[q, k])`. OF3 computes `Linear(z[k, q])` — the **transposed** pair bias. This is a deviation from the paper. The model trained with this convention and learned to compensate, so predictions are good in practice. Our AF3 fix reproduces the OF3 convention:

```python
if self.transpose and self.global_config.of3_weights:
    nonbatched_bias = jnp.swapaxes(nonbatched_bias, -1, -2)
```

Anyone reimplementing triangle attention ending node should use `Linear(z[q, k])` per the paper if training from scratch.

### 4. Diffusion transformer pair conditioning (per-block vs. shared)

OF3 applies a separate `LayerNorm + Linear` to the pair representation inside **each** diffusion transformer block. AF3's original code applied a single shared LayerNorm before all blocks. This is a genuine architectural difference that produces incompatible parameter layouts. If training AF3 from scratch, the choice must match whatever convention the weights were trained with.

### 5. Fourier noise embeddings

Both AF3 and OF3 initialize the random Fourier embedding weights with seed 42, but PyTorch's `torch.Generator().manual_seed(42)` + `normal_` + `uniform_` produces different values than JAX's equivalent. AF3 hardcodes its values as `_WEIGHT` and `_BIAS` constants; OF3 saves them as `register_buffer` in the checkpoint. When using OF3 weights, the checkpoint values must be used — not AF3's hardcoded constants.

---

## Summary: What the Released AF3 Code Gets Wrong

| Issue | Severity | Affects AF3 weights too? |
|---|---|---|
| `offsets_valid` missing `keys_mask` gate (N-terminal distortion) | High — ~0.85 Å CA–C at residue 0 | Likely yes |
| Bond matrix one-directional (ring ligand geometry) | Medium — ring bonds broken | Depends on training convention |
| Column attention pair bias transposed (OF3 deviation from paper) | Low — model compensates during training | No — OF3 weights only |
| Diffusion transformer pair conditioning per-block vs. shared | Architectural — incompatible param layouts | Only relevant if training from scratch |
| Fourier embedding constants differ between JAX and PyTorch RNG | Medium — wrong noise conditioning without fix | No — OF3 weights only |

---

# OpenBind-0 (OpenFold3 v0.5.0)

OpenBind-0 is OpenFold3's **v0.5.0** release. It is a different model from the preview-2 weights the sections above
describe, not a newer file of the same one: it has different forward conventions, and feeding its weights through
the preview-2 branches (or the reverse) does not error. Every shape agrees but one, and the fold simply comes out
slightly different. Both are Apache 2.0.

| | OpenFold3 preview-2 (`weight_family: openfold3`) | OpenBind-0 (`weight_family: openbind0`) |
|---|---|---|
| checkpoint | `s3://openfold/staging/of3-p2-155k.pt` | `https://openfold3-data.s3.amazonaws.com/openfold3-parameters/of3-ob-2025-06-30-174k.pt` (sha256 `bd43301c…8e29e4`) |
| column-attention pair bias | `Linear(z[k, q])` (transposed) | `Linear(z[q, k])`, as AF3's Algorithm 15 |
| diffusion transformer pair LayerNorm | one per block | **one** for the whole stack, as AF3 |
| diffusion single conditioning | 831 channels (see below) | **833** channels (OpenFold3's 32-class restype/profile blocks) |
| featuriser | AF3's, with implicit backbone links | OpenFold3's (see below) |

The family is a property of the converted weights: `of3_ported_weights.provenance.json` carries `weight_family`,
`alphafold3.model.of3_weight_family.detect_weight_family(model_dir)` reads it (or, without a stamp, tells them apart by
the one parameter only OpenBind-0 has, `transformer/pair_input_layer_norm`), and `run_alphafold.py` passes it to the
model config (`GlobalConfig.openbind0`) and the featuriser (`featurise_input(..., openbind0=True)`).
`openbind0` requires `of3_weights`; everything `of3_weights` switches on (symmetric token bonds, padded keys masked in
the atom-pair offsets, trained Fourier constants, 0-indexed elements) stays on.

## What is different in the graph (`GlobalConfig.openbind0`)

* `network/modules.py`: the transposed column pair bias is **not** applied.
* `network/diffusion_transformer.py`: the per-block pair LayerNorm path is **not** taken; the stock AF3 path runs, with the
  converted `transformer/pair_input_layer_norm` and a `pair_logits_projection` grouped by super-block.
* `network/diffusion_head.py`: the two zero columns OpenFold3 carries for unknown DNA are re-inserted before
  `single_cond_initial_norm`. A zero column is free before a Linear and not before a LayerNorm (it moves the mean and the
  variance and the width becomes 833), so the converter keeps the two rows (`_reorder_features_1d(pad_unk_dna=True)`).
* `network/template_modules.py`: a template residue typed UNK (a modified amino acid) keeps its geometry, read through ALA's
  dense slots.
* `network/confidence_head.py`: the per-atom heads (pLDDT, experimentally resolved) are indexed by the atom's *rank* among the
  token's real atoms, not by its dense slot. They agree until a slot is empty; a nucleotide's dropped 5' OP3 is slot 0, so by slot
  every atom of that nucleotide read its neighbour's logit (mean |ΔpLDDT| of 0.7-2.0 against the reference on DNA/RNA, 0.00-0.13
  once indexed by rank; the structure itself was unaffected).

## What the converter does differently

`of3_weight_converter.of3_release(state_dict)` tells the releases apart by tensor, never by file name. For OpenBind-0 the
transformer stack lives under `__layer_stack_with_per_layer` and the 24 per-block pair projections are regrouped by
super-block; the confidence head's `linear_i`/`linear_j` target-feature projections go to AF3's `right`/`left` (AF3 names
the pair-embedding sides the other way round in its two embedding sites; OpenFold3 uses `linear_i` = row `i` in both).
**The preview-2 conversion keeps its published mapping, which has these two swapped**: reconverting the preview-2 checkpoint
with this converter is byte-identical to the published file (sha256 `7ae00e49…`), which is why it was left alone;
`confidence_embed_row_col='row_is_right'` converts it the corrected way. The swap transposes the pair embedding the
confidence pairformer starts from and shows up in the asymmetric outputs (PAE, and pTM/ipTM/min-iPAE that ride on it).

## The featuriser (`model/pipeline/of3_features.py`, `of3_templates.py`)

The functions are taken from sokrypton/alphafold3 `main` (Apache 2.0), the reference LocalFold validates OpenBind-0
against, restricted to the conventions it declares for `openbind0` (see the module docstring for the list, with its
upstream reasons). They apply only with `openbind0=True`; preview-2 is featurised exactly as before.

One deliberate difference from the reference: this fork keeps explicit **polymer-polymer** crosslinks as token bonds for
AlphaFold 3 weights, and for preview-2. OpenFold3's `filter_fully_atomized_bonds` keeps a token bond only when **both** ends
are atomised, so for OpenBind-0 such a bond is dropped unless both residues are non-standard (`keep_fully_atomized_bonds`).

## How OpenBind-0 was validated

Against `sokrypton/alphafold3` `main --model openbind0` (the reference LocalFold's oracle dumps come from), on the same inputs,
at float32 with XLA's deterministic ops, on an H200:

* **Featuriser**: 36 jobs (AlphaFold 3's 13 example jobs and its kitchen-sink input, glycans, modified residues and bases, DNA/RNA,
  unknown residues, a 7,907-row alignment, two template cases, five RFD4 designs, padded and unpadded): **0 differing fields in 66
  feature arrays on every job**.
* **Converted weights**: identical to the reference converter's tensor for tensor except `ref_element` (the reference folds the
  element index shift into the weight rows, this fork shifts the index in the graph; the two select the same row).
* **Fold**: at full matmul precision the worst 8 cases agree to ≤0.003 Å RMSD (≤0.08 Å max atom), trunk contact probabilities to
  ≤1.2e-4. At JAX's default matmul precision 21 of 36 jobs agree to ≤0.02 Å; the multi-chain/ligand cases without an MSA drift by up to
  2.9 Å because the 200-step sampler amplifies a 1e-3 numerical difference, which is what the full-precision rerun shows.
* **Preview-2 is untouched**: reconversion is byte-identical; featurised batches are bitwise identical on the 29-job corpus; deterministic
  folds are bitwise identical in every output.

Not verified: against stock OpenFold3 v0.5.0 (PyTorch) directly (the reference was validated against it by its authors: 21 input types, 0.000 Å);
inputs with cyclic chains, SMILES-only ligands beyond the biotin example, `constraints`, ESM/language-model inputs; the reference does not
cover bonded inputs for OpenBind-0 beyond the example jobs here.
