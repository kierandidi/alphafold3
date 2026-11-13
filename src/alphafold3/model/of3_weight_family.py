# Copyright 2025 kierandidi. Licensed under the Apache License, Version 2.0.
"""Which OpenFold3 release a converted weights directory holds.

Converted OpenFold3 weights are two different models behind one file name
(`of3_ported_weights.bin.zst`): the preview-2 release (`openfold3`) and
OpenBind-0 (`openbind0`, OpenFold3 v0.5.0). They need different forward
branches (`GlobalConfig.openbind0`), and feeding one release's weights through
the other's graph does not fail: every shape agrees except the 833-wide diffusion
single conditioning, and for the shape-compatible parts the fold simply comes
out slightly different. So the family travels with the weights and is read back
here rather than being remembered by the caller.

Order of evidence:
  1. `of3_ported_weights.provenance.json` -> `weight_family`;
  2. otherwise the parameter file itself, which is told apart by one parameter:
     OpenBind-0 has a shared `transformer/pair_input_layer_norm`, preview-2
     stores one per block under `__layer_stack_no_per_layer`.
"""

import json
import os

import zstandard

from alphafold3.model import params as params_lib

WEIGHTS_FILE = 'of3_ported_weights.bin.zst'
PROVENANCE_FILE = 'of3_ported_weights.provenance.json'

PREVIEW2 = 'openfold3'
OPENBIND0 = 'openbind0'
FAMILIES = (PREVIEW2, OPENBIND0)

_TRANSFORMER = 'diffuser/~/diffusion_head/transformer'


def family_from_params_file(path: str | os.PathLike[str]) -> str | None:
  """Reads the parameter names (not their values) to name the release.

  Streams the whole file, since records are length-prefixed inside a zstd
  frame: a few seconds, which is why provenance is preferred.
  """
  with zstandard.open(path, 'rb') as stream:
    for scope, name, _ in params_lib.read_records(stream):
      if scope == f'{_TRANSFORMER}/pair_input_layer_norm':
        return OPENBIND0
      if scope.startswith(f'{_TRANSFORMER}/__layer_stack_no_per_layer'):
        return PREVIEW2
  return None


def detect_weight_family(model_dir: str | os.PathLike[str]) -> str | None:
  """Returns 'openfold3', 'openbind0', or None when `model_dir` is not OF3."""
  provenance = os.path.join(model_dir, PROVENANCE_FILE)
  if os.path.exists(provenance):
    with open(provenance) as f:
      family = json.load(f).get('weight_family')
    if family in FAMILIES:
      return family
  weights = os.path.join(model_dir, WEIGHTS_FILE)
  if os.path.exists(weights):
    return family_from_params_file(weights)
  return None
