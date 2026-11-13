# Copyright 2025 kierandidi. Licensed under the Apache License, Version 2.0.
"""Tests for the OpenFold3 weight family plumbing: release detection, the 833-wide conditioning, provenance."""

import json

import numpy as np
import pytest
import zstandard

from alphafold3.model import of3_weight_converter as conv
from alphafold3.model import of3_weight_family
from alphafold3.model import params as params_lib


def test_release_is_read_off_the_checkpoint_not_a_name():
  assert conv.of3_release({conv._SHARED_PAIR_NORM: 0}) == 'openbind0'
  assert conv.of3_release({'diffusion_module.diffusion_transformer.blocks.0.attention_pair_bias.layer_norm_z.weight': 0}) == 'openfold3'
  assert conv.of3_release({}) == 'openfold3'


def _features_1d(c_single=384):
  # row r holds the value r, so a permutation is visible
  return np.arange(c_single + 449, dtype=np.float32)


def test_features_1d_default_is_the_831_wide_preview2_layout():
  out = conv._reorder_features_1d(_features_1d())
  assert out.shape == (384 + 447,)
  # preview-2 must stay byte-identical: no padding rows
  np.testing.assert_array_equal(out, conv._reorder_features_1d(_features_1d(), pad_unk_dna=False))


def test_features_1d_openbind0_keeps_the_unknown_dna_rows_as_two_extra_channels():
  base = conv._reorder_features_1d(_features_1d())
  out = conv._reorder_features_1d(_features_1d(), pad_unk_dna=True)
  assert out.shape == (384 + 449,)
  # [single, aatype(31), UNK_DNA, profile(31), UNK_DNA, del_mean, atom_cross_att]
  single, a_end = 384, 384 + 31
  p_end = a_end + 1 + 31
  np.testing.assert_array_equal(out[:a_end], base[:a_end])
  assert out[a_end] == 384 + 384 + 30            # OF3 aatype class 30 (unknown DNA)
  np.testing.assert_array_equal(out[a_end + 1:p_end], base[a_end:a_end + 31])
  assert out[p_end] == 384 + 416 + 30            # ... and its profile column
  np.testing.assert_array_equal(out[p_end + 1:], base[a_end + 31:])
  # nothing is invented or lost: the extra rows are the only ones that are not in the 831-wide layout
  assert sorted(set(out) - set(base)) == [384 + 384 + 30, 384 + 416 + 30]


def _write_blob(path, records):
  with zstandard.open(path, 'wb') as f:
    for scope, name in records:
      f.write(params_lib.encode_record(scope, name, np.zeros((2,), np.float32)))


@pytest.mark.parametrize(
    ('records', 'family'),
    [
        ([('diffuser/~/diffusion_head/transformer/pair_input_layer_norm', 'scale')], 'openbind0'),
        ([('diffuser/~/diffusion_head/transformer/__layer_stack_no_per_layer/__layer_stack_no_per_layer', 'q')], 'openfold3'),
        ([('diffuser/evoformer', 'x')], None),
    ],
)
def test_family_from_the_parameter_file_alone(tmp_path, records, family):
  path = tmp_path / of3_weight_family.WEIGHTS_FILE
  _write_blob(path, records)
  assert of3_weight_family.detect_weight_family(tmp_path) == family


def test_provenance_wins_and_a_directory_without_weights_is_not_of3(tmp_path):
  assert of3_weight_family.detect_weight_family(tmp_path) is None
  (tmp_path / of3_weight_family.PROVENANCE_FILE).write_text(json.dumps({'weight_family': 'openbind0'}))
  assert of3_weight_family.detect_weight_family(tmp_path) == 'openbind0'


def test_provenance_round_trip(tmp_path):
  (tmp_path / 'of3_ported_weights.bin.zst').write_bytes(b'weights')
  checkpoint = tmp_path / 'ckpt.pt'
  checkpoint.write_bytes(b'checkpoint')
  conv.write_provenance(tmp_path, source_checkpoint=checkpoint, family='openbind0')
  record = json.loads((tmp_path / 'of3_ported_weights.provenance.json').read_text())
  assert record['weight_family'] == 'openbind0'
  assert record['converted_checkpoint']['bytes'] == len(b'weights')
  assert record['source_checkpoint']['bytes'] == len(b'checkpoint')
  assert of3_weight_family.detect_weight_family(tmp_path) == 'openbind0'
