# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Tests for model-side featurization extensions."""

from absl.testing import absltest
from alphafold3.model import features
from alphafold3.model.network import featurization
import jax
import jax.numpy as jnp
import numpy as np


def _token_features(*, residue_index, asym_id, cyclic_period):
  residue_index = jnp.asarray(residue_index, dtype=jnp.int32)
  num_tokens = residue_index.shape[0]
  zeros = jnp.zeros(num_tokens, dtype=jnp.int32)
  ones = jnp.ones(num_tokens, dtype=jnp.int32)
  return features.TokenFeatures(
      residue_index=residue_index,
      token_index=jnp.arange(1, num_tokens + 1, dtype=jnp.int32),
      aatype=zeros,
      mask=jnp.ones(num_tokens, dtype=bool),
      seq_length=jnp.asarray(num_tokens, dtype=jnp.int32),
      asym_id=jnp.asarray(asym_id, dtype=jnp.int32),
      entity_id=ones,
      sym_id=ones,
      cyclic_period=jnp.asarray(cyclic_period, dtype=jnp.int32),
      is_protein=jnp.ones(num_tokens, dtype=bool),
      is_rna=jnp.zeros(num_tokens, dtype=bool),
      is_dna=jnp.zeros(num_tokens, dtype=bool),
      is_ligand=jnp.zeros(num_tokens, dtype=bool),
      is_nonstandard_polymer_chain=jnp.zeros(num_tokens, dtype=bool),
      is_water=jnp.zeros(num_tokens, dtype=bool),
  )


def _decode_residue_offset(relative_encoding, max_relative_idx):
  residue_encoding = relative_encoding[..., : 2 * max_relative_idx + 2]
  return np.asarray(jnp.argmax(residue_encoding, axis=-1)) - max_relative_idx


class CreateRelativeEncodingTest(absltest.TestCase):

  def test_cyclic_chain_uses_shortest_signed_offset(self):
    token_features = _token_features(
        residue_index=np.arange(6),
        asym_id=np.ones(6),
        cyclic_period=np.full(6, 6),
    )

    encoded = featurization.create_relative_encoding(
        token_features, max_relative_idx=32, max_relative_chain=2
    )

    expected = np.array([
        [0, -1, -2, -3, 2, 1],
        [1, 0, -1, -2, -3, 2],
        [2, 1, 0, -1, -2, -3],
        [3, 2, 1, 0, -1, -2],
        [-2, 3, 2, 1, 0, -1],
        [-1, -2, 3, 2, 1, 0],
    ])
    np.testing.assert_array_equal(
        _decode_residue_offset(encoded, max_relative_idx=32), expected
    )

  def test_linear_and_cross_chain_offsets_are_unchanged(self):
    token_features = _token_features(
        residue_index=[10, 11, 12, 1, 2],
        asym_id=[1, 1, 1, 2, 2],
        cyclic_period=[0, 0, 0, 2, 2],
    )

    encoded = featurization.create_relative_encoding(
        token_features, max_relative_idx=32, max_relative_chain=2
    )
    decoded = _decode_residue_offset(encoded, max_relative_idx=32)

    np.testing.assert_array_equal(decoded[:3, :3], [
        [0, -1, -2],
        [1, 0, -1],
        [2, 1, 0],
    ])
    # The final one-hot bin denotes residues from different chains.
    np.testing.assert_array_equal(decoded[:3, 3:], np.full((3, 2), 33))
    np.testing.assert_array_equal(decoded[3:, :3], np.full((2, 3), 33))

  def test_legacy_batch_defaults_to_linear_chain(self):
    batch = _token_features(
        residue_index=[1, 2, 3], asym_id=[1, 1, 1], cyclic_period=[0, 0, 0]
    ).as_data_dict()
    del batch['cyclic_period']

    restored = features.TokenFeatures.from_data_dict(batch)

    np.testing.assert_array_equal(restored.cyclic_period, np.zeros(3))

  def test_cyclic_period_fallback_is_jittable(self):
    batch = _token_features(
        residue_index=[1, 2, 3], asym_id=[1, 1, 1], cyclic_period=[0, 0, 0]
    ).as_data_dict()
    del batch['cyclic_period']

    cyclic_period = jax.jit(
        lambda traced_batch: features.TokenFeatures.from_data_dict(
            traced_batch
        ).cyclic_period
    )(batch)

    np.testing.assert_array_equal(cyclic_period, np.zeros(3))

  def test_present_cyclic_period_does_not_evaluate_numpy_fallback(self):
    batch = _token_features(
        residue_index=[1, 2, 3], asym_id=[1, 1, 1], cyclic_period=[3, 3, 3]
    ).as_data_dict()

    cyclic_period = jax.jit(
        lambda traced_batch: features.TokenFeatures.from_data_dict(
            traced_batch
        ).cyclic_period
    )(batch)

    np.testing.assert_array_equal(cyclic_period, np.full(3, 3))


if __name__ == '__main__':
  absltest.main()
