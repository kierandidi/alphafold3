# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Global config for the model."""

from collections.abc import Sequence
from typing import Literal, TypeAlias

from alphafold3.common import base_config
import tokamax

_Shape2DType: TypeAlias = tuple[int | None, int | None]


class GlobalConfig(base_config.BaseConfig):
  """Global configuration for the AlphaFold3 model."""

  bfloat16: Literal['all', 'none', 'intermediate'] = 'all'
  final_init: Literal['zeros', 'linear'] = 'zeros'
  pair_attention_chunk_size: Sequence[_Shape2DType] = ((1536, 128), (None, 32))
  pair_transition_shard_spec: Sequence[_Shape2DType] = (
      (2048, None),
      (None, 1024),
  )
  # Note: flash_attention_implementation = 'xla' means no flash attention.
  flash_attention_implementation: tokamax.DotProductAttentionImplementation = (
      'triton'
  )
  # When True, enables OpenFold3 weight compatibility mode.
  # Set to False (default) when using original AlphaFold3 weights.
  of3_weights: bool = False
  # With `of3_weights`, selects OpenBind-0 (OpenFold3 v0.5.0) instead of the
  # OpenFold3 preview-2 weights. The two releases are different models with
  # different forward conventions; everything `of3_weights` switches on stays on
  # (symmetric token bonds, padded keys masked in the atom-pair offsets, trained
  # Fourier constants, 0-indexed elements) and three things change:
  #   * the column-attention pair bias is NOT computed from the transposed pair
  #     (preview-2 does; v0.5.0 follows AlphaFold 3's Algorithm 15);
  #   * the diffusion transformer LayerNorms the pair conditioning ONCE for the
  #     whole stack (AlphaFold 3's arrangement), not once per block;
  #   * the diffusion single conditioning is normalised over the 833 channels of
  #     OpenFold3's 32-class restype/profile blocks, not AlphaFold 3's 831.
  # Converted weights say which family they are (`of3_ported_weights.provenance.json`
  # `weight_family`); a mismatch is silent -- every shape agrees except the
  # 833-wide conditioning -- so the runner must pass the right one.
  openbind0: bool = False
