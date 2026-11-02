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

"""Guards the dependency pins that decide what numbers come out.

Some of this package's dependencies can be upgraded freely and some cannot,
and the difference is not visible from the import graph. `rdkit` was once
relaxed from `==` to a range on the reasoning that nothing links against its
ABI -- true, and beside the point. It does the chemistry, so a newer version
makes different choices that reach model inputs: the same wheel on the same
node and seed folded a 76-token monomer 0.1680 A differently under rdkit
2026.9.1 than under 2026.3.3.

A test cannot notice that a published range has started resolving to a
version that moves coordinates -- by then the wheel is built. It can notice
the edit that makes it possible, which is where the mistake is cheap to undo.
"""

import pathlib
import re
import tomllib

from absl.testing import absltest
from absl.testing import parameterized


def _find_pyproject() -> pathlib.Path | None:
  for parent in pathlib.Path(__file__).resolve().parents:
    candidate = parent / 'pyproject.toml'
    if candidate.is_file():
      return candidate
  return None


# Changing a version of any of these changes results, so each must be pinned
# to one version rather than to a range. Adding to this list is cheap; taking
# something off it needs a measurement, not an argument.
NUMERICS_CRITICAL = (
    'jax',
    'jax-mps',
    'tokamax',
    'rdkit',
    'dm-haiku',
)


class DependencyPinsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    path = _find_pyproject()
    if path is None:
      self.skipTest('pyproject.toml not available (installed package)')
    self.requirements = tomllib.loads(path.read_text())['project'][
        'dependencies'
    ]

  @parameterized.parameters(*NUMERICS_CRITICAL)
  def test_is_pinned_to_an_exact_version(self, package: str):
    matching = [
        r
        for r in self.requirements
        if re.match(rf'^{re.escape(package)}(\[|==|>|<|;|$)', r)
    ]
    self.assertNotEmpty(
        matching, f'{package} is in NUMERICS_CRITICAL but is not a dependency'
    )
    for requirement in matching:
      specifier = requirement.split(';')[0]
      self.assertIn(
          '==',
          specifier,
          f'{package} must be pinned to one version, not a range: changing it'
          ' changes results. Got: {requirement!r}'.format(
              requirement=requirement
          ),
      )
      self.assertNotRegex(
          specifier,
          r'[><~!]=|[><]',
          f'{package} carries a range operator: {requirement!r}',
      )

  def test_numpy_is_bounded_but_not_pinned(self):
    """numpy is the counterexample, and the bound is still load-bearing.

    2.4.1 and 2.5.4 both reproduce the validated coordinates, so pinning it
    would be cargo-culting. Leaving it unbounded was the original bug: a fresh
    install resolved 2.5.x while the lockfile said 2.4.1.
    """
    matching = [r for r in self.requirements if r.startswith('numpy')]
    self.assertLen(matching, 1)
    self.assertIn('>=', matching[0])
    self.assertIn('<', matching[0])


if __name__ == '__main__':
  absltest.main()
