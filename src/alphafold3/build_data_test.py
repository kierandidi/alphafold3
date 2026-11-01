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

"""Tests for the CCD fetch in build_data.

These cover the decision logic only -- which source is chosen, and what
happens when none of them serve the pinned file. The real CCD is ~497 MB, so
the fixtures here are a few bytes served over file:// URLs; the code under
test does not care what the payload is, only what it hashes to.
"""

import hashlib
import os
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
from alphafold3 import build_data


def _write(path: pathlib.Path, content: bytes) -> tuple[str, str]:
  """Writes ``content`` and returns its file:// URL and sha256."""
  path.write_bytes(content)
  return path.as_uri(), hashlib.sha256(content).hexdigest()


class DownloadCcdTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = pathlib.Path(self.create_tempdir().full_path)
    self.dest = self.tmp / 'out' / 'components.cif'
    # Each test decides this explicitly.
    os.environ.pop('AF3_CCD_REQUIRE_PINNED', None)

  def test_uses_the_source_that_matches_the_pin(self):
    stale_url, _ = _write(self.tmp / 'stale.cif', b'an older CCD')
    good_url, good_sha = _write(self.tmp / 'good.cif', b'the pinned CCD')

    actual = build_data._download_ccd(
        self.dest, (stale_url, good_url), good_sha
    )

    self.assertEqual(actual, good_sha)
    self.assertEqual(self.dest.read_bytes(), b'the pinned CCD')

  def test_prefers_the_pin_over_source_order(self):
    """A later source that matches beats an earlier one that does not."""
    first_url, _ = _write(self.tmp / 'first.cif', b'not the pinned CCD')
    second_url, second_sha = _write(self.tmp / 'second.cif', b'pinned')

    actual = build_data._download_ccd(
        self.dest, (first_url, second_url), second_sha
    )

    self.assertEqual(actual, second_sha)
    self.assertEqual(self.dest.read_bytes(), b'pinned')

  def test_unreachable_source_falls_through_to_the_next(self):
    missing = (self.tmp / 'nope.cif').as_uri()
    good_url, good_sha = _write(self.tmp / 'good.cif', b'the pinned CCD')

    actual = build_data._download_ccd(self.dest, (missing, good_url), good_sha)

    self.assertEqual(actual, good_sha)

  def test_drift_is_accepted_and_reported_not_raised(self):
    """wwPDB revising the CCD must not break installs."""
    drifted_url, drifted_sha = _write(self.tmp / 'new.cif', b'a newer CCD')
    pinned_sha = hashlib.sha256(b'the CCD we pinned').hexdigest()

    actual = build_data._download_ccd(self.dest, (drifted_url,), pinned_sha)

    # The hash returned is what was actually used, not what was hoped for --
    # this is the value that reaches ccd_provenance.json, so a drifted install
    # is diagnosable rather than merely suspected.
    self.assertEqual(actual, drifted_sha)
    self.assertNotEqual(actual, pinned_sha)
    self.assertEqual(self.dest.read_bytes(), b'a newer CCD')

  def test_drift_raises_when_the_pin_is_required(self):
    drifted_url, _ = _write(self.tmp / 'new.cif', b'a newer CCD')
    pinned_sha = hashlib.sha256(b'the CCD we pinned').hexdigest()
    os.environ['AF3_CCD_REQUIRE_PINNED'] = '1'
    self.addCleanup(os.environ.pop, 'AF3_CCD_REQUIRE_PINNED', None)

    with self.assertRaisesRegex(ValueError, 'AF3_CCD_REQUIRE_PINNED'):
      build_data._download_ccd(self.dest, (drifted_url,), pinned_sha)

    self.assertFalse(self.dest.exists())

  def test_require_pinned_is_off_when_set_to_zero(self):
    drifted_url, drifted_sha = _write(self.tmp / 'new.cif', b'a newer CCD')
    pinned_sha = hashlib.sha256(b'the CCD we pinned').hexdigest()
    os.environ['AF3_CCD_REQUIRE_PINNED'] = '0'
    self.addCleanup(os.environ.pop, 'AF3_CCD_REQUIRE_PINNED', None)

    actual = build_data._download_ccd(self.dest, (drifted_url,), pinned_sha)

    self.assertEqual(actual, drifted_sha)

  def test_no_source_reachable_raises(self):
    a = (self.tmp / 'a.cif').as_uri()
    b = (self.tmp / 'b.cif').as_uri()

    with self.assertRaisesRegex(ValueError, 'Could not obtain'):
      build_data._download_ccd(self.dest, (a, b), 'deadbeef')

  def test_leaves_nothing_behind_when_every_source_fails(self):
    """A failed run must not leave a partial file a later run would reuse."""
    missing = (self.tmp / 'nope.cif').as_uri()

    with self.assertRaises(ValueError):
      build_data._download_ccd(self.dest, (missing,), 'deadbeef')

    self.assertFalse(self.dest.exists())
    # The staging directory is cleaned up too.
    self.assertEqual(
        [p.name for p in self.dest.parent.iterdir()] if
        self.dest.parent.exists() else [],
        [],
    )

  def test_no_expected_hash_accepts_the_first_source(self):
    url, sha = _write(self.tmp / 'any.cif', b'whatever')

    actual = build_data._download_ccd(self.dest, (url,), None)

    self.assertEqual(actual, sha)


if __name__ == '__main__':
  absltest.main()
