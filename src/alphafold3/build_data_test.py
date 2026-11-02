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

The real CCD is ~497 MB, so these serve a few bytes over file:// URLs. The
code under test does not care what the payload is, only what it hashes to.
"""

import gzip
import hashlib
import pathlib

from absl.testing import absltest
from alphafold3 import build_data


def _write(path: pathlib.Path, content: bytes) -> tuple[str, str]:
  """Writes ``content`` and returns its file:// URL and sha256."""
  path.write_bytes(content)
  return path.as_uri(), hashlib.sha256(content).hexdigest()


class DownloadCcdTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = pathlib.Path(self.create_tempdir().full_path)
    self.dest = self.tmp / 'out' / 'components.cif'

  def test_accepts_bytes_matching_the_pin(self):
    url, sha = _write(self.tmp / 'ccd.cif', b'the pinned CCD')

    actual = build_data._download_ccd(self.dest, url, sha)

    self.assertEqual(actual, sha)
    self.assertEqual(self.dest.read_bytes(), b'the pinned CCD')

  def test_decompresses_a_gz_source(self):
    """The published asset is gzipped; the hash is of the decompressed file."""
    raw = b'the pinned CCD, compressed in transit'
    path = self.tmp / 'ccd.cif.gz'
    path.write_bytes(gzip.compress(raw))

    actual = build_data._download_ccd(
        self.dest, path.as_uri(), hashlib.sha256(raw).hexdigest()
    )

    self.assertEqual(actual, hashlib.sha256(raw).hexdigest())
    self.assertEqual(self.dest.read_bytes(), raw)

  def test_refuses_bytes_that_do_not_match_the_pin(self):
    """A revised CCD must not be substituted silently."""
    url, _ = _write(self.tmp / 'other.cif', b'a different CCD')
    pinned = hashlib.sha256(b'the pinned CCD').hexdigest()

    with self.assertRaisesRegex(ValueError, 'not the one this package pins'):
      build_data._download_ccd(self.dest, url, pinned)

    self.assertFalse(self.dest.exists())

  def test_unreachable_source_raises_and_says_what_to_do(self):
    missing = (self.tmp / 'nope.cif').as_uri()

    with self.assertRaisesRegex(ValueError, 'Could not download the CCD'):
      build_data._download_ccd(self.dest, missing, 'deadbeef')

    self.assertFalse(self.dest.exists())

  def test_no_expected_hash_accepts_whatever_is_served(self):
    """AF3_CCD_URL without AF3_CCD_SHA256: deliberately unverified."""
    url, sha = _write(self.tmp / 'any.cif', b'some other CCD entirely')

    actual = build_data._download_ccd(self.dest, url, None)

    self.assertEqual(actual, sha)
    self.assertEqual(self.dest.read_bytes(), b'some other CCD entirely')

  def test_leaves_nothing_behind_when_the_hash_is_wrong(self):
    """A rejected download must not be left where a later run reuses it."""
    url, _ = _write(self.tmp / 'other.cif', b'a different CCD')

    with self.assertRaises(ValueError):
      build_data._download_ccd(self.dest, url, 'deadbeef')

    self.assertFalse(self.dest.exists())
    self.assertEqual(sorted(p.name for p in self.dest.parent.iterdir()), [])

  def test_does_not_fall_back_to_wwpdb(self):
    """The single-source rule is the point of this module, so pin it down."""
    self.assertIsInstance(build_data.CCD_URL, str)
    self.assertNotIn('wwpdb', build_data.CCD_URL)
    self.assertFalse(hasattr(build_data, 'CCD_URLS'))


if __name__ == '__main__':
  absltest.main()
