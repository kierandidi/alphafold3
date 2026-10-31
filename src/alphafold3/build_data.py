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

"""Script for building intermediate data."""

import gzip
import hashlib
from importlib import resources
import json
import os
import pathlib
import shutil
import site
import tempfile
import urllib.request

import msgpack

import alphafold3.constants.converters

# `ccd_msgpack_gen` imports `alphafold3.cpp`, and importing that extension
# fails outright when components.cif is absent -- it looks the file up during
# module registration. Since fetching that file is this script's job, the
# import has to happen after the fetch, not at module scope.


# The wheel no longer bundles the ~497 MB Chemical Component Dictionary. It was
# 114 MB of a 132 MB wheel, which put the package over the file size PyPI
# allows by default, and libcifpp's CMake fetched it from wwPDB at build time
# with no hash check -- so the chemistry tables were silently whatever wwPDB
# happened to serve on the day the wheel was built. Fetch one known snapshot
# here instead and verify it, so two installs a year apart agree.
CCD_URL = (
    'https://github.com/kierandidi/alphafold3/releases/download/'
    'ccd-20261031/components.cif.gz'
)
CCD_SHA256 = '55a4a3833536556fad7d5e425dd08b8bae12142b3a3f57bfea17cc1546ab374c'


def _sha256(path: pathlib.Path) -> str:
  digest = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _libcifpp_data_dir() -> pathlib.Path:
  """Directory holding the libcifpp data files, which the wheel still ships."""
  for site_path in site.getsitepackages():
    candidate = pathlib.Path(site_path) / 'share/libcifpp'
    if candidate.is_dir():
      return candidate
  raise ValueError(
      'Could not locate share/libcifpp in any site-packages directory. If'
      ' libcifpp is installed somewhere non-standard, set LIBCIFPP_DATA_DIR to'
      ' the directory holding its data files.'
  )


def _download_ccd(dest: pathlib.Path, url: str, expected_sha256: str | None):
  """Fetches components.cif to ``dest``, atomically and verified."""
  print(f'Downloading the CCD from {url} (~114 MB)...', flush=True)
  dest.parent.mkdir(parents=True, exist_ok=True)
  # Stage in the destination directory so the final move cannot cross a
  # filesystem boundary, and so a failed run cannot leave a half-written
  # components.cif that later runs would treat as cached.
  with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
    staged = pathlib.Path(tmp) / 'components.cif'
    with urllib.request.urlopen(url) as response:
      if url.endswith('.gz'):
        with gzip.GzipFile(fileobj=response) as gz, open(staged, 'wb') as out:
          shutil.copyfileobj(gz, out, length=8 * 1024 * 1024)
      else:
        with open(staged, 'wb') as out:
          shutil.copyfileobj(response, out, length=8 * 1024 * 1024)

    actual = _sha256(staged)
    if expected_sha256 and actual != expected_sha256:
      raise ValueError(
          'The downloaded CCD is not the one this package pins.\n'
          f'  url      {url}\n'
          f'  expected {expected_sha256}\n'
          f'  actual   {actual}\n'
          'Chemistry tables built from a different CCD can change model'
          ' inputs, so this is refused rather than warned about. Set'
          ' AF3_CCD_SHA256 to the hash above to accept it deliberately, or'
          ' LIBCIFPP_DATA_DIR to use a local copy.'
      )
    staged.replace(dest)
  print(f'  CCD ready at {dest} (sha256 {actual})', flush=True)
  return actual


def build_data():
  """Builds intermediate data."""
  url = os.environ.get('AF3_CCD_URL', CCD_URL)
  # An explicit hash means "I know this is a different CCD and I mean it".
  expected = os.environ.get('AF3_CCD_SHA256', CCD_SHA256)

  libcifpp_data_dir = os.environ.get('LIBCIFPP_DATA_DIR')
  if libcifpp_data_dir:
    cif_path = pathlib.Path(libcifpp_data_dir) / 'components.cif'
    if not cif_path.exists():
      raise ValueError(
          f'LIBCIFPP_DATA_DIR is set to {libcifpp_data_dir!r} but it holds no'
          ' components.cif.'
      )
    ccd_sha256 = _sha256(cif_path)
  else:
    cif_path = _libcifpp_data_dir() / 'components.cif'
    if cif_path.exists():
      ccd_sha256 = _sha256(cif_path)
      print(f'Using the CCD already present at {cif_path}', flush=True)
    else:
      ccd_sha256 = _download_ccd(cif_path, url, expected)

  from alphafold3.constants.converters import ccd_msgpack_gen  # pylint: disable=g-import-not-at-top
  from alphafold3.constants.converters import chemical_component_sets_gen  # pylint: disable=g-import-not-at-top

  out_root = resources.files(alphafold3.constants.converters)
  ccd_msgpack_path = out_root.joinpath('ccd.msgpack')
  chemical_component_sets_msgpack_path = out_root.joinpath(
      'chemical_component_sets.msgpack'
  )
  ccd_codes_path = out_root.joinpath('ccd_codes.txt')

  ccd_msgpack_gen.main(['', str(cif_path), str(ccd_msgpack_path)])
  chemical_component_sets_gen.main(
      ['', str(chemical_component_sets_msgpack_path)]
  )

  # Fork addition, carried across the v3.0.5 pickle -> MessagePack switch:
  # a plain-text listing of CCD codes. Nothing in the package reads it, but
  # it is useful for inspection and external tooling, so it is kept.
  with open(str(ccd_msgpack_path), 'rb') as f:
    codes = sorted(msgpack.unpack(f).keys())
  pathlib.Path(str(ccd_codes_path)).write_text('\n'.join(codes))
  print(f'Wrote {len(codes):,} CCD codes to {ccd_codes_path}', flush=True)

  # Record which CCD these tables came from. Without this, two installs that
  # disagree give no way to tell whether the chemistry or the code differs.
  provenance_path = out_root.joinpath('ccd_provenance.json')
  pathlib.Path(str(provenance_path)).write_text(
      json.dumps(
          {
              'components_cif_sha256': ccd_sha256,
              'components_cif_path': str(cif_path),
              'num_components': len(codes),
          },
          indent=2,
      )
      + '\n'
  )
