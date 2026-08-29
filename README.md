![header](docs/header.jpg)

# AlphaFold 3

This package provides an implementation of the inference pipeline of
AlphaFold 3. See below for how to access the model parameters. You may only use
AlphaFold 3 model parameters if received directly from Google. Use is subject to
these
[terms of use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

Any publication that discloses findings arising from using this source code, the
model parameters or outputs produced by those should [cite](#citing-this-work)
the
[Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w)
paper.

Please also refer to the Supplementary Information for a detailed description of
the method.

AlphaFold 3 is also available at
[alphafoldserver.com](https://alphafoldserver.com) for non-commercial use,
though with a more limited set of ligands and covalent modifications.

If you have any questions, please contact the AlphaFold team at
[alphafold@google.com](mailto:alphafold@google.com).

## Obtaining Model Parameters

This repository contains all necessary code for AlphaFold 3 inference. You can
download the AlphaFold 3 model parameters from
https://storage.googleapis.com/alphafold3/af3.bin.zst. Use is subject to these
[terms of use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

## OpenFold3 Weights (Publicly Available Alternative)

[OpenFold3](https://github.com/aqlaboratory/openfold3), developed by the AlQuraishi Lab at Columbia University and the OpenFold Consortium, is an independent reproduction of AlphaFold 3 that has released model weights under the Apache 2.0 license. These weights can be used with this codebase as a freely available alternative to the Google DeepMind parameters.

**Step 1 — Download OpenFold3 weights:**

```bash
wget https://openfold.s3.amazonaws.com/staging/of3-p2-155k.pt
```

Weights are also hosted at [huggingface.co/OpenFold/OpenFold3](https://huggingface.co/OpenFold/OpenFold3).

**Step 2 — Convert to AlphaFold 3 format:**

A conversion script is included in this repository:

```bash
python convert_of3_weights.py \
  --of3_checkpoint ./of3-p2-155k.pt \
  --output_dir ./af3_of3_params/
```

This produces `af3_of3_params/of3_ported_weights.bin.zst` (~1.4 GB).

**Step 3 — Run inference:**

Pass `--of3_weights` to `run_alphafold.py` to enable the architectural adjustments needed for OpenFold3 parameters:

```bash
python run_alphafold.py \
  --json_path=fold_input.json \
  --model_dir=./af3_of3_params/ \
  --output_dir=./output/ \
  --of3_weights
```

The OpenFold3 weights are subject to the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0) and may be used for both academic and commercial purposes, without requiring a separate access request.

## Installation and Running Your First Prediction

See the [installation documentation](docs/installation.md).

### Cyclic-polymer runtime in this fork

The tested cyclic implementation is published on branch
`feat/cyclic-offset-runtime`. Pin commit
`a907519b8471d9b20e601f94c7f5ce4660986d0f` for reproducible builds. It changes
the AF3 relative-position encoding for an explicitly marked head-to-tail cyclic
protein chain while leaving ordinary linear inputs unchanged. AF3 and converted
OpenFold3 weights use the same engine.

Clone and build it with:

```bash
git clone --branch feat/cyclic-offset-runtime \
  https://github.com/kierandidi/alphafold3.git alphafold3-cyclic-runtime
cd alphafold3-cyclic-runtime
git checkout a907519b8471d9b20e601f94c7f5ce4660986d0f
DOCKER_BUILDKIT=1 docker build \
  --ulimit nofile=65536:65536 \
  --tag alphafold3:cyclic-a907519 \
  --file docker/Dockerfile .
```

The image is designed to run under an arbitrary allocation UID. This smoke test
checks both the compiled extension and the cyclic feature contract without model
parameters or a GPU:

```bash
docker run --rm --user 65534:65534 --entrypoint /usr/bin/env \
  alphafold3:cyclic-a907519 \
  /alphafold3_venv/bin/python -c \
  "import alphafold3.cpp; from alphafold3.model import features; assert {'cyclic_period', 'cyclic_position'} <= features.TokenFeatures.__dataclass_fields__.keys()"
```

Important: the model engine does not infer topology from an ordinary AF3 JSON
sequence. The caller must add two token features after normal AF3 featurisation:

* `cyclic_period`: the chain length on every token of a cyclic chain, otherwise
  zero;
* `cyclic_position`: the zero-based residue ordinal within that chain. This must
  not be derived from author residue IDs, which may be gapped.

The RFProteina branch `feat/cyclic-folding-runtime` supplies the tested adapter.
Its `run_alphafold_batch.py` accepts `cyclic_chain_ids` in each manifest entry;
the AF3 oracle adds that field automatically when the input structure contains
an explicit terminal carbonyl-C to N bond. To build the same OCI, SIF, and Enroot
bundle on another cluster, keep the two repositories as siblings and run:

```bash
git clone --branch feat/cyclic-folding-runtime \
  git@github.com:baker-laboratory/RFD4-Proteina-dev.git RFProteina-cyclic
cd RFProteina-cyclic
AF3_CYCLIC_SOURCE_REPO=../alphafold3-cyclic-runtime \
AF3_CYCLIC_OUTPUT_ROOT="$PWD/artifacts/af3-cyclic/a907519b8471" \
  bash scripts/build_cyclic_af3_runtime.sh
(cd artifacts/af3-cyclic/a907519b8471 && sha256sum --check SHA256SUMS)
```

The adapter currently enables cyclic offsets for protein chains only. RNA/DNA
and unmarked chains remain linear. A terminal bond is still passed through the
normal AF3 input path; the offset feature is additional model context, not a
replacement for chemical connectivity.

Disulfide and isopeptide macrocycles intentionally keep linear sequence
offsets: their N/C termini are still free, and multiple crosslinks cannot be
represented by one circular sequence coordinate. Supply each closure through
`bondedAtomPairs` (or mmCIF `struct_conn`). The featurizer maps ordinary
polymer bond atoms to their residue tokens, preserves actual atom tokens for
atomized OF3 residues, and feeds all such polymer--polymer crosslinks through
the existing bond embedding. This supports nested or multiple crosslinks
without falsely wrapping the whole chain. Explicit polymer crosslinks are not
discarded when the starting coordinates exceed a normal bond length; an open
design can therefore still request a closed prediction.

### Container images

Build the shared AF3/OpenFold3 Docker runtime from this checkout:

```bash
DOCKER_BUILDKIT=1 docker build \
  --file docker/Dockerfile \
  --tag af3-open:latest \
  .
```

Alternatively, build the standalone Apptainer image:

```bash
apptainer build af3-open.sif apptainer/af3-open.def
```

Both images contain the code and the one-time OpenFold3 conversion dependency,
but no model parameters. Mount either Google AF3 parameters or converted
OpenFold3 parameters when running inference.

Once you have installed AlphaFold 3, you can test your setup using e.g. the
following input JSON file named `fold_input.json`:

```json
{
  "name": "2PV7",
  "sequences": [
    {
      "protein": {
        "id": ["A", "B"],
        "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

You can then run AlphaFold 3 using the following command:

```
docker run -it \
    --volume $HOME/af_input:/root/af_input \
    --volume $HOME/af_output:/root/af_output \
    --volume <MODEL_PARAMETERS_DIR>:/root/models \
    --volume <DATABASES_DIR>:/root/public_databases \
    --gpus all \
    alphafold3 \
    python run_alphafold.py \
    --json_path=/root/af_input/fold_input.json \
    --model_dir=/root/models \
    --output_dir=/root/af_output
```

There are various flags that you can pass to the `run_alphafold.py` command, to
list them all run `python run_alphafold.py --help`. Two fundamental flags that
control which parts AlphaFold 3 will run are:

*   `--run_data_pipeline` (defaults to `true`): whether to run the data
    pipeline, i.e. genetic and template search. This part is CPU-only, time
    consuming and could be run on a machine without a GPU.
*   `--run_inference` (defaults to `true`): whether to run the inference. This
    part requires a GPU.

## AlphaFold 3 Input

See the [input documentation](docs/input.md).

## AlphaFold 3 Output

See the [output documentation](docs/output.md).

## Performance

See the [performance documentation](docs/performance.md).

## Known Issues

Known issues are documented in the
[known issues documentation](docs/known_issues.md).

Please
[create an issue](https://github.com/google-deepmind/alphafold3/issues/new/choose)
if it is not already listed in [Known Issues](docs/known_issues.md) or in the
[issues tracker](https://github.com/google-deepmind/alphafold3/issues).

## Citing This Work

Any publication that discloses findings arising from using this source code, the
model parameters or outputs produced by those should cite:

```bibtex
@article{Abramson2024,
  author  = {Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and Green, Tim and Pritzel, Alexander and Ronneberger, Olaf and Willmore, Lindsay and Ballard, Andrew J. and Bambrick, Joshua and Bodenstein, Sebastian W. and Evans, David A. and Hung, Chia-Chun and O’Neill, Michael and Reiman, David and Tunyasuvunakool, Kathryn and Wu, Zachary and Žemgulytė, Akvilė and Arvaniti, Eirini and Beattie, Charles and Bertolli, Ottavia and Bridgland, Alex and Cherepanov, Alexey and Congreve, Miles and Cowen-Rivers, Alexander I. and Cowie, Andrew and Figurnov, Michael and Fuchs, Fabian B. and Gladman, Hannah and Jain, Rishub and Khan, Yousuf A. and Low, Caroline M. R. and Perlin, Kuba and Potapenko, Anna and Savy, Pascal and Singh, Sukhdeep and Stecula, Adrian and Thillaisundaram, Ashok and Tong, Catherine and Yakneen, Sergei and Zhong, Ellen D. and Zielinski, Michal and Žídek, Augustin and Bapst, Victor and Kohli, Pushmeet and Jaderberg, Max and Hassabis, Demis and Jumper, John M.},
  journal = {Nature},
  title   = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  year    = {2024},
  volume  = {630},
  number  = {8016},
  pages   = {493–-500},
  doi     = {10.1038/s41586-024-07487-w}
}
```

## Acknowledgements

AlphaFold 3's release was made possible by the invaluable contributions of the
following people:

Andrew Cowie, Bella Hansen, Charlie Beattie, Chris Jones, Grace Margand,
Jacob Kelly, James Spencer, Josh Abramson, Kathryn Tunyasuvunakool, Kuba Perlin,
Lindsay Willmore, Max Bileschi, Molly Beck, Oleg Kovalevskiy,
Sebastian Bodenstein, Sukhdeep Singh, Tim Green, Toby Sargeant, Uchechi Okereke,
Yotam Doron, and Augustin Žídek (engineering lead).

We also extend our gratitude to our collaborators at Google and Isomorphic Labs.

AlphaFold 3 uses the following separate libraries and packages:

*   [abseil-cpp](https://github.com/abseil/abseil-cpp) and
    [abseil-py](https://github.com/abseil/abseil-py)
*   [Docker](https://www.docker.com)
*   [DSSP](https://github.com/PDB-REDO/dssp)
*   [HMMER Suite](https://github.com/EddyRivasLab/hmmer)
*   [Haiku](https://github.com/deepmind/dm-haiku)
*   [JAX](https://github.com/jax-ml/jax/)
*   [libcifpp](https://github.com/pdb-redo/libcifpp)
*   [NumPy](https://github.com/numpy/numpy)
*   [pybind11](https://github.com/pybind/pybind11) and
    [pybind11_abseil](https://github.com/pybind/pybind11_abseil)
*   [RDKit](https://github.com/rdkit/rdkit)
*   [Tokamax](https://github.com/openxla/tokamax)
*   [tqdm](https://github.com/tqdm/tqdm)

We thank all their contributors and maintainers!

## Get in Touch

If you have any questions not covered in this overview, please contact the
AlphaFold team at alphafold@google.com.

We would love to hear your feedback and understand how AlphaFold 3 has been
useful in your research. Share your stories with us at
[alphafold@google.com](mailto:alphafold@google.com).

## Licence and Disclaimer

This is not an officially supported Google product.

Copyright 2024 DeepMind Technologies Limited.

### AlphaFold 3 Source Code and Model Parameters

AlphaFold 3 source code is licensed under the Apache License, Version 2.0 (the
"License"); you may not use its source code except in compliance with the
License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

The AlphaFold 3 model parameters are made available under the
[AlphaFold 3 Model Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
(the "Terms"); you may not use these except in compliance with the Terms. You
may obtain a copy of the Terms at
[https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

Unless required by applicable law, AlphaFold 3 and its output are distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. You are solely responsible for determining the appropriateness of
using AlphaFold 3, or using or distributing its source code or output, and
assume any and all risks associated with such use or distribution and your
exercise of rights and obligations under the relevant terms. Output are
predictions with varying levels of confidence and should be interpreted
carefully. Use discretion before relying on, publishing, downloading or
otherwise using the AlphaFold 3 Assets.

AlphaFold 3 and its output are for theoretical modeling only. They are not
intended, validated, or approved for clinical use. You should not use the
AlphaFold 3 or its output for clinical purposes or rely on them for medical or
other professional advice. Any content regarding those topics is provided for
informational purposes only and is not a substitute for advice from a qualified
professional. See the relevant terms for the specific language governing
permissions and limitations under the terms.

### Third-party Software

Use of the third-party software, libraries or code referred to in the
[Acknowledgements](#acknowledgements) section above may be governed by separate
terms and conditions or license provisions. Your use of the third-party
software, libraries or code is subject to any such terms and you should check
that you can comply with any applicable restrictions or terms and conditions
before use.

### Mirrored and Reference Databases

The following databases have been: (1) mirrored by Google DeepMind; and (2) in
part, included with the inference code package for testing purposes, and are
available with reference to the following:

*   [BFD](https://bfd.mmseqs.com/) (modified), by Steinegger M. and Söding J.,
    modified by Google DeepMind, available under a
    [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/deed.en).
    See the Methods section of the
    [AlphaFold proteome paper](https://www.nature.com/articles/s41586-021-03828-1)
    for details.
*   [PDB](https://wwpdb.org) (unmodified), by H.M. Berman et al., available free
    of all copyright restrictions and made fully and freely available for both
    non-commercial and commercial use under
    [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
*   [MGnify: v2022\_05](https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2022_05/README.txt)
    (unmodified), by Mitchell AL et al., available free of all copyright
    restrictions and made fully and freely available for both non-commercial and
    commercial use under
    [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
*   [UniProt: 2021\_04](https://www.uniprot.org/) (unmodified), by The UniProt
    Consortium, available under a
    [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/deed.en).
*   [UniRef90: 2022\_05](https://www.uniprot.org/) (unmodified) by The UniProt
    Consortium, available under a
    [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/deed.en).
*   [NT: 2023\_02\_23](https://www.ncbi.nlm.nih.gov/nucleotide/) (modified) See
    the Supplementary Information of the
    [AlphaFold 3 paper](https://nature.com/articles/s41586-024-07487-w) for
    details.
*   [RFam: 14\_4](https://rfam.org/) (modified), by I. Kalvari et al., available
    free of all copyright restrictions and made fully and freely available for
    both non-commercial and commercial use under
    [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
    See the Supplementary Information of the
    [AlphaFold 3 paper](https://nature.com/articles/s41586-024-07487-w) for
    details.
*   [RNACentral: 21\_0](https://rnacentral.org/) (modified), by The RNAcentral
    Consortium available free of all copyright restrictions and made fully and
    freely available for both non-commercial and commercial use under
    [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
    See the Supplementary Information of the
    [AlphaFold 3 paper](https://nature.com/articles/s41586-024-07487-w) for
    details.
