# OpenBind-0 parity harness

Compares this fork's OpenBind-0 (`featurise_input(..., openbind0=True)` + `GlobalConfig.openbind0`) with the reference LocalFold validates
against, `sokrypton/alphafold3` `main --model openbind0`, on 31 edge-case jobs (AF3's 13 example jobs and kitchen-sink input, glycans, modified
residues/bases, DNA/RNA, unknown residues, a deep MSA, two template cases). Results of the 2026-11-13 run: 0 differing fields in 66 feature arrays on
every job (unpadded and padded); folds agree to <=0.003 Å at full matmul precision. Details: `OF3_AF3_PORTING_NOTES.md`.

```bash
LOCALFOLD=<sokrypton/localfold checkout> python build_corpus.py corpus        # the 31 jobs (needs the zstd CLI)
# reference env: a checkout of sokrypton/alphafold3, its own `build_data`, weights converted with its converters/openfold3.py
PYTHONPATH=<sokrypton checkout>/src python ref_run.py corpus/6mrr.json out/ref/6mrr.pkl --fold --model_dir <ref weights dir>
# this fork, weights from convert_of3_weights.py (provenance says openbind0)
python fork_run.py corpus/6mrr.json out/fork/6mrr.pkl --fold --model_dir <fork weights dir>
python cmp_batches.py out/ref out/fork            # featuriser: differing fields per job
python cmp_folds.py out/ref out/fork 6mrr ...     # fold: RMSD, pLDDT/PAE/contact differences, bond-length rms
```
Both sides must see the same settings: `XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true"` (XLA autotuning alone moves
coordinates ~0.1 Å between identical runs) and, to separate logic from numerics, `JAX_DEFAULT_MATMUL_PRECISION=highest` (at the default precision the
200-step sampler turns 1e-3 differences into 0.4-3 Å on multi-chain inputs without an MSA). `fork_run.py --batch-from <ref pkl>` feeds the reference's own
batch to this fork's graph; `--buckets 64,128,...` pads like the Space worker.
