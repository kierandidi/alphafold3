"""REFERENCE side (sokrypton/alphafold3 main, `--model openbind0`): featurise a job, optionally fold it.

  PYTHONPATH=overlay-ref:src/sokrypton-af3 python ref_run.py JOB.json OUT.pkl [--fold] [--recycles 3] [--steps 200] [--seed 0] [--buckets 128,256,...]

Mirrors dev/oracles/fold_check.py::_fold_setup (prepare_input -> featurise_input(**featurise_kwargs) -> model_features.apply)
but takes a fold-input JSON. The pickle holds the featurised batch (every array leaf, strings included) and, with
--fold, the model output, so the fork can be fed the SAME batch (graph isolation) or its own (featuriser parity).
"""
import argparse, datetime, os, pickle, sys
os.environ.setdefault('JAX_PLATFORMS', os.environ.get('JAX_PLATFORMS', 'cuda,cpu'))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('job'); ap.add_argument('out')
ap.add_argument('--model', default='openbind0')
ap.add_argument('--model_dir', required=True, help='directory with the reference converter\'s openbind0.bin.zst')
ap.add_argument('--fold', action='store_true')
ap.add_argument('--recycles', type=int, default=3)
ap.add_argument('--steps', type=int, default=200)
ap.add_argument('--samples', type=int, default=1)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--buckets', default='')
a = ap.parse_args()
sys.argv = sys.argv[:1]

import haiku as hk, jax, jax.numpy as jnp
from alphafold3.common import folding_input
from alphafold3.constants import decoded_ccd
from alphafold3.data import featurisation
from alphafold3.model import model as af3_model, model_registry
from alphafold3.model import params as afp
from alphafold3.model.components import utils
from alphafold3.model.pipeline import model_features

spec = model_registry.get(a.model)
fold_input = folding_input.Input.from_json(open(a.job).read())
fold_input = model_features.prepare_input(fold_input, spec)
ccd = decoded_ccd.get_ccd(user_ccd=fold_input.user_ccd)
buckets = [int(b) for b in a.buckets.split(',')] if a.buckets else None
featurise = lambda **kw: featurisation.featurise_input(
    fold_input=fold_input, ccd=ccd, buckets=buckets, ref_max_modified_date=datetime.date(2021, 9, 30), **model_registry.featurise_kwargs(spec), **kw)
batch = featurise()[0]
if spec.featurise:
    batch = model_features.apply(batch, spec, refeaturise=featurise, model_dir=a.model_dir, esm=None,
                                 has_msa=model_features.input_has_msa(fold_input), fold_input=fold_input,
                                 lm_pair=None, seed=a.seed, cyclic=None)
raw = {k: np.asarray(v) for k, v in batch.items() if hasattr(v, 'shape') or isinstance(v, (list, tuple))}
result = {'batch': raw}
if a.fold:
    cfg = af3_model.Model.Config()
    cfg.global_config.flash_attention_implementation = 'xla'
    cfg.global_config.bfloat16 = 'none'
    spec.configure(cfg)
    cfg.num_recycles = a.recycles
    cfg.heads.diffusion.eval.steps = a.steps
    cfg.heads.diffusion.eval.num_samples = a.samples
    b = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
    @hk.transform
    def forward(b): return af3_model.Model(cfg)(b)
    out = jax.jit(forward.apply)(afp.get_model_haiku_params(model_dir=a.model_dir), jax.random.PRNGKey(a.seed), b)
    result['out'] = jax.tree_util.tree_map(lambda x: np.asarray(x), out)
pickle.dump(result, open(a.out, 'wb'), protocol=4)
print('wrote', a.out, {k: v.shape for k, v in list(raw.items())[:3]}, 'n_keys', len(raw))
