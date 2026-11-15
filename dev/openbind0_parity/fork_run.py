"""FORK side (kierandidi/alphafold3 + OpenBind-0): featurise a job and/or fold it.

  PYTHONPATH=overlay-dev python fork_run.py JOB.json OUT.pkl [--fold] [--family openbind0|openfold3]
        [--batch-from REF.pkl]   fold the REFERENCE's batch (graph isolation) instead of the fork's own
        [--recycles 3] [--steps 200] [--seed 0] [--buckets ...] [--model_dir DIR]
"""
import argparse, datetime, os, pickle, sys
os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('job'); ap.add_argument('out')
ap.add_argument('--family', default='openbind0', choices=['openbind0', 'openfold3'])
ap.add_argument('--model_dir', required=True, help='directory with of3_ported_weights.bin.zst + provenance')
ap.add_argument('--fold', action='store_true')
ap.add_argument('--batch-from', default=None)
ap.add_argument('--recycles', type=int, default=3)
ap.add_argument('--steps', type=int, default=200)
ap.add_argument('--samples', type=int, default=1)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--buckets', default='')
ap.add_argument('--no-featurise-flag', action='store_true', help='featurise with of3_weights only (no openbind0 featuriser)')
a = ap.parse_args()
sys.argv = sys.argv[:1]
model_dir = a.model_dir

import haiku as hk, jax, jax.numpy as jnp
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.model import model as af3_model, params as afp, features
from alphafold3.model.components import utils

fold_input = folding_input.Input.from_json(open(a.job).read())
buckets = [int(b) for b in a.buckets.split(',')] if a.buckets else None
openbind0 = a.family == 'openbind0'
if a.batch_from:
    batch = pickle.load(open(a.batch_from, 'rb'))['batch']
else:
    ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    kw = dict(of3_weights=True)
    if openbind0 and not a.no_featurise_flag:
        kw['openbind0'] = True
    batch = featurisation.featurise_input(fold_input=fold_input, ccd=ccd, buckets=buckets, ref_max_modified_date=datetime.date(2021, 9, 30), **kw)[0]
raw = {k: np.asarray(v) for k, v in batch.items() if hasattr(v, 'shape') or isinstance(v, (list, tuple))}
result = {'batch': raw}
if a.fold:
    cfg = af3_model.Model.Config()
    cfg.global_config.flash_attention_implementation = 'xla'
    cfg.global_config.bfloat16 = 'none'
    cfg.global_config.of3_weights = True
    if hasattr(cfg.global_config, 'openbind0'):
        cfg.global_config.openbind0 = openbind0
    elif openbind0:
        raise SystemExit('this fork build has no openbind0 flag')
    cfg.num_recycles = a.recycles
    cfg.heads.diffusion.eval.steps = a.steps
    cfg.heads.diffusion.eval.num_samples = a.samples
    b = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
    @hk.transform
    def forward(b): return af3_model.Model(cfg)(b)
    out = jax.jit(forward.apply)(afp.get_model_haiku_params(model_dir=model_dir), jax.random.PRNGKey(a.seed), b)
    result['out'] = jax.tree_util.tree_map(lambda x: np.asarray(x), out)
pickle.dump(result, open(a.out, 'wb'), protocol=4)
print('wrote', a.out, 'n_keys', len(raw))
