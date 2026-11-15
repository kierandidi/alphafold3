"""Table: fork vs reference end-to-end on every corpus job (featurisation + fold).

  python cmp_folds.py REFDIR FORKDIR job...
Columns: differing batch fields (fork's own featuriser vs the reference's), all-atom RMSD of the two predicted structures
(superposed), max raw atom displacement, mean |d pLDDT| over valid atoms, max |d PAE|, max |d contact_probs| (a trunk-only
output), and bond-length rms deviation from the reference conformer for each side (a chemistry check: every atom pair
within one reference-space closer than 1.9 A in ref_pos, predicted over conformer distance).
"""
import pickle, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from cmp_batches import cmp_pair

def kabsch(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b); d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((a @ r - b) ** 2, -1)))), a @ r, b

def bond_rms(batch, pos, mask):
    ref_pos = batch['ref_pos'].reshape(-1, 3); uid = batch['ref_space_uid'].reshape(-1); rm = batch['ref_mask'].reshape(-1).astype(bool)
    p = pos.reshape(-1, 3)
    idx = np.flatnonzero(rm & mask.reshape(-1))
    devs = []
    by = {}
    for i in idx: by.setdefault(int(uid[i]), []).append(i)
    for members in by.values():
        m = np.array(members)
        if len(m) < 2: continue
        d0 = np.linalg.norm(ref_pos[m][:, None] - ref_pos[m][None], axis=-1)
        d1 = np.linalg.norm(p[m][:, None] - p[m][None], axis=-1)
        i, j = np.where(np.triu((d0 < 1.9) & (d0 > 0.5), 1))
        devs.extend((d1[i, j] / d0[i, j] - 1).tolist())
    return float(np.sqrt(np.mean(np.square(devs)))) if devs else float('nan'), len(devs)

rd, fd = sys.argv[1], sys.argv[2]
print(f'{"job":34s} {"diff-fields":>11s} {"RMSD(A)":>8s} {"max|dx|":>8s} {"d-pLDDT":>8s} {"max dPAE":>9s} {"max dContact":>12s} {"bond-rms ref":>13s} {"bond-rms fork":>14s}')
for j in sys.argv[3:]:
    try:
        R = pickle.load(open(f'{rd}/{j}.pkl', 'rb')); F = pickle.load(open(f'{fd}/{j}.pkl', 'rb'))
    except FileNotFoundError:
        print(f'{j:34s} missing'); continue
    rows, only_r, only_f = cmp_pair(R['batch'], F['batch'])
    ro, fo = R['out'], F['out']
    n_tok = ro['diffusion_samples']['mask'].shape[1]      # real tokens (outputs are cropped to them)
    ra, fa = ro['diffusion_samples']['atom_positions'][0][:n_tok], fo['diffusion_samples']['atom_positions'][0][:n_tok]
    mk = ro['diffusion_samples']['mask'][0].astype(bool) & (np.abs(ra).sum(-1) > 0)
    rms, a, b = kabsch(ra[mk], fa[mk]); mx = float(np.abs(a - b).max())
    pl = float(np.abs(ro['predicted_lddt'][0][:n_tok][mk] - fo['predicted_lddt'][0][:n_tok][mk]).mean())
    n = min(ro['full_pae'].shape[-1], fo['full_pae'].shape[-1]); dpae = float(np.abs(ro['full_pae'][..., :n, :n] - fo['full_pae'][..., :n, :n]).max()) if ro['full_pae'].shape == fo['full_pae'].shape else float('nan')
    dct = float(np.abs(ro['distogram']['contact_probs'] - fo['distogram']['contact_probs']).max()) if ro['distogram']['contact_probs'].shape == fo['distogram']['contact_probs'].shape else float('nan')
    bb = {k: (v[:n_tok] if k in ('ref_pos','ref_space_uid','ref_mask') else v) for k, v in R['batch'].items() if hasattr(v, 'shape')}
    br, nb = bond_rms(bb, ra, mk); bf, _ = bond_rms(bb, fa, mk)
    ndiff = len(rows)
    print(f'{j:34s} {ndiff:11d} {rms:8.4f} {mx:8.3f} {pl:8.4f} {dpae:9.4f} {dct:12.2e} {br:13.4f} {bf:14.4f}   (n_bonds={nb})')
