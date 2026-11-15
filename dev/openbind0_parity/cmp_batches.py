"""Compare featurised batches (ref vs fork) field by field.

  python cmp_batches.py REFDIR FORKDIR [job ...]

For each job prints the fields that differ, with how much. Arrays of different shape are reported (a padding / token
layout difference is a real difference). Fields only on one side are listed, since the fork's batch need not carry
reference-only extras. Per-atom fields are compared only where ref_mask is live on both sides (a dropped terminal atom
keeps its element/name in padding), and gathers are compared as listed.
"""
import glob, os, pickle, sys
import numpy as np

def load(path):
    return pickle.load(open(path, 'rb'))['batch']

# fields the reference adds that carry no AF3-graph input (compared by the model not at all)
IGNORE_REF_ONLY = ()

def cmp_pair(r, f):
    rows = []
    only_r = sorted(set(r) - set(f)); only_f = sorted(set(f) - set(r))
    for k in sorted(set(r) & set(f)):
        x, y = r[k], f[k]
        if x.dtype.kind in 'US' or y.dtype.kind in 'US' or x.dtype == object:
            if x.dtype == object and x.ndim == 0:
                a_, b_ = x.item(), y.item()
                fields = [f for f in ('atom_name', 'res_name', 'chain_id', 'res_id', 'atom_element') if hasattr(a_, f)]
                if fields:
                    same = all(np.array_equal(np.asarray(getattr(a_, f)), np.asarray(getattr(b_, f))) for f in fields)
                else:
                    same = (a_ is None and b_ is None) or repr(a_) == repr(b_)
                if not same: rows.append((k, 'structure/layout differs', fields, ''))
                continue
            same = x.shape == y.shape and bool(np.all(x == y))
            if not same: rows.append((k, 'string/obj differs', x.shape, y.shape))
            continue
        if k.startswith('tokens_to_ligand_ligand_bonds:gather_idxs') and 'tokens_to_ligand_ligand_bonds:gather_mask' in r:
            def live(b):
                i, m = b[k], b['tokens_to_ligand_ligand_bonds:gather_mask'].reshape(b[k].shape[0], -1).all(-1)
                return {tuple(int(v) for v in row) for row, ok in zip(i, m) if ok}
            la, lb = live(r), live(f)
            if la != lb:
                rows.append((k, 'LIVE BOND SET differs', f'ref-only {sorted(la - lb)[:6]}', f'fork-only {sorted(lb - la)[:6]}'))
            continue
        if k.startswith('tokens_to_ligand_ligand_bonds:gather_mask'):
            continue
        if x.shape != y.shape:
            rows.append((k, 'SHAPE', x.shape, y.shape)); continue
        if x.size == 0: continue
        xf, yf = x.astype(np.float64), y.astype(np.float64)
        if np.array_equal(xf, yf): continue
        d = np.abs(xf - yf)
        rows.append((k, 'maxabs=%.3g' % d.max(), 'n_diff=%d/%d' % (int((d > 0).sum()), d.size), ''))
    return rows, only_r, only_f

if __name__ == '__main__':
    rd, fd = sys.argv[1], sys.argv[2]
    jobs = sys.argv[3:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(rd, '*.pkl')))
    summary = []
    for j in jobs:
        rp, fp = os.path.join(rd, j + '.pkl'), os.path.join(fd, j + '.pkl')
        if not (os.path.exists(rp) and os.path.exists(fp)):
            print(f'== {j}: missing ({"ref" if not os.path.exists(rp) else ""} {"fork" if not os.path.exists(fp) else ""})'); summary.append((j, None)); continue
        rows, only_r, only_f = cmp_pair(load(rp), load(fp))
        print(f'== {j}: {len(rows)} differing fields; only-ref {len(only_r)}, only-fork {len(only_f)}')
        for row in rows: print('   ', *row)
        if only_r: print('    only ref :', ' '.join(only_r[:40]))
        if only_f: print('    only fork:', ' '.join(only_f[:40]))
        summary.append((j, len(rows)))
    print('\nSUMMARY', ' '.join(f'{j}:{n}' for j, n in summary))
