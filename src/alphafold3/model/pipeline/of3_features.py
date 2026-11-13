# Copyright 2025 kierandidi. Licensed under the Apache License, Version 2.0.
#
# The functions below are taken, with minimal changes, from sokrypton/alphafold3
# (https://github.com/sokrypton/alphafold3, `main`, Apache-2.0; `model_features.py` and
# `chiral_features.py`), which is the reference LocalFold validates OpenBind-0 against.
# What is kept is exactly the featurisation conventions OpenBind-0 (OpenFold3 v0.5.0)
# declares there (`model_registry._FEATURISE['openbind0']`); everything for other models
# is left out. Each function carries its upstream docstring, which says why.
"""OpenBind-0's (OpenFold3 v0.5.0) featurisation conventions, applied to an AF3 batch.

AlphaFold 3's featuriser builds a different input from OpenFold3's in a handful of
places that no weight transform can absorb. The OpenFold3 *preview-2* port in this
fork never applied them; the OpenBind-0 family does, and they are all gated on
`featurise_input(..., openbind0=True)`:

  input side  (`prepare_input`, `conventions`, `pipeline_kwargs`)
    * an unknown amino acid is the CCD's UNK (N CA C O CB CG), not GLY;
    * OXT / OP3 are never emitted at a chain end, and a non-standard residue loses
      its CCD leaving atoms wherever it sits (`leaving_all`); a ligand is built
      whole, and the CCD leaving group at each declared bond is dropped;
    * the main MSA is each chain's first 16384 rows, deduplicated ignoring
      insertions, before its profile and deletion mean;
  batch side  (`apply`, in this order)
    * reference conformers centred per `ref_space_uid`;
    * token bonds across two consecutive atomised residues (never to a standard
      neighbour);
    * an atomised polymer token reads restype UNK in `aatype` alone, its MSA query
      row and profile keep the sequence letter;
    * terminal OXT/OP3 masked out of every atom gather and the layouts compacted;
    * padded atom-query slots stop being live queries and keys;
    * DNA and ligand tokens read gap in every MSA row with a zero profile; the MSA
      carries OpenFold3's row count; every modified residue is protein UNK;
    * an empty template slot, and any (slot, token) without template atoms, reads
      the GAP restype.
"""

import contextlib
import dataclasses
import functools

from alphafold3.constants import residue_names
from alphafold3.model.pipeline import of3_templates
import numpy as np

# Backbone atom pair that links residue i to residue i+1, per polymer class.
_LINK_ATOMS = (('C', 'N'), ("O3'", 'P'))


@functools.lru_cache(maxsize=1)
def _standard_residues():
  from alphafold3.model import features as af3_features
  return frozenset(af3_features._STANDARD_RESIDUES)

def _standard_residue_names():
  return _standard_residues()


def _centre_conformers(batch):
  """Subtract each conformer group's own mean from ref_pos -- see
  model_config.CENTRE_REF_CONFORMERS for who does this and why it matters.

  Grouped by `ref_space_uid`, which is the grouping those natives use, and
  MASKED: an unmasked mean would drag the centre toward padding atoms. Padding
  keeps zero afterwards, because the mask is what every consumer reads and a
  nonzero coordinate there would put real numbers where the graph expects none.
  """
  if 'ref_pos' not in batch:
    # Nothing to centre. A real featurised batch always carries the reference
    # structure, but a caller can hand `apply` a partial batch -- the unit
    # tests build atom-window-only batches -- and a featurisation step should
    # skip a convention it has no input for rather than raise.
    return
  pos = np.array(batch['ref_pos'], dtype=np.float32)
  mask = np.asarray(batch['ref_mask']) > 0
  uid = np.asarray(batch['ref_space_uid'])
  flat = pos.reshape(-1, 3)
  fm = mask.reshape(-1)
  fu = uid.reshape(-1).astype(np.int64)
  if not fu.size:
    return
  n = int(fu.max()) + 1
  w = fm.astype(np.float32)
  count = np.bincount(fu, weights=w, minlength=n)
  centre = np.stack(
      [np.bincount(fu, weights=w * flat[:, k], minlength=n) for k in range(3)],
      axis=-1) / np.maximum(count, 1.0)[:, None]
  flat = (flat - centre[fu]) * w[:, None]
  batch['ref_pos'] = flat.reshape(pos.shape)


def _drop_atoms_by_name(batch, names):
  """Mask out atoms another model's tokenizer does not create.

  AF3 gives the C-terminal residue an OXT; chai-1 does not -- its 6MRR run has
  573 atoms where ours has 574, and the extra one is exactly that OXT (verified
  against chai's own AtomRefPos/AtomNameOneHot dump). One atom in 574 sounds
  ignorable, but it lands in the last token's pooled atom representation and in
  the atom-pair window around it, and the trunk amplifies s_init error several
  fold per pass.

  The atom layout is COMPACTED (real atoms first, 18 full blocks of 32 here), so
  dropping an atom that is not last would renumber every atom after it. This
  only masks: the dense slot keeps its coordinates and simply stops being real,
  which is what every consumer already keys off.
  """
  chars = np.asarray(batch['ref_atom_name_chars'])
  decode = lambda row: ''.join(
      chr(int(c) + 32) if 0 <= c < 64 else '' for c in row).strip()
  ref_mask = np.array(batch['ref_mask'])
  n_atoms_per_token = ref_mask.shape[1]
  # STANDARD RESIDUES ONLY. Every vendor that drops these drops them from
  # standard residues and keeps them elsewhere, and says so: of3's
  # `remove_std_residue_terminal_atoms` is named for it ("terminal atoms can be
  # kept for any non-standard residues, as they are tokenized per-atom") and
  # rf3's filter carries `& ~is_atomized`.
  #
  # It is not pedantry, because these names mean different things in different
  # residues. **O3P is a sidechain atom of phosphoserine.** Dropping by name
  # alone took our SEP-20 ubiquitin to 604 atoms where of3's own featuriser has
  # 605 -- we removed the OXT correctly and then removed a phosphate oxygen that
  # native keeps. An ATOMISED residue is one token per atom, so it is identified
  # here by its token holding exactly one real atom.
  # `n_real > 1` alone identifies an atomised residue ONLY under AF3's
  # convention, where each of its atoms is its own token. boltz2 keeps a
  # modified residue as ONE token holding every atom
  # (`modified_as_one_token`), so that test called phosphoserine a standard
  # residue and dropped its O3P -- the hazard this comment block already
  # warned about, reintroduced through the other knob. `is_modified` is set by
  # `_mark_modified_residues`, which apply() runs BEFORE this, so the flag is
  # the reliable test and the atom count is the fallback for models that do not
  # set it.
  #
  # Measured on ACSEFGHIKLWY with SEP at 3, boltz2: dropping O3P leaves the
  # phosphate under-coordinated and collapses P-O1P and P-O2P to ~0.5 of ideal
  # (mean bond ratio 0.880, rms 0.395 A). Keeping it reads 0.987 at rms 0.080,
  # which is genuine Boltz-2's own 0.986-1.011 at 0.043-0.076.
  n_real = ref_mask.sum(axis=1)
  modified = np.asarray(batch.get('is_modified',
                                  np.zeros(ref_mask.shape[0], bool))
                        ).astype(bool).reshape(-1)
  drop = {(t, a) for t in range(ref_mask.shape[0])
          for a in range(n_atoms_per_token)
          if ref_mask[t, a] and decode(chars[t, a]) in names
          and n_real[t] > 1 and not modified[t]}
  return _mask_atoms(batch, drop)


def _mask_atoms(batch, drop):
  """Mask (token, atom slot) pairs out of every atom gather and compact the layouts; see _drop_atoms_by_name."""
  if not drop:
    return 0
  ref_mask = np.array(batch['ref_mask'])
  n_atoms_per_token = ref_mask.shape[1]
  flat = {t * n_atoms_per_token + a for t, a in drop}
  for t, a in drop:
    ref_mask[t, a] = False
  batch['ref_mask'] = ref_mask
  # The NAME goes too, so a dropped slot is indistinguishable from an empty one.
  # A consumer that counts a token's real atoms by name (the per-atom
  # confidence heads by rank, PER_ATOM_HEADS_BY_RANK) otherwise counted the
  # masked 5'-terminal OP3 that RF3's drop removes, and read every atom of that
  # nucleotide off by one (rna pLDDT corr 0.89 against stock RF3).
  if 'ref_atom_name_chars' in batch:
    chars = np.array(batch['ref_atom_name_chars'])
    for t, a in drop:
      chars[t, a] = 0
    batch['ref_atom_name_chars'] = chars
  if 'pred_dense_atom_mask' in batch:
    pdm = np.array(batch['pred_dense_atom_mask'])
    for t, a in drop:
      pdm[t, a] = False
    batch['pred_dense_atom_mask'] = pdm

  # queries carry the dropped atoms too; clear them before anything gathered
  # THROUGH the queries (the key window) is rebuilt from them.
  q_idxs = np.asarray(batch['token_atoms_to_queries:gather_idxs'])
  q_mask = np.array(batch['token_atoms_to_queries:gather_mask'])
  dead_queries = set()
  for r, c in zip(*np.nonzero(q_mask)):
    if int(q_idxs[r, c]) in flat:
      q_mask[r, c] = False
      dead_queries.add(r * q_idxs.shape[1] + c)
  batch['token_atoms_to_queries:gather_mask'] = q_mask

  for key, dead in (('queries_to_token_atoms', flat),
                    ('token_atoms_to_pseudo_beta', flat),
                    ('queries_to_keys', dead_queries)):
    idxs = np.asarray(batch[f'{key}:gather_idxs'])
    mask = np.array(batch[f'{key}:gather_mask'])
    if key == 'queries_to_token_atoms':
      # this one is indexed BY (token, atom), so drop by position
      for t, a in drop:
        mask[t, a] = False
    else:
      mask &= ~np.isin(idxs, list(dead))
    batch[f'{key}:gather_mask'] = mask

  _remove_dropped_atoms_from_layouts(batch, drop)
  return len(drop)


def _remove_dropped_atoms_from_layouts(batch, drop):
  '''REMOVE the dropped atoms from the layouts, the way the vendors do.

  Masking a finished batch is not the same thing as the vendor never emitting
  the atom, because a mask leaves a HOLE in the flat atom axis and two things
  are cut on that axis:

    * the 32-atom attention windows. Every block after the hole has a different
      membership than the vendor's compacted axis. This is the whole of the
      openfold3/openbind0 atom-encoder regression of 2026-09-12 (14 BAD cells
      that read corr 1.000000 two days earlier, with the gate headers reading
      `574 real atoms` before and `573` after). rf3, if2 and boltz2 took the
      same drop and stayed exact only because their native adapters consume our
      flat features hole and all, so it cancels on both sides.
    * the mmCIF gather, which reaches users. `flat_output_layout` and
      `empty_output_struc` are built during featurisation and the drop never
      touched them, so the model stopped predicting an atom the output still
      asked for -- and `predicted_structure_from_coords` writes a missing atom
      at (0, 0, 0) behind a `logging.warning`. openfold3 and boltz2 were handing
      back 574 output atoms for 573 predicted ones.

  `token_atoms_mask` is just `atom_name.astype(bool)`, so blanking the name
  compacts the flat axis by itself; the AtomCrossAtt gathers are then rebuilt
  from the compacted layout and the output layout is filtered to match.
  '''
  import dataclasses

  from alphafold3.model import features as _features

  lay = _features._unwrap(batch.get('token_atoms_layout'))
  if lay is None:
    return

  names = np.array(lay.atom_name, copy=True)
  gone = [(str(lay.chain_id[t, a]), int(lay.res_id[t, a]), str(names[t, a]))
          for t, a in drop if names[t, a]]
  for t, a in drop:
    names[t, a] = ''
  lay = dataclasses.replace(lay, atom_name=names)
  batch['token_atoms_layout'] = np.array(lay, object)

  # Rebuild the atom cross-attention gathers on the COMPACTED axis. The subset
  # sizes and the padded atom count are read back off the batch rather than
  # re-derived from config, so this cannot drift from the arrays it replaces.
  q_idxs = np.asarray(batch['token_atoms_to_queries:gather_idxs'])
  num_subsets, q_size = q_idxs.shape
  k_size = np.asarray(batch['queries_to_keys:gather_idxs']).shape[1]
  shapes = _features.PaddingShapes(
      num_tokens=lay.shape[0], msa_size=0, num_chains=0, num_templates=0,
      num_atoms=num_subsets * q_size)
  # `flat_atom_order` ranks the rows of the OLD flat axis, so removing rows
  # invalidates it. No model needs both today -- opendde is the only one with a
  # structural atom order and it KEEPS its terminal atoms (918/1204 against its
  # own featuriser) -- so this refuses the combination rather than silently
  # dropping a permutation that reorders every atom.
  if batch.get('flat_atom_order') is not None:
    raise NotImplementedError(
        'dropping atoms from a model that also permutes the flat atom axis '
        '(flat_atom_order) would need the permutation re-ranked over the '
        'remaining atoms; no model currently does both')
  aca = _features.AtomCrossAtt.compute_features(
      all_token_atoms_layout=lay,
      queries_subset_size=q_size,
      keys_subset_size=k_size,
      padding_shapes=shapes)
  batch.update(aca.as_data_dict())

  # The OUTPUT side: drop the same atoms from the flat output layout and from
  # the empty structure the coordinates are written into, so nothing is left
  # asking for an atom the model no longer predicts.
  fol = _features._unwrap(batch.get('flat_output_layout'))
  struc = _features._unwrap(batch.get('empty_output_struc'))
  if fol is None or not gone:
    return
  keep = np.ones(fol.shape[0], bool)
  gone_set = set(gone)
  for i in range(fol.shape[0]):
    if (str(fol.chain_id[i]), int(fol.res_id[i]),
        str(fol.atom_name[i])) in gone_set:
      keep[i] = False
  if keep.all():
    return
  batch['flat_output_layout'] = np.array(fol[keep], object)
  if struc is not None:
    # empty_output_struc is indexed 1:1 by flat_output_layout, so the SAME
    # boolean is applied to both. Filtering by atom NAME instead would also
    # remove atoms that were deliberately kept -- an OXT inside an atomised
    # residue, which is exactly the distinction the drop itself is careful
    # about.
    if struc.num_atoms != keep.shape[0]:
      raise ValueError(
          'empty_output_struc has %d atoms, flat_output_layout %d -- they are '
          'meant to be 1:1' % (struc.num_atoms, keep.shape[0]))
    batch['empty_output_struc'] = np.array(struc.filter(mask=keep), object)


def _mask_padded_atoms(batch, prefix=''):
  """A padded atom-query slot is padding, not an atom.

  AF3's AtomCrossAtt featuriser (DeepMind's as well) fills the query slots past the real atoms with the LAST padded
  dense slot and a gather mask of True, so in a bucket-padded batch every one of them is a live query and a live key --
  1152 of 1536 at a 52-token protein padded to 64. They carry zero reference features, but they still enter every real
  atom's attention. Here a query is live only where it gathers a real atom (ref_mask), and each key, token->query and
  token->key mask follows the query it indexes. An unpadded batch already reads exactly this, so it is untouched.
  """
  qk = prefix + 'token_atoms_to_queries:gather_'
  if qk + 'mask' not in batch or prefix + 'ref_mask' not in batch:
    return batch
  ref = np.asarray(batch[prefix + 'ref_mask']).astype(bool).reshape(-1)
  q_idx = np.asarray(batch[qk + 'idxs'])
  q_mask = np.asarray(batch[qk + 'mask']).astype(bool) & ref[np.clip(q_idx, 0, ref.size - 1)]
  batch[qk + 'mask'] = q_mask.astype(np.asarray(batch[qk + 'mask']).dtype)
  flat = q_mask.reshape(-1)
  tq = prefix + 'tokens_to_queries:gather_mask'
  if tq in batch:
    batch[tq] = (np.asarray(batch[tq]).astype(bool) & q_mask).astype(np.asarray(batch[tq]).dtype)
  for name in ('queries_to_keys', 'tokens_to_keys'):
    km = prefix + name + ':gather_mask'
    kidx = prefix + 'queries_to_keys:gather_idxs'
    if km in batch and kidx in batch:
      live = flat[np.clip(np.asarray(batch[kidx]), 0, flat.size - 1)]
      batch[km] = (np.asarray(batch[km]).astype(bool) & live).astype(np.asarray(batch[km]).dtype)
  return batch


def _dedupe_self_msa(batch, query_once=False):
  """ESMFold2's self-MSA is the query ONCE, not the query twice.

  AF3 concatenates a paired and an unpaired MSA, and each one begins with the
  query -- so a chain with no alignments at all arrives with `num_alignments 2`
  and two IDENTICAL unmasked rows. Native ESMFold2's own featuriser emits a
  depth-1 MSA for that input: its dump carries `feat.msa` at (1, 1, 68), and
  `esmfold2_reference.self_msa` is the query alone.

  Two identical rows are NOT free, even though an outer-product MEAN over
  duplicates is unchanged. Measured on 6MRR at one pass: the trunk's MSA
  injection reads std 619.2 against native's 593.9 (corr 0.998823), and feeding
  the reference the same duplicated row reproduces our number to 3.7e-07 --
  which is what identified the depth, rather than the encoder, as the
  difference. The pair-weighted averaging and the row transition are what make
  it depth-sensitive.

  IT FIRES ONLY ON A SELF-MSA -- one where EVERY live row is the query. With a
  real alignment the duplicate is NOT ours to drop: protenix builds a paired and
  an unpaired MSA exactly as AF3 does, so its own batch for a 4-row a3m carries
  FIVE rows (paired query, unpaired query, three homologs). The depth-1
  behaviour these vendors show is the NO-ALIGNMENT case, where their pipelines
  skip the MSA path and synthesise a single dummy row.

  Getting that wrong is expensive and quiet: dropping the duplicate alongside a
  real MSA cost 2.3 A on ubiquitin (ours 4.024 against native's 1.704, both
  stable across processes), and the earlier version of this docstring asserted
  the opposite -- "a real alignment whose first row happens to be the query is
  left alone" -- while the code dropped it anyway.
  """
  if 'msa' not in batch:
    return batch                      # a partial batch, as the unit tests build
  msa = np.asarray(batch['msa'])
  mask = np.asarray(batch['msa_mask'])
  live = np.flatnonzero(mask.any(-1))
  if live.size < 2:
    return batch
  a, b = live[0], live[1]
  if not np.array_equal(msa[a], msa[b]):
    return batch
  # query_once: ESMFold2 builds its MSA from the a3m alone, so beside a real alignment too the query appears once
  if not query_once and not all(np.array_equal(msa[a], msa[i]) for i in live[1:]):
    return batch                      # a real alignment: not a self-MSA
  m = mask.copy()
  m[b] = False
  batch['msa_mask'] = m
  if 'num_alignments' in batch:
    batch['num_alignments'] = np.asarray(
        np.asarray(batch['num_alignments']) - 1, dtype=np.asarray(
            batch['num_alignments']).dtype)
  return batch


def _dna_gap_msa(batch):
  """OpenFold3 adds query rows for protein and RNA only (colabfold_msa_server): a DNA chain reads gap in every MSA row
  and a zero profile and deletion mean; a ligand token's profile is zero too (ours: a gap one-hot)."""
  from alphafold3.constants import residue_names
  dna = np.asarray(batch['is_dna']).astype(bool).reshape(-1)
  prof = np.array(batch['profile'])
  prof[dna | np.asarray(batch['is_ligand']).astype(bool).reshape(-1)] = 0.0
  batch['profile'] = prof
  if not dna.any():
    return batch
  msa = np.array(batch['msa'])
  msa[:, dna] = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP).index(residue_names.GAP)
  batch['msa'] = msa
  for k in ('deletion_matrix', 'deletion_mean'):
    v = np.array(batch[k])
    v[..., dna] = 0
    batch[k] = v
  if not any(np.asarray(batch[k]).any() for k in ('is_protein', 'is_rna') if k in batch):
    _dedupe_self_msa(batch)           # no protein or RNA: OF3's MSA is the one dummy row
  return batch


def _has_alignment(a3m):
  """An a3m with a row beyond the query (a query-only a3m is no alignment: OF3 gets no file for it)."""
  return bool(a3m) and sum(line.startswith('>') for line in a3m.splitlines()) > 1


def _of3_paired_query_row(batch, fold_input):
  """A copy of the query row at row 1, gap on the chains with no paired MSA (see _of3_msa_rows)."""
  from alphafold3.constants import residue_names
  gap = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP).index(residue_names.GAP)
  asym = np.asarray(batch['asym_id']).reshape(-1)
  paired = [i + 1 for i, c in enumerate(fold_input.chains) if _has_alignment(getattr(c, 'paired_msa', None))]
  unpaired_tok = ~np.isin(asym, paired)
  for k in ('msa', 'msa_mask', 'deletion_matrix'):
    if k not in batch:
      continue
    v = np.asarray(batch[k])
    row = v[:1].copy()
    if k == 'msa':
      row[0, unpaired_tok] = gap
    elif k == 'deletion_matrix':
      row[0, unpaired_tok] = 0
    batch[k] = np.concatenate([v[:1], row, v[1:-1]], 0)


def _of3_msa_rows(batch, fold_input=None):
  """OpenFold3's MSA rows on chains it does not align, and its row count.

  Only protein and RNA chains get alignments there. A DNA or ligand token reads
  the GAP in every row and an all-zero profile, where we wrote the query base or a
  one-hot gap profile. The block always carries the query plus each chain's dummy
  row -- two rows for a heteromer with no paired MSA, where AF3's unpaired dedupe
  (features.py) leaves one -- and with no protein or RNA chain at all it is ONE
  all-gap row. Measured against stock OpenFold3 0.4.5 / v0.5.0
  (vendor_native_parity: dna, protein_dna, ligand_ccd, ion, heterodimer).

  With a precomputed paired MSA, OF3 stacks its query block, then the paired
  block AS GIVEN -- whose first row is the query again -- then the main MSA
  minus the paired rows (create_paired_from_precomputed / create_main): the query
  sits in rows 0 and 1, and a chain with no paired MSA reads the gap in row 1
  (its gap-filled placeholder). Measured: paired_msa, 74 rows to our 73.
  """
  if 'msa' not in batch:
    return batch
  if fold_input is not None and any(_has_alignment(getattr(c, 'paired_msa', None)) for c in fold_input.chains):
    _of3_paired_query_row(batch, fold_input)
  from alphafold3.constants import residue_names
  gap = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP).index(
      residue_names.GAP)
  msa = np.array(batch['msa'])
  mask = np.array(batch['msa_mask'])
  real = np.asarray(batch['seq_mask']).astype(bool).reshape(-1)
  aligned = ((np.asarray(batch['is_protein']) | np.asarray(batch['is_rna']))
             .astype(bool).reshape(-1) & real)
  unaligned = real & ~aligned
  live = np.flatnonzero(mask.any(-1))
  msa[np.ix_(live, np.flatnonzero(unaligned))] = gap
  keys = [k for k in ('msa', 'msa_mask', 'deletion_matrix') if k in batch]
  arrays = {'msa': msa, 'msa_mask': mask}
  arrays.update({k: np.asarray(batch[k]) for k in keys if k not in arrays})
  depth = msa.shape[0]
  if not aligned.any():
    rows = live[:1]                       # one all-gap row
  elif live.size == 1:                    # the dummy row AF3 deduped away
    rows = np.concatenate([live, live])
  else:
    rows = None
  if rows is not None:
    # The chosen rows FIRST, then padded back to the input depth with masked
    # zero rows. Returning only the live rows shrank the row axis below num_msa,
    # and the trunk's truncation then gathered past its end (see
    # featurization.truncate_msa_batch).
    def _repad(v):
      v = v[rows]
      out = np.zeros((depth,) + v.shape[1:], v.dtype)
      out[:len(rows)] = v
      return out
    arrays = {k: _repad(v) for k, v in arrays.items()}
  for k, v in arrays.items():
    batch[k] = v
  if 'num_alignments' in batch:
    batch['num_alignments'] = np.asarray(
        int(np.asarray(batch['msa_mask']).any(-1).sum()),
        dtype=np.asarray(batch['num_alignments']).dtype)
  # The profile is NOT rebuilt from these rows: OF3 profiles its own
  # deduplicated main MSA (of3_main_msa), and a duplicated query row leaves the
  # frequencies unchanged. Only the unaligned tokens' profile goes to zero.
  if 'profile' in batch:
    prof = np.array(batch['profile'])
    prof[unaligned] = 0.0
    batch['profile'] = prof
  return batch


def _modified_restype_protein_unknown(batch, fold_input, nucleic_unknown=False):
  """OpenFold3 types EVERY non-standard residue -- SEP, PSU, 5CM alike -- as the
  protein unknown (featurization/structure.py:85, residues.py:320), where we keep
  the parent or a nucleic unknown. Its MSA and profile already agree with ours,
  so only the restype moves. Stock-OF3 comparison: ptm / rna_mod / dna_mod."""
  asym = np.asarray(batch['asym_id']).reshape(-1)
  resi = np.asarray(batch['residue_index']).reshape(-1)
  aatype = np.array(batch['aatype'])
  for index, chain in enumerate(fold_input.chains):
    mods = (getattr(chain, 'ptms', None)
            or getattr(chain, 'modifications', None) or ())
    # helixfold3: a modified NUCLEOTIDE reads the nucleic unknown (30), which a DNA
    # token then carries as HelixFold3's DN (model.hf3_dn_columns)
    from alphafold3.common import folding_input as fi
    nucleic = nucleic_unknown and not isinstance(chain, fi.ProteinChain)
    for _code, residue in mods:
      aatype[(asym == index + 1) & (resi == int(residue))] = 30 if nucleic else 20   # protein UNK
  batch['aatype'] = aatype
  return batch


def _add_token_bonds(batch, pairs, order_code):
  """Token pairs, both directions, into the free (masked-off) rows of the ligand-ligand bond gather, so the batch keeps
  its shapes; grown if it has too few. order_code is the pairs' bond order."""
  key = 'tokens_to_ligand_ligand_bonds'
  idxs = np.array(batch[f'{key}:gather_idxs'])
  mask = np.array(batch[f'{key}:gather_mask'])
  order = batch.get('ligand_ligand_bond_order')
  present = {(int(a), int(b))
             for (a, b), ok in zip(idxs, mask.all(axis=1)) if ok}
  new = []
  for a, b in pairs:
    for pair in ((a, b), (b, a)):
      if pair not in present:
        present.add(pair)
        new.append(pair)
  if not new:
    return batch

  free = np.flatnonzero(~mask.all(axis=1))
  if free.size < len(new):
    # AF3 sizes this gather from the bond table, so a batch whose only bonds are
    # the ones we are adding arrives with no spare rows. Grow it rather than drop
    # bonds; the bond axis is not a bucketed dimension.
    grow = len(new) - free.size
    idxs = np.concatenate([idxs, np.zeros((grow, 2), idxs.dtype)])
    mask = np.concatenate([mask, np.zeros((grow, 2), mask.dtype)])
    if order is not None:
      order = np.concatenate([np.asarray(order),
                              np.zeros((grow,), np.asarray(order).dtype)])
    free = np.flatnonzero(~mask.all(axis=1))
  rows = free[:len(new)]
  idxs[rows] = np.asarray(new, idxs.dtype)
  mask[rows] = True
  batch[f'{key}:gather_idxs'] = idxs
  batch[f'{key}:gather_mask'] = mask
  if order is not None:
    order = np.array(order)
    order[rows] = order_code
    batch['ligand_ligand_bond_order'] = order
  return batch


def _atomized_backbone_bonds(batch, fold_input=None):
  """OpenFold3's token bonds at atomised residues: a backbone link between two consecutive atomised residues.

  AF3 extracts inter-residue bonds only where one side is a LIGAND chain, so a residue atomised INSIDE a polymer chain
  keeps its own CCD bonds and its declared covalent links but loses the backbone bond to each neighbour. OpenFold3's
  token_bonds keeps a bond only when BOTH atoms are atomised (filter_fully_atomized_bonds): a link counts between two
  consecutive non-standard residues and never to a standard neighbour (sokrypton's mode 'both_atomized').

  Bonds are written into the free (masked-off) rows of the ligand-ligand gather, so the batch keeps its shapes. Both
  directions are written: the matrix is symmetric, and AF3's featurisation gives only one direction.
  """
  layout = batch.get('token_atoms_layout')
  if layout is None:
    return batch
  layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout

  ref_mask = np.asarray(batch['ref_mask']).astype(bool)
  num_token, num_dense = ref_mask.shape[-2], ref_mask.shape[-1]

  standard = _standard_residue_names()
  # residues in token order: (chain, res_id) -> [res_name, {atom_name: token}]
  residues = {}
  for t in range(num_token):
    for s in range(num_dense):
      if not ref_mask[t, s]:
        continue
      name = str(layout.atom_name[t, s])
      if not name:
        continue
      rkey = (str(layout.chain_id[t, s]), int(layout.res_id[t, s]))
      entry = residues.setdefault(rkey, [str(layout.res_name[t, s]), {}])
      entry[1].setdefault(name, t)

  # a declared bond between two consecutive residues is their link (isoAsp joins through CG, not C)
  linked = {frozenset(((str(a[0]), int(a[1])), (str(b[0]), int(b[1]))))
            for a, b in (getattr(fold_input, 'bonded_atom_pairs', None) or ())}
  wanted = []
  keys = list(residues)
  for first, second in zip(keys, keys[1:]):
    if first[0] != second[0] or second[1] != first[1] + 1:
      continue                              # different chain, or a numbering gap
    (name_a, atoms_a), (name_b, atoms_b) = residues[first], residues[second]
    if name_a in standard and name_b in standard:
      continue                              # AF3 does not drop this one
    if frozenset((first, second)) in linked:
      continue
    if not (name_a not in standard and name_b not in standard):
      continue                              # of3 filter_fully_atomized_bonds: both partners atomised
    for tail, head in _LINK_ATOMS:
      if tail in atoms_a and head in atoms_b:
        wanted.append((atoms_a[tail], atoms_b[head]))
        break
  # a backbone link is a single bond
  _add_token_bonds(batch, wanted, 1)
  return batch


def _empty_template_gap(batch, gap_index=21, slots='first'):
  """A vendor's "no template" slot is a GAP template, not a zero template.

  Both pipelines pad the template axis to 4 slots and mask every atom, so the
  distogram, the unit vector and both masks come out zero either way. The
  RESTYPE one-hot does not: AF3's `empty_template_features` fills `aatype` with
  0, protenix's featuriser fills its one empty template with the GAP class and
  zero-pads the other three. Measured on native protenix1 at ubiquitin, its
  template_aatype is exactly that -- slot 0 all 31 (gap), slots 1-3 all 0.

  So the term is applied only when there is no real template anywhere, which is
  the case these vendors build that way; a batch carrying a real template is
  left alone.

  `slots` is the vendor's own padding, and the two differ:

    'first'  protenix -- ONE empty template with the gap restype, then three
             zero-padded slots ([31, 0, 0, 0] measured off its dump).
    'all'    opendde -- `make_dummy_feature` fills the WHOLE (4, N) block with
             31, commented "# gap" (data/utils.py), so every slot is a gap
             template.
    'atoms'  openfold3/openbind0 -- `create_template_feature_precursor_of3` starts every (slot, token) as GAP and
             writes only what a template covers, so an empty slot, and a chain without templates in a complex, read
             GAP even when another chain or slot has a real template; AF3 pads those with 0.

  `gap_index` is in OUR vocabulary: `template_modules._AF3_TO_OF3` maps 21 -> 31,
  which is the class the converted `a_proj` column expects.
  """
  aatype = batch.get('template_aatype')
  mask = batch.get('template_atom_mask')
  if aatype is None or mask is None:
    return
  aatype, mask = np.asarray(aatype), np.asarray(mask)
  if slots == 'atoms':
    if aatype.shape[0]:
      batch['template_aatype'] = np.where(mask.any(-1), aatype, gap_index).astype(aatype.dtype)
    return
  if not aatype.shape[0] or mask.any():
    return                                  # no slots, or a real template
  aatype = aatype.copy()
  if slots == 'all':
    aatype[:] = gap_index
  else:
    aatype[0] = gap_index
  batch['template_aatype'] = aatype


def apply_unknown_restype_on_atomized(batch, nucleic=False, nucleic_unknown=None, channels=('aatype', 'profile', 'msa'),
                                     msa_nucleic_only=False):
  '''an atomised polymer token is UNKNOWN to RF3, not its parent residue type.

  AlphaFold 3 atomises a modified residue but keeps the PARENT restype on every one
  of its tokens: D-aspartate reads back as ASP in `aatype`, in `profile` and in every
  `msa` row. atomworks does not -- it gives an atomised token the unknown restype, and
  a native RF3 batch for 5KX0 has 20 (UNK) in all three channels on all 93 atomised
  tokens, against our 3/14/6 (ASP/PRO/GLU).

  This is the D-amino-acid bug. 5KX0 is a de novo cyclic peptide with 11 D-residues,
  and the parent restype is the one feature that ASSERTS the wrong enantiomer: it
  names the L residue whose backbone the model has seen a million times, and it does
  so in the three channels that dominate s_inputs. The reference conformer and the
  chiral centres both say D -- correctly, and byte-for-byte as native builds them --
  but they are outvoted. Native builds 11/11 D-residues as D; with the parent restype
  we built 2/11.

  Protein tokens only. The evidence here is a protein batch, and AF3's 31-class vocab
  carries ONE nucleotide-unknown where atomworks carries two (UNKNOWN_RNA and
  UNKNOWN_DNA), so which of them a modified base should map to is not something this
  input can settle -- and mapping a modified base to its PARENT is what made three
  ports fold tRNA. Leaving nucleics alone keeps that.

  nucleic takes the atomised tokens of RNA and DNA chains too: ESMFold2's tokeniser gives every token of a modified
  nucleotide the PROTEIN unknown (esm prepare_input.tokenize_nucleotide, PROTEIN_UNK_RES_TYPE), and with the parent
  base our methylated DNA folded 18 A from native. RF3 does the same: native atomworks puts restype and profile 20 on
  all 160 atomised 5CM tokens of methylated_dna, and the parent base read pLDDT 84.2 against native's 78.5.

  nucleic_unknown gives a modified nucleotide's atomised tokens that class instead: chai-1's DX/RX, both our N (30), on
  which the forward tells DNA from RNA.

  channels: of3 sets UNK on restype alone and keeps the sequence letter in its msa and profile.
  msa_nucleic_only: esm reads a protein MSA from the a3m, whose query keeps the parent letter; only a nucleotide's row 0
  is its (UNK) restype (prepare_input.compute_msa_features).

  RF3 and ESMFold2 only: every other ported family trains on AlphaFold 3's parent-restype
  convention. Boltz-2 wants the unknown restype too, but on a residue it keeps as ONE
  token, and it needs the `is_modified` flag alongside -- see _mark_modified_residues.
  '''
  from alphafold3.constants import residue_names

  layout = batch.get('token_atoms_layout')
  if layout is None:
    return batch
  layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout

  standard = _standard_residues()
  mask = np.asarray(batch['ref_mask']).astype(bool)
  polymer = np.asarray(batch['is_protein']).astype(bool).reshape(-1)
  is_nucleic = np.asarray(batch['is_rna']).astype(bool).reshape(-1) | np.asarray(batch['is_dna']).astype(bool).reshape(-1)
  if nucleic or nucleic_unknown is not None:
    polymer = polymer | np.asarray(batch['is_rna']).astype(bool).reshape(-1) | np.asarray(
        batch['is_dna']).astype(bool).reshape(-1)
  unknown = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP).index(
      residue_names.UNK)

  num_token, num_dense = mask.shape[-2], mask.shape[-1]
  atomized = np.zeros((num_token,), bool)
  for t in range(num_token):
    if not polymer[t]:
      continue
    for s in range(num_dense):
      if mask[t, s]:
        atomized[t] = str(layout.res_name[t, s]) not in standard
        break
  if not atomized.any():
    return batch

  cls = np.full((num_token,), unknown)
  if nucleic_unknown is not None:
    cls[is_nucleic] = nucleic_unknown
  aatype = np.array(batch['aatype'])
  aatype[atomized] = cls[atomized]
  batch['aatype'] = aatype
  if 'profile' not in channels:
    return batch

  if msa_nucleic_only:
    atomized = atomized & is_nucleic
  profile = np.array(batch['profile'], copy=True)
  profile[atomized] = 0.0
  profile[atomized, cls[atomized]] = 1.0
  batch['profile'] = profile

  msa = batch.get('msa')
  if msa is not None:
    msa = np.array(msa, copy=True)
    msa[:, atomized] = cls[atomized]
    batch['msa'] = msa
  return batch


def apply_query_letter_on_atomized(batch, fold_input):
  """An atomised protein residue's MSA query row and profile take the INPUT sequence letter, as of3 builds its query row
  from the sequence: PXZ written as G reads G there, where AF3's query row reads the CCD parent (none: UNK)."""
  from alphafold3.common import folding_input
  from alphafold3.constants import residue_names

  layout = batch.get('token_atoms_layout')
  if layout is None or fold_input is None or batch.get('msa') is None:
    return batch
  layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout
  # the sequence as typed: ProteinChain.sequence already swaps a modified position for its CCD parent (X for PXZ)
  seqs = {c.id: getattr(c, '_sequence', c.sequence) for c in fold_input.chains if isinstance(c, folding_input.ProteinChain)}
  vocab = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP)
  standard = _standard_residues()
  mask = np.asarray(batch['ref_mask']).astype(bool)
  msa = np.array(batch['msa'], copy=True)
  profile = np.array(batch['profile'], copy=True)
  rows = np.asarray(batch['msa_mask']).astype(bool) if batch.get('msa_mask') is not None else np.ones(msa.shape, bool)
  for t in np.flatnonzero(np.asarray(batch['is_protein']).astype(bool).reshape(-1)):
    s = np.flatnonzero(mask[t])
    if not s.size or str(layout.res_name[t, s[0]]) in standard:
      continue
    seq = seqs.get(str(layout.chain_id[t, s[0]]))
    r = int(layout.res_id[t, s[0]]) - 1
    if seq is None or not 0 <= r < len(seq):
      continue
    three = residue_names.PROTEIN_COMMON_ONE_TO_THREE.get(seq[r], residue_names.UNK)
    cls = vocab.index(three) if three in vocab else vocab.index(residue_names.UNK)
    old = int(msa[0, t])
    if old == cls:
      continue
    # the query and its copies (AF3 adds the query again to an empty MSA) carry the old class
    hit = rows[:, t] & (msa[:, t] == old)
    n = max(int(rows[:, t].sum()), 1)
    profile[t, old] -= hit.sum() / n
    profile[t, cls] += hit.sum() / n
    msa[hit, t] = cls
  batch['msa'], batch['profile'] = msa, np.clip(profile, 0.0, 1.0)
  return batch



def keep_fully_atomized_bonds(bond_layout):
  """OpenFold3's `filter_fully_atomized_bonds` for explicit polymer-polymer bonds.

  OpenFold3's token_bonds keeps a bond only when BOTH partners are atomised (a ligand atom or a residue outside the 20
  amino acids / standard nucleotides, which is tokenised one token per atom). A crosslink between two standard residues,
  or between a standard residue and an atomised one, is therefore not a token bond there, although AlphaFold 3's weights
  are fed it by this fork's explicit-crosslink handling. Bonds to a ligand are not filtered here (they are not
  polymer-polymer bonds): OpenFold3 atomises the polymer residue bonded to a ligand, which the RFD4 oracle does ahead of
  featurisation by giving it a custom CCD code.
  """
  if not bond_layout.atom_name.size:
    return bond_layout
  standard = _standard_residues()
  atomized = np.array(
      [[name not in standard for name in row] for row in bond_layout.res_name])
  return bond_layout[atomized.all(axis=1)]


# ─── input side ──────────────────────────────────────────────────────────────

MAIN_MSA_MAX_ROWS = 16384


@contextlib.contextmanager
def unk_as_ccd():
  """An unknown amino acid builds from the CCD's UNK (N CA C O CB CG) instead of GLY, for the duration."""
  from alphafold3.structure import chemical_components as struc_chem_comps  # pylint: disable=g-import-not-at-top
  previous = struc_chem_comps.UNK_AS_GLY
  struc_chem_comps.UNK_AS_GLY = False
  try:
    yield
  finally:
    struc_chem_comps.UNK_AS_GLY = previous


def leaving_atoms(struct, fold_input, ccd):
  """Atom names to drop per (chain_id, res_id), as OpenFold3's featuriser removes them.

  1. the CCD leaving group at each declared bond (`declared_bond_leaving_atoms`), read before cleaning drops the
     polymer-polymer bonds;
  2. 'leaving_all': every non-standard polymer residue's CCD leaving heavy atoms wherever it sits -- ends included --
     or, when its CCD marks none, OXT (protein) / OP3 and O3P (nucleic): OpenFold3's MOLECULE_TYPE_TO_LEAVING_ATOMS.
     Standard residues are left alone here: their terminal OXT / OP3 are masked out of the finished batch by
     `drop_atoms` (which is why a modified residue keeps the atoms a standard one loses at the chain end).
  """
  from alphafold3.constants import mmcif_names  # pylint: disable=g-import-not-at-top
  from alphafold3.model import features as af3_features  # pylint: disable=g-import-not-at-top
  from alphafold3.model.pipeline import structure_cleaning  # pylint: disable=g-import-not-at-top

  res_name_of = dict(
      zip(zip(struct.chain_id.tolist(), struct.res_id.tolist()), struct.res_name.tolist()))
  leaving = dict(structure_cleaning.declared_bond_leaving_atoms(
      fold_input.bonded_atom_pairs, res_name_of, ccd))
  polymer = ~np.isin(struct.chain_type, list(mmcif_names.NON_POLYMER_CHAIN_TYPES))
  chains, res_ids, res_names = struct.chain_id[polymer], struct.res_id[polymer], struct.res_name[polymer]
  protein = struct.chain_type[polymer] == mmcif_names.PROTEIN_CHAIN
  for chain in set(chains.tolist()):
    sel = chains == chain
    for r, name, prot in set(zip(res_ids[sel].tolist(), res_names[sel].tolist(), protein[sel].tolist())):
      if name in af3_features._STANDARD_RESIDUES:  # pylint: disable=protected-access
        continue
      entry = ccd.get(name) or {}
      drop = {n for n, f, el in zip(entry.get('_chem_comp_atom.atom_id', ()),
                                    entry.get('_chem_comp_atom.pdbx_leaving_atom_flag', ()),
                                    entry.get('_chem_comp_atom.type_symbol', ())) if f == 'Y' and el != 'H'}
      if not drop:
        drop = {'OXT'} if prot else {'OP3', 'O3P'}
      leaving[(chain, r)] = leaving.get((chain, r), set()) | drop
  return leaving


def prepare_input(fold_input):
  """OpenFold3 reads each protein or RNA chain's main MSA as its first 16384 rows, deduplicated ignoring insertions,
  before its profile and deletion mean; AF3 keeps every row and profiles them all. The paired MSA stays as given."""
  import string  # pylint: disable=g-import-not-at-top
  from alphafold3.common import folding_input  # pylint: disable=g-import-not-at-top
  from alphafold3.data import parsers  # pylint: disable=g-import-not-at-top

  strip = str.maketrans('', '', string.ascii_lowercase)

  def dedupe(a3m):
    seqs, descs = parsers.parse_fasta(a3m)
    seen, lines = set(), []
    for s, d in list(zip(seqs, descs))[:MAIN_MSA_MAX_ROWS]:
      if s.translate(strip) not in seen:
        seen.add(s.translate(strip))
        lines += [f'>{d}', s]
    return '\n'.join(lines) + '\n'

  chains = []
  for c in fold_input.chains:
    if isinstance(c, folding_input.ProteinChain) and c.unpaired_msa:
      # the sequence as typed (.sequence is the CCD-parent letters)
      c = folding_input.ProteinChain(
          id=c.id, sequence=getattr(c, '_sequence', c.sequence), ptms=c.ptms, description=c.description,
          unpaired_msa=dedupe(c.unpaired_msa), paired_msa=c.paired_msa, templates=c.templates)
    elif isinstance(c, folding_input.RnaChain) and c.unpaired_msa:
      c = folding_input.RnaChain(
          id=c.id, sequence=getattr(c, '_sequence', c.sequence), modifications=c.modifications,
          description=c.description, unpaired_msa=dedupe(c.unpaired_msa))
    chains.append(c)
  fold_input = dataclasses.replace(fold_input, chains=chains)
  # OpenFold3's CIF-direct template reading: realigned to the template's whole canonical sequence, scored, ranked
  return of3_templates.apply(fold_input)


# ─── batch side ──────────────────────────────────────────────────────────────

def apply(batch, fold_input):
  """Apply OpenBind-0's conventions to a featurised batch, in place (sokrypton's `apply`, in its order)."""
  _centre_conformers(batch)
  _atomized_backbone_bonds(batch, fold_input)
  # restype UNK on an atomised polymer token, in `aatype` alone: the MSA and profile keep the sequence letter
  apply_unknown_restype_on_atomized(batch, nucleic=True, channels=('aatype',))
  apply_query_letter_on_atomized(batch, fold_input)
  _drop_atoms_by_name(batch, ('OXT', 'OP3', 'O3P'))
  # a padded atom-query slot is padding, not an atom
  _mask_padded_atoms(batch)
  if 'struct/token_atoms_to_queries:gather_mask' in batch:
    _mask_padded_atoms(batch, prefix='struct/')
  if 'msa' in batch:
    _dna_gap_msa(batch)
  _modified_restype_protein_unknown(batch, fold_input)
  _of3_msa_rows(batch, fold_input)
  # a modified template residue reads UNK on its own N/CA/C/CB (before the gap fill reads the atom masks)
  of3_templates.unknown_restype_templates(batch, fold_input)
  # every (slot, token) without template atoms reads GAP, whether the slot is empty or not
  _empty_template_gap(batch, slots='atoms')
  return batch
