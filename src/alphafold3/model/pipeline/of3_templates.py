# Copyright 2025 kierandidi. Licensed under the Apache License, Version 2.0.
#
# Taken, with minimal changes, from sokrypton/alphafold3 (`model/pipeline/template_policy.py`, Apache-2.0), reduced
# to the one policy OpenBind-0 (OpenFold3 v0.5.0) declares there:
#   dict(modified='unk_restype', realign='whole_chain', hit_letter='parent', rank=True)
"""How OpenFold3 (CIF-direct mode) reads a template, as a pre-featurisation rewrite plus one batch-level rule.

AF3 takes a template as given: the mmCIF and the query->template index map. OpenFold3 instead
  * aligns the query to the template chain's WHOLE canonical (parent-letter) sequence itself (`realign`);
  * drops a template scoring seq_id * q_cov < 0.1 and stable-sorts the rest best first (`rank`);
  * types a modified template residue (SEP, MSE, ...) as UNK while keeping its own N / CA / C / CB, read by atom name
    (`unknown_restype_templates`, after featurisation); one missing any of the four reads GAP.
"""

import dataclasses
import re

import numpy as np

_STANDARD_AA = frozenset((
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL'))

# OpenFold3 TemplatePreprocessorSettings.cif_direct_min_score
RANK_MIN_SCORE = 0.1

_UNK_RESTYPE, _GAP_RESTYPE = 20, 21   # our protein UNK and gap indices
# dense slots every standard amino acid shares (protein_data_processing): N, CA, C, CB
_BACKBONE_DENSE = (0, 1, 2, 4)


def apply(fold_input):
  """`fold_input` with each protein chain's templates as OpenFold3 reads them (realigned, scored, ranked)."""
  from alphafold3.common import folding_input as fi  # pylint: disable=g-import-not-at-top
  chains = []
  for chain in fold_input.chains:
    if not isinstance(chain, fi.ProteinChain) or not chain.templates:
      chains.append(chain)
      continue
    query = getattr(chain, '_sequence', chain.sequence)
    kept = []
    for t in chain.templates:
      seqres, _ = template_seqres(t.mmcif, 'parent')
      qmap = global_align(query, seqres)
      tpl = fi.Template(mmcif=t.mmcif, query_to_template_map=qmap)
      # seq_id and q_cov are both over the query's length (every query residue is a non-gap column of a global alignment)
      ident = sum(query[i] == seqres[j] for i, j in qmap.items() if j < len(seqres))
      score = ident * len(qmap) / len(query) ** 2
      if score < RANK_MIN_SCORE:
        continue
      kept.append((score, tpl))
    kept = [tpl for _, tpl in sorted(kept, key=lambda x: -x[0])]   # stable: ties keep their order
    # (ProteinChain and Template are slotted classes, not dataclasses: rebuilt)
    chains.append(fi.ProteinChain(
        id=chain.id, sequence=query, ptms=chain.ptms, description=chain.description,
        unpaired_msa=chain.unpaired_msa, paired_msa=chain.paired_msa, templates=kept))
  return dataclasses.replace(fold_input, chains=chains)


def unknown_restype_templates(batch, fold_input):
  """modified='unk_restype', on the featurised batch: each template residue outside the 20 standard amino acids
  (raw name: SEP, MSE, UNK, ...) reads restype UNK, with its N, CA, C and CB taken by atom name from the template
  itself into the dense slots every standard residue keeps them in (template_modules reads an UNK template residue's
  geometry from those slots for the models that use this); one missing any of the four is dropped to a gap.
  `fold_input` is the one template_policy.apply returned (its query->template maps are the ones featurised)."""
  from alphafold3 import structure
  from alphafold3.constants import atom_types
  from alphafold3.common import folding_input as fi
  if 'template_aatype' not in batch:
    return batch
  aatype = np.array(batch['template_aatype'])
  pos = np.array(batch['template_atom_positions'])
  mask = np.array(batch['template_atom_mask'])
  asym = np.asarray(batch['asym_id']).reshape(-1)
  resi = np.asarray(batch['residue_index']).reshape(-1)
  atom37 = [atom_types.ATOM37_ORDER[n] for n in ('N', 'CA', 'C', 'CB')]
  for ci, chain in enumerate(fold_input.chains):
    if not isinstance(chain, fi.ProteinChain) or not chain.templates:
      continue
    for k, t in enumerate(chain.templates[:aatype.shape[0]]):
      struc = structure.from_mmcif(t.mmcif, fix_mse_residues=False, fix_arginines=False, include_bonds=False,
                                   include_water=False, include_other=True).filter_to_entity_type(protein=True)
      names = next(iter(struc.chain_res_name_sequence(include_missing_residues=True).values()))
      arrays = struc.to_res_arrays(include_missing_residues=True, atom_order=atom_types.ATOM37_ORDER)
      for q, j in t.query_to_template_map.items():
        if j >= len(names) or names[j] in _STANDARD_AA:
          continue
        tok = np.nonzero((asym == ci + 1) & (resi == q + 1))[0]
        have = arrays.atom_mask[j][atom37] > 0
        pos[k, tok] = 0
        mask[k, tok] = 0
        if have.all():
          aatype[k, tok] = _UNK_RESTYPE
          for slot, a in zip(_BACKBONE_DENSE, atom37):
            pos[k, tok, slot] = arrays.atom_positions[j][a]
            mask[k, tok, slot] = 1
        else:
          aatype[k, tok] = _GAP_RESTYPE
  batch['template_aatype'] = aatype.astype(np.asarray(batch['template_aatype']).dtype)
  batch['template_atom_positions'] = pos.astype(np.asarray(batch['template_atom_positions']).dtype)
  batch['template_atom_mask'] = mask.astype(np.asarray(batch['template_atom_mask']).dtype)
  return batch


def template_seqres(cif, modified_letter='X'):
  """(one-letter SEQRES of the template's protein chain, its oldest revision date). A modified residue is 'X'
  (protenix's parser) or its CCD parent's letter (IntelliFold's); MSE is M either way."""
  from alphafold3 import structure
  from alphafold3.constants import residue_names
  struc = structure.from_mmcif(cif, fix_mse_residues=False, fix_arginines=False, include_bonds=False,
                               include_water=False, include_other=True)
  prot = struc.filter_to_entity_type(protein=True)
  names = next(iter(prot.chain_res_name_sequence(include_missing_residues=True).values()))
  one = residue_names.PROTEIN_COMMON_THREE_TO_ONE
  if modified_letter == 'parent':
    parents = next(iter(prot.chain_res_name_sequence(
        include_missing_residues=True, fix_non_standard_polymer_res=True).values()))
    seq = ''.join(one.get(n, one.get(pn, 'X')) for n, pn in zip(names, parents))
  else:
    seq = ''.join('M' if n == 'MSE' else one.get(n, 'X') for n in names)
  dates = (re.findall(r'(\d{4}-\d{2}-\d{2})', cif.split('_pdbx_audit_revision_history', 1)[1][:4000])
           if '_pdbx_audit_revision_history' in cif else [])
  return seq, (min(dates) if dates else '')


def global_align(a, b):
  """Needleman-Wunsch (match 1, mismatch -1, gap -1) of a onto b -> {i_a: i_b}. Stands in for the alignment a vendor
  re-aligns a template with (kalign for protenix, boltz's own): on the near-identical sequences a template is, the
  optimal global alignment is the one they return."""
  n, m = len(a), len(b)
  score = np.zeros((n + 1, m + 1), np.int32)
  score[:, 0] = -np.arange(n + 1)
  score[0, :] = -np.arange(m + 1)
  for i in range(1, n + 1):
    for j in range(1, m + 1):
      score[i, j] = max(score[i - 1, j - 1] + (1 if a[i - 1] == b[j - 1] else -1),
                        score[i - 1, j] - 1, score[i, j - 1] - 1)
  i, j, out = n, m, {}
  while i > 0 and j > 0:
    if score[i, j] == score[i - 1, j - 1] + (1 if a[i - 1] == b[j - 1] else -1):
      out[i - 1] = j - 1
      i, j = i - 1, j - 1
    elif score[i, j] == score[i - 1, j] - 1:
      i -= 1
    else:
      j -= 1
  return out
