import numpy as np
import pytest

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.constants import mmcif_names
from alphafold3.data import featurisation
from alphafold3.model import features
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.pipeline import pipeline


def _layout(atom_name, res_id, *, chain_type=None):
  shape = np.asarray(atom_name, dtype=object).shape
  return atom_layout.AtomLayout(
      atom_name=np.asarray(atom_name, dtype=object),
      res_id=np.broadcast_to(np.asarray(res_id, dtype=int), shape),
      chain_id=np.full(shape, 'A', dtype=object),
      chain_type=(
          None
          if chain_type is None
          else np.full(shape, chain_type, dtype=object)
      ),
  )


def _padding(num_tokens):
  return features.PaddingShapes(
      num_tokens=num_tokens,
      msa_size=1,
      num_chains=1,
      num_templates=0,
      num_atoms=num_tokens,
  )


@pytest.mark.parametrize('bond_atoms', [(('SG', 'SG')), (('NZ', 'CG'))])
def test_polymer_crosslink_maps_to_standard_residue_tokens(bond_atoms):
  all_tokens = _layout(['CA', 'CA'], [1, 8])
  crosslink = _layout(
      [bond_atoms], [1, 8], chain_type=mmcif_names.PROTEIN_CHAIN
  )
  token_crosslink = features._bond_layout_at_token_centres(
      all_tokens, crosslink
  )
  np.testing.assert_array_equal(token_crosslink.atom_name, [['CA', 'CA']])
  np.testing.assert_array_equal(token_crosslink.res_id, [[1, 8]])
  np.testing.assert_array_equal(token_crosslink.chain_id, [['A', 'A']])

  info = features.LigandLigandBondInfo.compute_features(
      all_tokens=all_tokens,
      bond_layout=None,
      polymer_polymer_bonds=crosslink,
      padding_shapes=_padding(2),
  )

  gather = info.tokens_to_ligand_ligand_bonds
  np.testing.assert_array_equal(gather.gather_idxs[0], [0, 1])
  np.testing.assert_array_equal(gather.gather_mask[0], [True, True])
  assert not gather.gather_mask[1:].any()


def test_polymer_crosslink_preserves_atomized_bond_tokens():
  all_tokens = _layout(['NZ', 'CG'], [1, 8])
  crosslink = _layout(
      [('NZ', 'CG')], [1, 8], chain_type=mmcif_names.PROTEIN_CHAIN
  )

  info = features.LigandLigandBondInfo.compute_features(
      all_tokens=all_tokens,
      bond_layout=None,
      polymer_polymer_bonds=crosslink,
      padding_shapes=_padding(2),
  )

  gather = info.tokens_to_ligand_ligand_bonds
  np.testing.assert_array_equal(gather.gather_idxs[0], [0, 1])
  np.testing.assert_array_equal(gather.gather_mask[0], [True, True])


def test_multiple_polymer_crosslinks_are_embedded_independently():
  all_tokens = _layout(['CA', 'CA', 'CA', 'CA'], [1, 3, 6, 8])
  crosslinks = _layout(
      [('SG', 'SG'), ('SG', 'SG')],
      [(1, 8), (3, 6)],
      chain_type=mmcif_names.PROTEIN_CHAIN,
  )

  info = features.LigandLigandBondInfo.compute_features(
      all_tokens=all_tokens,
      bond_layout=None,
      polymer_polymer_bonds=crosslinks,
      padding_shapes=_padding(4),
  )

  gather = info.tokens_to_ligand_ligand_bonds
  valid = np.flatnonzero(gather.gather_mask.all(axis=1))
  assert len(valid) == 2
  np.testing.assert_array_equal(gather.gather_idxs[valid], [[0, 3], [1, 2]])


def test_pipeline_excludes_only_linear_backbone_links_from_crosslinks():
  fold_input = folding_input.Input(
      name='linear_and_cyclic_links',
      chains=(
          folding_input.ProteinChain(
              id='A',
              sequence='CKAD',
              ptms=(),
              paired_msa='',
              unpaired_msa='>query\nCKAD\n',
              templates=(),
          ),
      ),
      rng_seeds=(1,),
      bonded_atom_pairs=(
          (('A', 10, 'C'), ('A', 20, 'N')),
          (('A', 40, 'C'), ('A', 10, 'N')),
          (('A', 20, 'NZ'), ('A', 40, 'CG')),
      ),
      residue_numbering={'A': (10, 20, 30, 40)},
  )
  structure = fold_input.to_structure(chemical_components.Ccd())
  bonds = pipeline.inter_chain_bonds.get_bond_layout(
      bond_threshold=np.inf,
      struct=structure,
      allowed_chain_types1=list(mmcif_names.POLYMER_CHAIN_TYPES),
      allowed_chain_types2=list(mmcif_names.POLYMER_CHAIN_TYPES),
      include_bond_types=(mmcif_names.COVALENT_BOND,),
      allow_multiple_bonds_per_atom=True,
      include_intra_chain_polymer=True,
  )

  filtered = pipeline._without_linear_polymer_backbone_bonds(
      structure, bonds, flatten_non_standard_residues=True
  )

  assert filtered.atom_name.tolist() == [['C', 'N'], ['NZ', 'CG']]
  assert filtered.res_id.tolist() == [[40, 10], [20, 40]]


def test_pipeline_keeps_atomized_modified_residue_backbone_link():
  fold_input = folding_input.Input(
      name='modified_backbone',
      chains=(
          folding_input.ProteinChain(
              id='A',
              sequence='SY',
              ptms=(('SEP', 1), ('PTR', 2)),
              paired_msa='',
              unpaired_msa='>query\nSY\n',
              templates=(),
          ),
      ),
      rng_seeds=(1,),
      bonded_atom_pairs=((('A', 1, 'C'), ('A', 2, 'N')),),
  )
  structure = fold_input.to_structure(chemical_components.Ccd())
  bonds = pipeline.inter_chain_bonds.get_bond_layout(
      bond_threshold=np.inf,
      struct=structure,
      allowed_chain_types1=list(mmcif_names.POLYMER_CHAIN_TYPES),
      allowed_chain_types2=list(mmcif_names.POLYMER_CHAIN_TYPES),
      include_bond_types=(mmcif_names.COVALENT_BOND,),
      allow_multiple_bonds_per_atom=True,
      include_intra_chain_polymer=True,
  )

  atomized = pipeline._without_linear_polymer_backbone_bonds(
      structure, bonds, flatten_non_standard_residues=True
  )
  residue_tokenized = pipeline._without_linear_polymer_backbone_bonds(
      structure, bonds, flatten_non_standard_residues=False
  )

  assert atomized.atom_name.tolist() == [['C', 'N']]
  assert not residue_tokenized.atom_name.size


def test_pipeline_keeps_dipeptide_head_to_tail_link():
  fold_input = folding_input.Input(
      name='cyclic_dipeptide',
      chains=(
          folding_input.ProteinChain(
              id='A',
              sequence='AG',
              ptms=(),
              paired_msa='',
              unpaired_msa='>query\nAG\n',
              templates=(),
          ),
      ),
      rng_seeds=(1,),
      bonded_atom_pairs=(
          (('A', 1, 'C'), ('A', 2, 'N')),
          (('A', 2, 'C'), ('A', 1, 'N')),
      ),
  )
  structure = fold_input.to_structure(chemical_components.Ccd())
  bonds = pipeline.inter_chain_bonds.get_bond_layout(
      bond_threshold=np.inf,
      struct=structure,
      allowed_chain_types1=list(mmcif_names.POLYMER_CHAIN_TYPES),
      allowed_chain_types2=list(mmcif_names.POLYMER_CHAIN_TYPES),
      include_bond_types=(mmcif_names.COVALENT_BOND,),
      allow_multiple_bonds_per_atom=True,
      include_intra_chain_polymer=True,
  )

  filtered = pipeline._without_linear_polymer_backbone_bonds(
      structure, bonds, flatten_non_standard_residues=True
  )

  assert filtered.atom_name.tolist() == [['C', 'N']]
  assert filtered.res_id.tolist() == [[2, 1]]


@pytest.mark.parametrize(
    ('sequence', 'bond_atoms'),
    [('CAAAAAC', ('SG', 'SG')), ('KAAAAAD', ('NZ', 'CG'))],
)
def test_full_pipeline_embeds_explicit_polymer_macrocycle(sequence, bond_atoms):
  chain = folding_input.ProteinChain(
      id='A',
      sequence=sequence,
      ptms=(),
      paired_msa='',
      unpaired_msa=f'>query\n{sequence}\n',
      templates=(),
  )
  fold_input = folding_input.Input(
      name='crosslinked_peptide',
      chains=(chain,),
      rng_seeds=(1,),
      bonded_atom_pairs=((
          ('A', 1, bond_atoms[0]),
          ('A', len(sequence), bond_atoms[1]),
      ),),
  )

  batch = featurisation.featurise_input(
      fold_input=fold_input,
      ccd=chemical_components.Ccd(),
      buckets=(16,),
      verbose=False,
  )[0]

  gather_mask = batch['tokens_to_ligand_ligand_bonds:gather_mask']
  gather_idxs = batch['tokens_to_ligand_ligand_bonds:gather_idxs']
  valid = np.flatnonzero(gather_mask.all(axis=1))
  assert len(valid) == 1
  np.testing.assert_array_equal(gather_idxs[valid[0]], [0, len(sequence) - 1])


@pytest.mark.parametrize(
    ('sequence', 'bond_atoms', 'bond_type'),
    [
        ('KAAAAAD', ('NZ', 'CG'), mmcif_names.COVALENT_BOND),
        ('CAAAAAC', ('SG', 'SG'), mmcif_names.DISULFIDE_BRIDGE),
    ],
)
def test_full_pipeline_keeps_explicit_bond_with_unclosed_input_coordinates(
    sequence, bond_atoms, bond_type
):
  chain = folding_input.ProteinChain(
      id='A',
      sequence=sequence,
      ptms=(),
      paired_msa='',
      unpaired_msa=f'>query\n{sequence}\n',
      templates=(),
  )
  fold_input = folding_input.Input(
      name='unclosed_isopeptide',
      chains=(chain,),
      rng_seeds=(1,),
      bonded_atom_pairs=((
          ('A', 1, bond_atoms[0]),
          ('A', len(sequence), bond_atoms[1]),
      ),),
  )
  ccd = chemical_components.Ccd()
  structure = fold_input.to_structure(ccd)
  structure = structure.copy_and_update(
      bonds=structure.bonds.copy_and_update(
          type=np.full(structure.bonds.size, bond_type, dtype=object)
      )
  )
  endpoint = np.flatnonzero(
      (structure.chain_id == 'A')
      & (structure.res_id == len(sequence))
      & (structure.atom_name == bond_atoms[1])
  )
  assert len(endpoint) == 1
  coords = structure.coords.copy()
  coords[endpoint[0]] += np.asarray([100.0, 0.0, 0.0])
  structure = structure.copy_and_update_coords(coords)

  batch = pipeline.WholePdbPipeline(
      config=pipeline.WholePdbPipeline.Config(buckets=[16])
  ).process_structure(
      structure,
      random_state=np.random.RandomState(1),
      ccd=ccd,
      unpaired_msa_by_chain_id={'A': f'>query\n{sequence}\n'},
      paired_msa_by_chain_id={'A': ''},
      templates_by_chain_id={'A': ()},
      random_seed=1,
  )

  gather = batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
  gather_mask = gather.gather_mask
  gather_idxs = gather.gather_idxs
  valid = np.flatnonzero(gather_mask.all(axis=1))
  assert len(valid) == 1
  np.testing.assert_array_equal(gather_idxs[valid[0]], [0, len(sequence) - 1])


def test_full_pipeline_still_removes_bad_polymer_ligand_bond():
  sequence = 'CAAAA'
  fold_input = folding_input.Input(
      name='bad_polymer_ligand_bond',
      chains=(
          folding_input.ProteinChain(
              id='A',
              sequence=sequence,
              ptms=(),
              paired_msa='',
              unpaired_msa=f'>query\n{sequence}\n',
              templates=(),
          ),
          folding_input.Ligand(id='L', ccd_ids=('ZN',)),
      ),
      rng_seeds=(1,),
      bonded_atom_pairs=((('A', 1, 'SG'), ('L', 1, 'ZN')),),
  )
  ccd = chemical_components.Ccd()
  structure = fold_input.to_structure(ccd)
  endpoint = np.flatnonzero(
      (structure.chain_id == 'L') & (structure.atom_name == 'ZN')
  )
  assert len(endpoint) == 1
  coords = structure.coords.copy()
  coords[endpoint[0]] += np.asarray([100.0, 0.0, 0.0])
  structure = structure.copy_and_update_coords(coords)

  batch = pipeline.WholePdbPipeline(
      config=pipeline.WholePdbPipeline.Config(buckets=[16])
  ).process_structure(
      structure,
      random_state=np.random.RandomState(1),
      ccd=ccd,
      unpaired_msa_by_chain_id={'A': f'>query\n{sequence}\n'},
      paired_msa_by_chain_id={'A': ''},
      templates_by_chain_id={'A': ()},
      random_seed=1,
  )

  gather_mask = (
      batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_mask
  )
  assert not gather_mask.all(axis=1).any()
