"""Build the edge-case corpus (AF3 fold-input JSONs) mirroring LocalFold's check-batch-fields targets."""
import json, glob, os, shutil
import sys
FX = os.path.join(os.environ.get('LOCALFOLD', 'localfold'), 'tools/fixtures')   # a checkout of sokrypton/localfold
OUT = sys.argv[1] if len(sys.argv) > 1 else 'corpus'
os.makedirs(OUT, exist_ok=True)
S6MRR = 'GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE'
def ccd(name):
    p = f'{FX}/ccd/{name}.cif'
    return open(p).read() if os.path.exists(p) else None
def job(name, seqs, bonds=None, user_ccd=(), seeds=(1,)):
    d = {'name': name, 'modelSeeds': list(seeds), 'sequences': seqs, 'dialect': 'alphafold3', 'version': 2}
    if bonds: d['bondedAtomPairs'] = bonds
    json.dump(d, open(f'{OUT}/{name}.json', 'w'), indent=1)
prot = lambda i, s, **k: {'protein': {'id': i, 'sequence': s, 'unpairedMsa': '', 'pairedMsa': '', 'templates': [], **k}}
dna  = lambda i, s, **k: {'dna': {'id': i, 'sequence': s, **k}}
rna  = lambda i, s, **k: {'rna': {'id': i, 'sequence': s, 'unpairedMsa': '', **k}}
lig  = lambda i, codes: {'ligand': {'id': i, 'ccdCodes': codes}}

job('6mrr', [prot('A', S6MRR)])
job('short12', [prot('A', 'GSMKQIEDKIEE')])                       # padded bucket: the 128-atom key window edge
job('gol-sep3', [prot('A', S6MRR, modifications=[{'ptmType': 'SEP', 'ptmPosition': 3}]), lig('B', ['GOL'])], user_ccd=['SEP','GOL'])
job('dna-unk', [dna('A', 'NACGNTACGTN')])
job('rna-unk', [rna('A', 'NACGNUACGUN')])
job('dna-5cm', [dna('A', 'GATCGATCGATCGATCGATC', modifications=[{'modificationType': '5CM', 'basePosition': 5}, {'modificationType': '5CM', 'basePosition': 9}])], user_ccd=['5CM'])
job('hy3', [prot('A', 'PGSPGSPGSP', modifications=[{'ptmType': 'HY3', 'ptmPosition': 4}])], user_ccd=['HY3'])
job('dna-5prime', [dna('A', 'GATTACA', modifications=[{'modificationType': '6OG', 'basePosition': 1}, {'modificationType': '6MA', 'basePosition': 2}])], user_ccd=['6OG','6MA'])
job('rna-5prime', [rna('A', 'GUACGUAC', modifications=[{'modificationType': '2MG', 'basePosition': 1}, {'modificationType': '5MC', 'basePosition': 4}])], user_ccd=['2MG','5MC'])
job('rna-mods', [rna('A', 'GGCAGGUAGCUGGCCUAGAGAUACU'.ljust(25,'A')[:25], modifications=[{'modificationType': 'OMG', 'basePosition': 4}, {'modificationType': 'PSU', 'basePosition': 13}, {'modificationType': '5MC', 'basePosition': 18}])], user_ccd=['OMG','PSU','5MC'])
# glycan: RNase B + the pentasaccharide, one ligand chain of five components, bonded to ND2 of Asn34
rnaseb = json.load(open(f'{FX}/af3-jobs/rnaseb_glycosylated.json'))
seq_b = rnaseb['sequences'][0]['protein']['sequence']
job('glycan', [prot('A', seq_b), lig('B', ['NAG','NAG','BMA','MAN','MAN'])],
    bonds=[[['A',34,'ND2'],['B',1,'C1']],[['B',1,'O4'],['B',2,'C1']],[['B',2,'O4'],['B',3,'C1']],[['B',3,'O3'],['B',4,'C1']],[['B',3,'O6'],['B',5,'C1']]],
    user_ccd=['NAG','BMA','MAN'])
# deep alignment: 5CAJ A with its 7907-row A3M as unpairedMsa (main-MSA cap 16384 / dedupe / profile)
a3m = open(f'{FX}/5caj-a.a3m').read()
q = a3m.split('\n')[1]
job('deep-msa', [prot('A', q, unpairedMsa=a3m, pairedMsa='')])
# rna alignment on its own (synthetic 17-nt query with 3 homologs)
rna_a3m = '>query\nGGCUAGCUAGCUAGCUA\n>h1\nGGCUAGCUAGCCAGCUA\n>h2\nGGCUUGCUAGCUAGCUA\n>h3\nGGCUAGC-AGCUAGCUA\n'
job('rna-msa', [rna('A', 'GGCUAGCUAGCUAGCUA', unpairedMsa=rna_a3m)])
# standalone glycans
gs = json.load(open(f'{FX}/glycan-standalone.json')); gs['name']='glycan-standalone'
for s in gs['sequences']:
    k = list(s)[0]
    if k=='protein': s[k].update(unpairedMsa='', pairedMsa='', templates=[])
json.dump(gs, open(f'{OUT}/glycan-standalone.json','w'), indent=1)
# AF3's own examples
for f in sorted(glob.glob(f'{FX}/af3-jobs/*.json')) + [f'{FX}/kitchen-sink-relabelled.json']:
    d = json.load(open(f)); name = 'job-' + os.path.basename(f)[:-5]
    d['name'] = name
    for s in d['sequences']:
        k = list(s)[0]; b = s[k]
        if k=='protein':
            b.setdefault('unpairedMsa',''); b.setdefault('pairedMsa',''); b.setdefault('templates',[])
        if k=='rna': b.setdefault('unpairedMsa','')
        b.pop('description', None)
    needed = sorted({m.get('ptmType') or m.get('modificationType') for s in d['sequences'] for b in s.values() for m in b.get('modifications',[])} |
                    {c for s in d['sequences'] for b in s.values() for c in b.get('ccdCodes',[])})
    d.pop('userCCD', None)
    json.dump(d, open(f'{OUT}/{name}.json','w'), indent=1)
print(len(glob.glob(f'{OUT}/*.json')), 'jobs')

# template cases: AF3's modified-residue (SEP) template job, and a 6MRR self-template from the RCSB fixture
import subprocess
d = json.load(open(f'{FX}/template-5k9p-sep.json')); d['name'] = 'template-5k9p-sep'
json.dump(d, open(f'{OUT}/template-5k9p-sep.json', 'w'))
cif = subprocess.run(['zstd', '-dc', f'{FX}/rcsb/6mrr.cif.zst'], capture_output=True, text=True, check=True).stdout
json.dump({'name': 'template-6mrr', 'modelSeeds': [1], 'dialect': 'alphafold3', 'version': 2, 'sequences': [{'protein': {
    'id': 'A', 'sequence': S6MRR, 'unpairedMsa': '', 'pairedMsa': '', 'templates': [
        {'mmcif': cif, 'queryIndices': list(range(len(S6MRR))), 'templateIndices': list(range(len(S6MRR)))}]}}]},
    open(f'{OUT}/template-6mrr.json', 'w'))
print(len(glob.glob(f'{OUT}/*.json')), 'jobs (with templates)')
