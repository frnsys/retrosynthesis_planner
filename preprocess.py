import re
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem, RDLogger
from collections import defaultdict
from hashlib import md5

# Silence logs
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

TOKEN_RE = re.compile(r'(\[[^\]]+]|[0-9]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|−|\+|\\\\\/|:|~|@|\?|>|\*|\$|\%[0–9]{2}|[0–9])')
reagents = {}

def process(smarts):
    rxn = AllChem.ReactionFromSmarts(smarts)
    prods = rxn.GetProducts()
    if len(prods) > 1:
        return None

    rxn.Initialize()
    try:
        reactants = list(zip(rxn.GetReactants(), rxn.GetReactingAtoms()))
    except ValueError:
        # Likely that initialization failed
        # print('Failed to initialize')
        return None

    prod_smis = []
    for mol in prods:
        # Clear atom mappings
        [x.ClearProp('molAtomMapNumber') for x in mol.GetAtoms()]
        smi = Chem.MolToSmiles(mol)
        prod_smis.append(smi)

    react_smis = []
    reagent_syms = []
    for mol, atoms in reactants:
        # Clear atom mappings
        [x.ClearProp('molAtomMapNumber') for x in mol.GetAtoms()]

        # Remove molecules with no reacting atoms (reagents)
        # But represent as a symbol
        if not atoms:
            smi = Chem.MolToSmiles(mol)
            if smi not in reagents:
                reagents[smi] = len(reagents)
            reagent_syms.append('[A{}]'.format(reagents[smi]))

        else:
            smi = Chem.MolToSmiles(mol)
            react_smis.append(smi)

    source = '.'.join(react_smis)
    if reagent_syms:
        source += '>' + '.'.join(reagent_syms)
    target = '.'.join(prod_smis)
    source_toks = TOKEN_RE.findall(source)
    target_toks = TOKEN_RE.findall(target)
    return source_toks, target_toks

def clean(line): return line.strip().split()[0]


seen = set()
reactions = []
vocab = defaultdict(int)
with open('data/reactions.rsmi', 'r') as f:
    it = tqdm(map(process, map(clean, f)))
    for toks in it:
        if toks is None: continue
        h = md5('_'.join(''.join(ts) for ts in toks).encode('utf8')).hexdigest()
        if h in seen:
            continue
        else:
            seen.add(h)
        reactions.append(toks)
        for ts in toks:
            for t in ts:
                vocab[t] += 1
        it.set_postfix(
            reactions=len(reactions),
            vocab=len(vocab),
            reagents=len(reagents))

print('Reagents:', len(reagents))
print('Reactions:', len(reactions))
print('Vocab size:', len(vocab))

with open('data/reagents.dat', 'w') as f:
    lines = []
    for reagent, id in sorted(reagents.items(), key=lambda kv: kv[1]):
        lines.append('{}\t{}'.format(reagent, id))
    f.write('\n'.join(lines))

with open('data/vocab.dat', 'w') as f:
    lines = []
    for vocab, count in sorted(vocab.items(), key=lambda kv: kv[1], reverse=True):
        lines.append('{}\t{}'.format(vocab, id))
    f.write('\n'.join(lines))

with open('data/reactions.dat', 'w') as f:
    lines = []
    for source_toks, target_toks in reactions:
        lines.append('{}\t{}'.format(
            ' '.join(source_toks),
            ' '.join(target_toks)))
    f.write('\n'.join(lines))
