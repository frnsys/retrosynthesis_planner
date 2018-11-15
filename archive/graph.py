"""
Experiment with creation a reaction graph
(i.e. compounds are nodes and edges are transformations)

(Too resource intenstive)

sudo apt install -y libxml2-dev zlib1g-dev
"""
import json
import igraph
from tqdm import tqdm
from hashlib import md5
from preprocess import process, clean
from itertools import product

seen = set()
mol_idx = {}
reactions = []
edgelist = []
with open('data/reactions.rsmi', 'r') as f:
    it = tqdm(map(process, map(clean, f)))
    for toks in it:
        if toks is None: continue
        h = md5('_'.join(''.join(ts) for ts in toks).encode('utf8')).hexdigest()
        if h in seen:
            continue
        else:
            seen.add(h)
        source_toks, target_toks = toks
        # Drop reagents
        sources = ''.join(source_toks).split('>')[0].split('.')
        targets = ''.join(target_toks).split('.')
        for mol in sources:
            if mol not in mol_idx:
                # Add to index
                mol_idx[mol] = len(mol_idx)
        for mol in targets:
            if mol not in mol_idx:
                # Add to index
                mol_idx[mol] = len(mol_idx)

        for src, tgt in product(sources, targets):
            edgelist.append((mol_idx[src], mol_idx[tgt]))

        it.set_postfix(
            vertices=len(mol_idx),
            edges=len(edgelist))

with open('mol.json', 'w') as f:
    json.dump(mol_idx, f)
with open('edges.json', 'w') as f:
    json.dump(edgelist, f)

G = igraph.Graph(n=len(mol_idx), edges=edgelist, directed=True)

import ipdb; ipdb.set_trace()
