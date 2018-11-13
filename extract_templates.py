"""
Modified version of:
<https://github.com/connorcoley/ochem_predict_nn/blob/master/data/generate_reaction_templates.py>
"""

import re
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem, RDLogger
from itertools import chain
from multiprocessing import Pool

# Silence logs
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def canonicalize_template(template):
    '''This function takes one-half of a template SMARTS string
    (i.e., reactants or products) and re-orders them based on
    an equivalent string without atom mapping.'''

    # Strip labels to get sort orders
    template_nolabels = re.sub('\:[0-9]+\]', ']', template)

    # Split into separate molecules *WITHOUT wrapper parentheses*
    template_nolabels_mols = template_nolabels[1:-1].split(').(')
    template_mols          = template[1:-1].split(').(')

    # Split into fragments within those molecules
    for i in range(len(template_mols)):
        nolabel_mol_frags = template_nolabels_mols[i].split('.')
        mol_frags         = template_mols[i].split('.')

        # Get sort order within molecule, defined WITHOUT labels
        sortorder = [j[0] for j in sorted(enumerate(nolabel_mol_frags), key = lambda x:x[1])]

        # Apply sorting and merge list back into overall mol fragment
        template_nolabels_mols[i] = '.'.join([nolabel_mol_frags[j] for j in sortorder])
        template_mols[i]          = '.'.join([mol_frags[j] for j in sortorder])

    # Get sort order between molecules, defined WITHOUT labels
    sortorder = [j[0] for j in sorted(enumerate(template_nolabels_mols), key = lambda x:x[1])]

    # Apply sorting and merge list back into overall transform
    template = '(' + ').('.join([template_mols[i] for i in sortorder]) + ')'

    return template

def reassign_atom_mapping(transform):
    '''This function takes an atom-mapped reaction and reassigns
    the atom-mapping labels (numbers) from left to right, once
    that transform has been canonicalized.'''

    all_labels = re.findall('\:([0-9]+)\]', transform)

    # Define list of replacements which matches all_labels *IN ORDER*
    replacements = []
    replacement_dict = {}
    counter = 1
    for label in all_labels: # keep in order! this is important
        if label not in replacement_dict:
            replacement_dict[label] = str(counter)
            counter += 1
        replacements.append(replacement_dict[label])

    # Perform replacements in order
    transform_newmaps = re.sub('\:[0-9]+\]',
        lambda match: (':' + replacements.pop(0) + ']'),
        transform)

    return transform_newmaps

def canonicalize_transform(transform):
    '''This function takes an atom-mapped SMARTS transform and
    converts it to a canonical form by, if nececssary, rearranging
    the order of reactant and product templates and reassigning
    atom maps.'''

    transform_reordered = '>>'.join([canonicalize_template(x) for x in transform.split('>>')])
    return reassign_atom_mapping(transform_reordered)

def convert_to_retro(transform):
    '''This function takes a forward synthesis and converts it to a
    retrosynthesis. Only transforms with a single product are kept, since
    retrosyntheses should have a single reactant (and split it up accordingly).'''

    # Split up original transform
    reactants = transform.split('>>')[0]
    products  = transform.split('>>')[1]

    # Don't force products to be from different molecules (?)
    # -> any reaction template can be intramolecular (might remove later)
    products = products[1:-1].replace(').(', '.')

    # Don't force the "products" of a retrosynthesis to be two different molecules!
    reactants = reactants[1:-1].replace(').(', '.')

    return '>>'.join([products, reactants])

ATOM_MAP_PROP = 'molAtomMapNumber'
ATOM_EQL_PROPS = [
    'GetSmarts',
    'GetAtomicNum',
    'GetTotalNumHs',
    'GetFormalCharge',
    'GetDegree',
    'GetNumRadicalElectrons'
]

def wildcard(atom):
    # Terminal atom
    if atom.GetDegree() == 1:
        return atom.GetSmarts()

    parts = []
    if atom.GetAtomicNum() != 6:
        parts.append('#{}'.format(atom.GetAtomicNum()))
    elif atom.GetIsAromatic():
        parts.append('c')
    else:
        parts.append('C')
    if atom.GetFormalCharge() != 0:
        # Write charge with sign;
        # if charge is 1, include only the sign
        charge = atom.GetFormalCharge()
        charge = '{:+}'.format(charge).replace('1', '')
        parts.append(charge)

    # Drop last semicolon
    symbol = '[' + ';'.join(parts)

    # Include atom mapping, if any
    if atom.HasProp(ATOM_MAP_PROP):
        symbol += ':' + atom.GetProp(ATOM_MAP_PROP)

    symbol += ']'
    return symbol


def fragmentize(mols, changed_atoms, radius=1, include_unmapped=False, include_ids=None):
    include_ids = include_ids or set()
    ids = set()
    fragments = []
    for mol in mols:
        center = set()
        atoms = set()
        replacements = []
        for a in mol.GetAtoms():
            if not a.HasProp(ATOM_MAP_PROP):
                if include_unmapped:
                    atoms.add(a)
                continue

            id = a.GetProp(ATOM_MAP_PROP)
            if id not in changed_atoms:
                if id in include_ids:
                    atoms.add(a)
                    replacements.append((a.GetIdx(), wildcard(a)))
                continue

            # Prepare symbol replacements
            sma = a.GetSmarts()
            if a.GetTotalNumHs() == 0:
                sma = sma.replace(':', ';H0:')
            if a.GetFormalCharge() == 0:
                sma = sma.replace(':', ';+0:')
            replacements.append((a.GetIdx(), sma))
            center.add(a)

        # Include neighbors of changed atoms
        for _ in range(radius):
            to_add = set()
            for a in center:
                for neighbor in a.GetNeighbors():
                    # Already included, skip
                    if neighbor in atoms: continue

                    # TODO group stuff

                    replacements.append((a.GetIdx(), wildcard(a)))
                    to_add.add(a)
            center = center | to_add

        atoms = center | atoms
        if not atoms: continue

        symbols = [atom.GetSmarts() for atom in mol.GetAtoms()]
        for i, symbol in replacements:
            symbols[i] = symbol

        ids = ids | set(a.GetProp(ATOM_MAP_PROP) for a in atoms)

        # Clear atom mapping
        [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]

        fragments.append('(' + AllChem.MolFragmentToSmiles(mol, [a.GetIdx() for a in atoms],
                                                           atomSymbols=symbols, allHsExplicit=True,
                                                           isomericSmiles=False, allBondsExplicit=True) + ')')

    fragments = '.'.join(fragments)
    return fragments, ids



def extract(smarts):
    rxn = AllChem.ReactionFromSmarts(smarts)
    products = rxn.GetProducts()
    reactants = rxn.GetReactants()
    agents = rxn.GetAgents()

    # Only consider single product reactions
    if len(products) > 1:
        return

    product = Chem.Mol(products[0])
    [a.ClearProp('molAtomMapNumber') for a in product.GetAtoms()]
    product_smi = Chem.MolToSmiles(product)

    # Sanitize and canonicalize
    for mols in [products, reactants, agents]:
        for mol in mols:
            try:
                Chem.SanitizeMol(mol)
                Chem.rdMolTransforms.CanonicalizeMol(mol)
            except ValueError:
                return

    # Assure that all product atoms are mapped
    # (i.e. we can trace where they came from)
    for prod in products:
        all_mapped = all(a.HasProp(ATOM_MAP_PROP) for a in prod.GetAtoms())
        if not all_mapped:
            return

    # THIS IS TOO MUCH
    r_atoms = dict(chain(*[[(a.GetProp(ATOM_MAP_PROP), a) for a in m.GetAtoms() if a.HasProp(ATOM_MAP_PROP)] for m in reactants]))
    p_atoms = dict(chain(*[[(a.GetProp(ATOM_MAP_PROP), a) for a in m.GetAtoms() if a.HasProp(ATOM_MAP_PROP)] for m in products]))

    # Should consist of the same tags and atoms
    # TODO looks like they may not necessarily have the same tags?
    # "leaving atoms"?
    if set(r_atoms.keys()) != set(p_atoms.keys()) \
            or len(r_atoms) != len(p_atoms): return

    changed_atoms = []
    for id, r_atom in r_atoms.items():
        p_atom = p_atoms[id]
        eql = all(getattr(p_atom, prop) == getattr(r_atom, prop) for prop in ATOM_EQL_PROPS)
        if not eql:
            changed_atoms.append(id)

    # TODO "leaving" atoms?

    # Get fragments for reactants and products
    reactant_frags, reactant_ids = fragmentize(reactants, changed_atoms, radius=1)
    product_frags, _ = fragmentize(products, changed_atoms, radius=1, include_unmapped=True, include_ids=reactant_ids)

    transform = '{}>>{}'.format(reactant_frags, product_frags)

    # Validate
    rxn = AllChem.ReactionFromSmarts(transform)
    _, n_errs = rxn.Validate(silent=True)
    if n_errs > 0:
        return

    rxn_canonical = canonicalize_transform(transform)
    retro_canonical = convert_to_retro(rxn_canonical)
    return retro_canonical, product_smi

def clean(line): return line.strip().split()[0]

transforms = []
with open('data/reactions.rsmi', 'r') as f:
    with Pool() as p:
        for res in tqdm(p.imap(extract, map(clean, f))):
            if res is None: continue
            rxn, product = res
            transforms.append((rxn, product))

with open('data/templates.dat', 'w') as f:
    f.write('\n'.join(['\t'.join(rxn_prod) for rxn_prod in transforms]))

import ipdb; ipdb.set_trace()