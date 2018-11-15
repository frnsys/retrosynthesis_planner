import json
import molvs
import random
import policies
from tqdm import tqdm
from mcts import Node, mcts
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem

# Load base compounds
starting_mols = set()
with open('data/emolecules.smi', 'r') as f:
    for line in tqdm(f, desc='Loading base compounds'):
        smi = line.strip()
        smi = molvs.standardize_smiles(smi)
        starting_mols.add(smi)
print('Base compounds:', len(starting_mols))

# Load policy networks
with open('model/rules.json', 'r') as f:
    rules = json.load(f)
    rollout_rules = rules['rollout']
    expansion_rules = rules['expansion']

rollout_net = policies.RolloutPolicyNet(n_rules=len(rollout_rules))
expansion_net = policies.ExpansionPolicyNet(n_rules=len(expansion_rules))
filter_net = policies.InScopeFilterNet()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, 'model/model.ckpt')


def transform(mol, rule):
    """Apply transformation rule to a molecule to get reactants"""
    rxn = AllChem.ReactionFromSmarts(rule)
    results = rxn.RunReactants([mol])

    # Only look at first set of results (TODO any reason not to?)
    results = results[0]
    reactants = [Chem.MolToSmiles(smi) for smi in results]
    return reactants


def expansion(node):
    """Try expanding each molecule in the current state
    to possible reactants"""

    # Assume each mol is a SMILES string
    mols = node.state

    # Convert mols to format for prediction
    # If the mol is in the starting set, ignore
    mols = [mol for mol in mols if mol not in starting_mols]
    fprs = policies.fingerprint_mols(mols)

    # Predict applicable rules
    preds = sess.run(expansion_net.pred_op, feed_dict={
        expansion_net.keep_prob: 1.,
        expansion_net.X: fprs,
        expansion_net.k: 5
    })

    # Generate children for reactants
    children = []
    for mol, rule_idxs in zip(mols, preds):
        # State for children will
        # not include this mol
        new_state = mols - {mol}

        mol = Chem.MolFromSmiles(mol)
        for idx in rule_idxs:
            # Extract actual rule
            rule = expansion_rules[idx]

            # TODO filter_net should check if the reaction will work?
            # should do as a batch

            # Apply rule
            reactants = transform(mol, rule)

            if not reactants: continue

            state = new_state | set(reactants)
            terminal = all(mol in starting_mols for mol in state)
            child = Node(state=state, is_terminal=terminal, parent=node, action=rule)
            children.append(child)
    return children


def rollout(node, max_depth=200):
    cur = node
    for _ in range(max_depth):
        if cur.is_terminal:
            break

        # Select a random mol (that's not a starting mol)
        mols = [mol for mol in cur.state if mol not in starting_mols]
        mol = random.choice(mols)
        fprs = policies.fingerprint_mols([mol])

        # Predict applicable rules
        preds = sess.run(rollout_net.pred_op, feed_dict={
            expansion_net.keep_prob: 1.,
            expansion_net.X: fprs,
            expansion_net.k: 1
        })

        rule = rollout_rules[preds[0][0]]
        reactants = transform(Chem.MolFromSmiles(mol), rule)
        state = cur.state | set(reactants)

        # State for children will
        # not include this mol
        state = state - {mol}

        terminal = all(mol in starting_mols for mol in state)
        cur = Node(state=state, is_terminal=terminal, parent=cur, action=rule)

    # Max depth exceeded
    else:
        print('Rollout reached max depth')

        # Partial reward if some starting molecules are found
        reward = sum(1 for mol in cur.state if mol in starting_mols)/len(cur.state)

        # Reward of -1 if no starting molecules are found
        if reward == 0:
            return -1.

        return reward

    # Reward of 1 if solution is found
    return 1.


def plan(target_mol):
    """Generate a synthesis plan for a target molecule (in SMILES form).
    If a path is found, returns a list of (action, state) tuples.
    If a path is not found, returns None."""
    root = Node(state={target_mol})

    path = mcts(root, expansion, rollout, iterations=2000, max_depth=200)
    if path is None:
        print('No synthesis path found. Try increasing `iterations` or `max_depth`.')
    else:
        print('Path found:')
        path = [(n.action, n.state) for n in path[1:]]
    return path


if __name__ == '__main__':
    # target_mol = '[H][C@@]12OC3=C(O)C=CC4=C3[C@@]11CCN(C)[C@]([H])(C4)[C@]1([H])C=C[C@@H]2O'
    target_mol = 'CC(=O)NC1=CC=C(O)C=C1'
    root = Node(state={target_mol})
    path = plan(target_mol)
    import ipdb; ipdb.set_trace()
