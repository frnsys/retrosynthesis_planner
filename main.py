import os
import random
import tensorflow as tf
from seq2seq import Seq2Seq, END, pad_arrays
from mcts import Node, mcts
from preprocess import tokenize


# Load vocab
vocab2id = {}
with open('data/vocab.idx', 'r') as f:
    for i, line in enumerate(f):
        term = line.strip()
        vocab2id[term] = i

# TODO load actual model
model = Seq2Seq()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

save_path = 'model'
ckpt_path = os.path.join(save_path, 'model.ckpt')
saver = tf.train.Saver()
saver.restore(sess, ckpt_path)

# TODO where to get a good list of these?
starting_mols = set()


def to_doc(mol):
    toks = tokenize(mol)
    return [vocab2id['<S>']] + [vocab2id[tok] for tok in toks] + [END]


def expansion(node):
    """Try expanding each molecule in the current state
    to possible reactants"""

    # Assume each mol is a SMILES string
    mols = node.state

    # Convert mols to format for prediction
    mol_docs = []
    for mol in mols:
        # If the mol is in the starting set, ignore
        if mol in starting_mols:
            continue

        # Preprocess for model
        doc = to_doc(mol)
        mol_docs.append((mol, doc))

    # Predict reactants
    mols_ordered, docs = zip(*mol_docs)
    preds = sess.run(model.pred_op, feed_dict={
        model.keep_prob: 1.,
        model.X: pad_arrays(docs),
        model.beam_width: 10
    })

    # Generate children for reactants
    children = []
    for mol, ps in zip(mols_ordered, preds):
        # State for children will
        # not include this mol
        new_state = mols - {mol}

        for p in ps:
            # TODO process preds
            # convert to SMILES
            # TODO should we discard reagents?
            # or store them on edges?
            smis = p
            state = new_state | set(smis)
            terminal = all(mol in starting_mols for mol in state)
            child = Node(state=state, is_terminal=terminal)
            children.append(child)
    return children


def rollout(node, max_depth=200):
    cur = node
    for _ in range(max_depth):
        if cur.is_terminal:
            break

        # Select a random mol (that's not a starting mol)
        mols = {mol for mol in node.state if mol not in starting_mols}
        mol = random.choice(mols)

        # State for children will
        # not include this mol
        new_state = node.state - {mol}

        # Preprocess for model
        doc = to_doc(mol)

        preds = sess.run(model.pred_op, feed_dict={
            model.keep_prob: 1.,
            model.X: [doc],
            model.beam_width: 1
        })
        pred = preds[0]

        # TODO properly extract reactant SMIs
        # TODO ignore reagents or what?
        smis = pred
        state = new_state | set(smis)
        terminal = all(mol in starting_mols for mol in state)
        cur = Node(state=state, is_terminal=terminal)

    # Max depth exceeded
    else:
        return 0.

    # TODO look up rewards from paper
    return 1.


target_mol = 'FOO'
root = Node(state={target_mol})

path = mcts(root, expansion, rollout, iterations=2000, max_depth=200)
if path is None:
    print('No synthesis path found. Try increasing `iterations` or `max_depth`.')
else:
    print('Path found:')
    print(path)
    import ipdb; ipdb.set_trace()
