import random
from seq2seq import Seq2Seq, END, pad_arrays
from mcts import Node, mcts
from preprocess import tokenize


model = Seq2Seq.load('model')

# TODO where to get a good list of these?
starting_mols = set()


def to_doc(mol):
    toks = tokenize(mol)
    return [model.vocab2id['<S>']] + [model.vocab2id[tok] for tok in toks] + [END]


def process_seq(seq):
    smis = ''.join([model.id2vocab[id] for id in seq])
    parts = smis.split('>')
    if len(parts) > 1:
        # There shouldn't be more than two parts
        reactants, reagents = parts[0], parts[1]
    else:
        reactants = parts[0]
        reagents = []
    reactants = reactants.split('.')

    return reactants, reagents



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
    preds = model.sess.run(model.pred_op, feed_dict={
        model.keep_prob: 1.,
        model.X: pad_arrays(docs),
        model.max_decode_iter: 500,
        # model.beam_width: 10
    })

    # Generate children for reactants
    children = []
    for mol, seqs in zip(mols_ordered, preds):
        # State for children will
        # not include this mol
        new_state = mols - {mol}

        for s in seqs:
            reactants, reagents = process_seq(s)
            # TODO should we discard reagents?
            # or store them on edges?

            state = new_state | set(reactants)
            terminal = all(mol in starting_mols for mol in state)
            child = Node(state=state, is_terminal=terminal, parent=node)
            children.append(child)
    return children


def rollout(node, max_depth=200):
    cur = node
    for _ in range(max_depth):
        if cur.is_terminal:
            break

        # Select a random mol (that's not a starting mol)
        mols = [mol for mol in node.state if mol not in starting_mols]
        mol = random.choice(mols)

        # State for children will
        # not include this mol
        new_state = node.state - {mol}

        # Preprocess for model
        doc = to_doc(mol)

        preds = model.sess.run(model.pred_op, feed_dict={
            model.keep_prob: 1.,
            model.X: [doc],
            model.max_decode_iter: 500,
            # model.beam_width: 1
        })
        seq = preds[0][0]
        reactants, reagents = process_seq(seq)

        # TODO ignore reagents or what?

        state = new_state | set(reactants)
        terminal = all(mol in starting_mols for mol in state)
        cur = Node(state=state, is_terminal=terminal, parent=cur)

    # Max depth exceeded
    else:
        print('Rollout reached max depth')
        return 0.

    # TODO look up rewards from paper
    return 1.


target_mol = '[H][C@@]12OC3=C(O)C=CC4=C3[C@@]11CCN(C)[C@]([H])(C4)[C@]1([H])C=C[C@@H]2O'
root = Node(state={target_mol})

path = mcts(root, expansion, rollout, iterations=2000, max_depth=200)
if path is None:
    print('No synthesis path found. Try increasing `iterations` or `max_depth`.')
else:
    print('Path found:')
    print(path)
    import ipdb; ipdb.set_trace()
