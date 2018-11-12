from tqdm import tqdm
from seq2seq import Seq2Seq
from preprocess import tokenize

model = Seq2Seq.load('model')

def process(smi):
    smi = smi.strip()

    # Only include compounds that exclusively
    # use tokens the model can generate
    if any(tok not in model.vocab2id for tok in tokenize(smi)):
        return None

    # TODO may need to canonicalize SMILES
    return smi.split('.')

smiles = set()
with open('data/emolecules.smi', 'r') as f:
    for smis in tqdm(map(process, f)):
        if smis is None: continue
        for smi in smis:
            smiles.add(smi)

print('Compounds:', len(smiles))

with open('data/base_compounds.smi', 'w') as f:
    f.write('\n'.join(smiles))
