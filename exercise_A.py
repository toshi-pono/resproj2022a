from rdkit import Chem
from rdkit.Chem import Draw

m = Chem.MolFromSmiles('CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O')
Draw.MolToFile(m, 'exercise_A.png')
