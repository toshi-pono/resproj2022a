from rdkit import Chem
from rdkit.Chem import Draw

c = Chem.MolFromSmiles('CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O')
Draw.MolToFile(c, 'exercise_A.png')
