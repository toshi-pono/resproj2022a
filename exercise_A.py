from rdkit import Chem
from rdkit.Chem import Draw


def smiles_to_image(smiles: str, file: str) -> None:
    """
    SMILESから構造式を描画し、ファイルに保存する
    """
    mol_obj = Chem.MolFromSmiles(smiles)
    Draw.MolToFile(mol_obj, file)


if __name__ == "__main__":
    smiles_to_image('CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O', 'output/exercise_A.png')
