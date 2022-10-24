from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import Chem


def smiles_to_descriptors(smiles: str) -> list[float]:
    """
    Convert a SMILES string to a descriptor vector

    Parameters
    ----------
    smiles : str
        SMILES string
    """
    mol_obj = Chem.MolFromSmiles(smiles)
    targets = descriptor_names()
    descriptors = MoleculeDescriptors.MolecularDescriptorCalculator(
        targets).CalcDescriptors(mol_obj)

    return list(descriptors)


def descriptor_names() -> list[str]:
    """
    Get the names of the descriptors
    """
    return [descriptor[0] for descriptor in Descriptors._descList]
