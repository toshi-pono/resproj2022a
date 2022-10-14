from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import Chem


def SmileToDescriptorVec(smile: str) -> list[float]:
    mol = Chem.MolFromSmiles(smile)

    descriptor_names = [descriptor_name[0]
                        for descriptor_name in Descriptors._descList]
    descriptor_calculation = MoleculeDescriptors.MolecularDescriptorCalculator(
        descriptor_names)
    RDkit = descriptor_calculation.CalcDescriptors(mol)

    return list(RDkit)
