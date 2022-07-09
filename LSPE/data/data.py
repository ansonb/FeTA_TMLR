"""
    File to load dataset based on user control from main file
"""

from data.molecules import MoleculeDataset
from data.ogb_mol import OGBMOLDataset
from data.tudatasets import TUDataset
from data.SBMs import SBMsDataset

def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC' or DATASET_NAME == 'ZINC-full':
        return MoleculeDataset(DATASET_NAME)
    
    # handling for MOLPCBA and MOLTOX21 dataset
    if DATASET_NAME in ['OGBG-MOLPCBA', 'OGBG-MOLTOX21', 'OGBG-MOLHIV']:
        return OGBMOLDataset(DATASET_NAME)

    TU_DATASETS = ['MUTAG','NCI1','PROTEINS','PTC']
    if DATASET_NAME in TU_DATASETS:
        return TUDataset(DATASET_NAME)

    SBM_DATASETS = ['PATTERN','CLUSTER']
    if DATASET_NAME in SBM_DATASETS:
        return SBMsDataset(DATASET_NAME)