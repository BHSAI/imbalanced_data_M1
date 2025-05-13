import logging

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools

from utils import is_valid_structure, standardize_and_canonicalize, computeMorganFP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
def main():
    '''
    1. Load the original and RNN/DEG generated datasets
    2. Combine them with labels, standardize the SMILES and remove duplicates
    '''

    training_set = pd.read_csv('../Data/Training/M1/M1_training_set.csv', index_col=None)
    gen_inactives = pd.read_csv('../Data/Training/M1/M1_RNN_inactives.csv', index_col=None)
    gen_inactives[['Activity']]=2

    all_data = pd.concat([training_set, gen_inactives]).reset_index(drop=True)

    all_data['is_valid'] = all_data['SMILES'].apply(is_valid_structure)
    excluded_df = all_data[all_data['is_valid'] == False].drop(columns=['is_valid'])
    df_valid = all_data[all_data['is_valid']== True].drop(columns=['is_valid'])
    df_valid['SMILES'] = df_valid['SMILES'].apply(standardize_and_canonicalize)

    '''
    3. Arrange the SMILES based on length
    4. Remove all RNN/DEG generated SMILES that are shorter than the smallest original SMILES and remove duplicates
    '''

    dfx = df_valid
    all_rdkit_noduplicates = dfx.drop_duplicates(subset=['SMILES','Activity']).dropna().sort_values(by="SMILES", key=lambda x: x.str.len()).reset_index(drop=True)

    min_index = next(i for i, val in enumerate(dfx['Activity']) if val == 0)
    all_rdkit_noduplicates = dfx.iloc[min_index:].sort_values(by="Activity").drop_duplicates(subset=['SMILES']).reset_index(drop=True)

    print('Gen Combined', all_rdkit_noduplicates['Activity'].value_counts())
    all_rdkit_noduplicates = all_rdkit_noduplicates.drop_duplicates(subset=['SMILES'])
    print('Gen Combined w/o duplicates', all_rdkit_noduplicates['Activity'].value_counts())

    '''
    5. Remove duplicates after sorting actives, inactives and generated compounds
    '''

    PandasTools.AddMoleculeColumnToFrame(frame=all_rdkit_noduplicates, smilesCol='SMILES', molCol='Molecule')
    all_rdkit_noduplicates['Morgan2FP'] = all_rdkit_noduplicates['Molecule'].map(computeMorganFP)

    gen_training = all_rdkit_noduplicates[all_rdkit_noduplicates['Activity']==2].reset_index(drop=True)
    actives = all_rdkit_noduplicates[all_rdkit_noduplicates['Activity']==1].reset_index(drop=True)
    inactives = all_rdkit_noduplicates[all_rdkit_noduplicates['Activity']==0].reset_index(drop=True)

    source_ac = []
    source_in = []

    for i in range(inactives.shape[0]):
        for j in range(gen_training.shape[0]):
            if np.array_equal(inactives.Morgan2FP[i],gen_training.Morgan2FP[j]):
                print('Inactives overlap',i,j)
                source_in.append(j)

    for i in range(actives.shape[0]):
        for j in range(gen_training.shape[0]):
            if np.array_equal(actives.Morgan2FP[i],gen_training.Morgan2FP[j]):
                print('Actives overlap',i,j)
                source_ac.append(j)

    gen_training = gen_training.drop(source_in + source_ac).reset_index(drop=True)

    '''
    5. Combine actives, inactives and RNN/DEG, and remove duplicates with external test set
    '''

    final_training_set = pd.concat([actives, inactives, gen_training]).reset_index(drop=True)
    print('Initial Training set',final_training_set['Activity'].value_counts())

    ext_test_set = pd.read_csv('../Data/Test/M1_test_scaffold_split.csv', index_col=None)
    PandasTools.AddMoleculeColumnToFrame(frame=ext_test_set, smilesCol='SMILES', molCol='Molecule')
    ext_test_set['Morgan2FP'] = ext_test_set['Molecule'].map(computeMorganFP)

    source=[]
    for i in range(final_training_set.shape[0]):
        for j in range(ext_test_set.shape[0]):
            if np.array_equal(final_training_set.Morgan2FP[i],ext_test_set.Morgan2FP[j]):
                print('Public overlap',i,j)
                source.append(i)

    final_training_set = final_training_set.drop(source).reset_index(drop=True)
    print('Final training set',final_training_set['Activity'].value_counts())
    print('Ext Test set',ext_test_set['Activity'].value_counts())

    final_training_set[['SMILES','Activity']].to_csv('../Data/Training/M1/M1_training_set_RNN.csv', index=False)

if __name__ == "__main__":
    main()
