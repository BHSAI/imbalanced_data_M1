import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit import DataStructs
from chembl_structure_pipeline import standardizer
from chembl_structure_pipeline import exclude_flag

def is_valid_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    exclude = exclude_flag.exclude_flag(mol)
    return not exclude

def standardize_and_canonicalize(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        m_no_salts = standardizer.get_parent_mol(molecule) # Remove salts
        tostandarize = m_no_salts[0]
        std_mol = standardizer.standardize_mol(tostandarize)
        canonical_smiles = Chem.MolToSmiles(std_mol)
        return canonical_smiles
    except Exception as e:
        print(f"Error processing SMILES: {smiles}")
        print(e)
        return None

def computeMorganFP(mol, depth=2, nBits=1024):
    a = np.zeros(nBits)
    try:
      DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol,depth,nBits),a)
    except:
      return None
    return a

'''
1. Load the original and RNN generated datasets
2. Combine them with labels, standardize the SMILES and remove duplicates
'''

training_set = pd.read_csv('../Data/Training/M1_training_set.csv', index_col=None)

rnn_inactives = pd.read_csv('../Data/Input/M1_RNN_inactives.csv', index_col=0)
rnn_inactives.columns = ['SMILES']
rnn_inactives[['Activity']]=2

all_data_filter = pd.concat([training_set, rnn_inactives]).reset_index(drop=True)

all_data_filter['is_valid'] = all_data_filter['SMILES'].apply(is_valid_structure)
excluded_df = all_data_filter[all_data_filter['is_valid'] == False].drop(columns=['is_valid'])
df_valid = all_data_filter[all_data_filter['is_valid']== True].drop(columns=['is_valid'])
df_valid['SMILES'] = df_valid['SMILES'].apply(standardize_and_canonicalize)

'''
3. Arrange the SMILES based on length
4. Remove all RNN generated SMILES that are shorter than the smallest original SMILES and remove duplicates
'''

dfx = df_valid
all_rdkit_noduplicates = dfx.drop_duplicates(subset=['SMILES','Activity']).dropna().sort_values(by="SMILES", key=lambda x: x.str.len()).reset_index(drop=True)

for i in range(all_rdkit_noduplicates.shape[0]):
    if all_rdkit_noduplicates['Activity'][i]==0:
        print(i)
        break

all_rdkit_noduplicates = all_rdkit_noduplicates.iloc[i:].sort_values(by="Activity").reset_index(drop=True)
print('RNN Combined', all_rdkit_noduplicates['Activity'].value_counts())
all_rdkit_noduplicates = all_rdkit_noduplicates.drop_duplicates(subset=['SMILES'])
print('RNN Combined w/o duplicates', all_rdkit_noduplicates['Activity'].value_counts())

'''
5. Remove duplicates after sorting actives, inactives and RNN
'''

PandasTools.AddMoleculeColumnToFrame(frame=all_rdkit_noduplicates, smilesCol='SMILES', molCol='Molecule')
all_rdkit_noduplicates['Morgan2FP'] = all_rdkit_noduplicates['Molecule'].map(computeMorganFP)
rnn_training = all_rdkit_noduplicates[all_rdkit_noduplicates['Activity']==2].reset_index(drop=True)
ac = all_rdkit_noduplicates[all_rdkit_noduplicates['Activity']==1].reset_index(drop=True)
inac = all_rdkit_noduplicates[all_rdkit_noduplicates['Activity']==0].reset_index(drop=True)

source_ac = []
source_in = []

for i in range(inac.shape[0]):
    for j in range(rnn_training.shape[0]):
        if np.array_equal(inac.Morgan2FP[i],rnn_training.Morgan2FP[j]):
            print('Inactives',i,j)
            source_in.append(j)

for i in range(ac.shape[0]):
    for j in range(rnn_training.shape[0]):
        if np.array_equal(ac.Morgan2FP[i],rnn_training.Morgan2FP[j]):
            print('Actives',i,j)
            source_ac.append(j)

rnn_training = rnn_training.drop(source_in).reset_index(drop=True)
rnn_training = rnn_training.drop(source_ac).reset_index(drop=True)

'''
5. Combine actives, inactives and RNN, and remove duplicates with external test set
'''

final_training_set = pd.concat([ac, inac, rnn_training]).reset_index(drop=True)
print('Initial Training set',final_training_set['Activity'].value_counts())

ext_test_set = pd.read_csv('../Data/Test/M1_test_scaffold_split.csv', index_col=None)
PandasTools.AddMoleculeColumnToFrame(frame=ext_test_set, smilesCol='SMILES', molCol='Molecule')
ext_test_set['Morgan2FP'] = ext_test_set['Molecule'].map(computeMorganFP)

source=[]
for i in range(final_training_set.shape[0]):
    for j in range(ext_test_set.shape[0]):
        if np.array_equal(final_training_set.Morgan2FP[i],ext_test_set.Morgan2FP[j]):
            print('Public',i,j)
            source.append(i)

final_training_set = final_training_set.drop(source).reset_index(drop=True)
print('Final training set',final_training_set['Activity'].value_counts())
print('Ext Test set',ext_test_set['Activity'].value_counts())

final_training_set[['SMILES','Activity']].to_csv('../Data/Training/M1_training_set_RNN.csv', index=False)
