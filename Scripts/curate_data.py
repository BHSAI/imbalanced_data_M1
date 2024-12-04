import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, PandasTools
from chembl_structure_pipeline import standardizer, exclude_flag
import deepchem as dc

scaffoldsplitter = dc.splits.ScaffoldSplitter()

'''
Define the range to be used for active/inactive classification
'''

active_range=1
inactive_range=10

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
1. Collect and combine the data from BindingDB and ChEMBL
'''

bdb = pd.read_csv('../Data/Input/M1_BindingDB.csv').dropna()
chembl_EC50 = pd.read_csv('../Data/Input/M1_ChEMBL_EC50.csv').dropna()
chembl_IC50 = pd.read_csv('../Data/Input/M1_ChEMBL_IC50.csv').dropna()
chembl_Ki = pd.read_csv('../Data/Input/M1_ChEMBL_Ki.csv').dropna()

bdb1 = bdb[bdb['Value'].str.contains('>|<')==False][['SMILES','Value']]
bdb2 = bdb[bdb['Value'].str.contains('>')==True][['SMILES','Value']]
bdb2['Value'] = bdb2['Value'].str.split('>').str[1]
bdb3 = bdb2[bdb2['Value'].astype(float) > (inactive_range*1000)]

chm = pd.concat([chembl_EC50, chembl_IC50, chembl_Ki])
chm1 = chm[chm['Relation'].str.contains('>|<')==False][['SMILES','Value']]
chm2 = chm[chm['Relation'].str.contains('>')==True][['SMILES','Value']]
chm3 = chm2[chm2['Value'].astype(float) > (inactive_range*1000)]

comb_all = pd.concat([bdb1, bdb3, chm1, chm3])
comb_all['SMILES'] = comb_all['SMILES'].str.split('|').str[0]
comb_all['Value'] = (comb_all['Value'].astype(float))/1000
comb_all = comb_all.sort_values(by=['Value']).reset_index(drop=True)
print('Original combined data', comb_all.shape[0])

'''
2. Standardize the smiles and remove invalid ones
'''

comb_all['is_valid'] = comb_all['SMILES'].apply(is_valid_structure)
excluded_df = comb_all[comb_all['is_valid'] == False].drop(columns=['is_valid'])
df_valid = comb_all[comb_all['is_valid']== True].drop(columns=['is_valid'])
df_valid['SMILES'] = df_valid['SMILES'].apply(standardize_and_canonicalize)
print('Original combined data after SMILES standardization', df_valid.shape[0])

df_valid = df_valid[['SMILES','Value']].reset_index(drop=True)
df_valid.to_csv('../Data/Input/M1_public_wActivity.csv')

'''
3. Label the data as actives and inactives based the range defined earlier
'''

df_valid['Activity'] = 2

df_valid.loc[df_valid['Value'] < active_range, 'Activity'] = 1
df_valid.loc[df_valid['Value'] > inactive_range, 'Activity'] = 0
df_valid_filtered = df_valid[df_valid['Activity'] < 2]
print('Original combined data w/Labels', df_valid_filtered['Activity'].value_counts())

'''
4. Remove duplicate and conflicting SMILES; calculate 1024 bit Morgan Fingerprints
'''

all = df_valid_filtered.drop_duplicates(subset=['SMILES','Activity'])
data = all.drop_duplicates(subset=['SMILES'], keep=False).reset_index(drop=True)
print('Labelled combined data w/o duplicates', data['Activity'].value_counts())

PandasTools.AddMoleculeColumnToFrame(frame=data, smilesCol='SMILES', molCol='Molecule')
data['Morgan2FP'] = data['Molecule'].map(computeMorganFP)

'''
5. Split the data into actives and inactives
6. Define the deepchem dataset and use scaffold split for training and test seperately for actives and inactives (due to the data imbalance problem)
'''

actives = data[data['Activity'] == 1]
inactives = data[data['Activity'] == 0]
acx = actives['Morgan2FP'].to_numpy()
inx = inactives['Morgan2FP'].to_numpy()
acy = actives['Activity'].to_numpy()
iny = inactives['Activity'].to_numpy()

dataset_ac = dc.data.DiskDataset.from_numpy(X=acx,y=acy,w=np.zeros(len(acx)),ids=actives['SMILES'])
dataset_in = dc.data.DiskDataset.from_numpy(X=inx,y=iny,w=np.zeros(len(inx)),ids=inactives['SMILES'])
train_dataset_ac, test_dataset_ac = scaffoldsplitter.train_test_split(dataset_ac, frac_train=0.8, seed = 10)
train_dataset_in, test_dataset_in = scaffoldsplitter.train_test_split(dataset_in, frac_train=0.8, seed = 10)

train_dataset_ac_df = pd.DataFrame([train_dataset_ac.ids, train_dataset_ac.X, train_dataset_ac.y], index=['SMILES', 'Morgan2FP','Activity']).T
train_dataset_in_df = pd.DataFrame([train_dataset_in.ids, train_dataset_in.X, train_dataset_in.y], index=['SMILES', 'Morgan2FP','Activity']).T
test_dataset_ac_df = pd.DataFrame([test_dataset_ac.ids, test_dataset_ac.X, test_dataset_ac.y], index=['SMILES', 'Morgan2FP','Activity']).T
test_dataset_in_df = pd.DataFrame([test_dataset_in.ids, test_dataset_in.X, test_dataset_in.y], index=['SMILES', 'Morgan2FP','Activity']).T

'''
7. Check for common fingerprints between the active and inactive datasets and remove such entries
'''

ind_ac_tr=[]
ind_in_tr=[]
for i in range(train_dataset_in_df.shape[0]):
    for j in range(train_dataset_ac_df.shape[0]):
        if np.array_equal(train_dataset_in_df.Morgan2FP[i],train_dataset_ac_df.Morgan2FP[j]):
            print('Train',i,j)
            ind_ac_tr.append(j)
            ind_in_tr.append(i)

train_dataset_ac_df = train_dataset_ac_df.drop(ind_ac_tr).reset_index(drop=True)
train_dataset_in_df = train_dataset_in_df.drop(ind_in_tr).reset_index(drop=True)

ind_ac_te=[]
ind_in_te=[]
for i in range(test_dataset_in_df.shape[0]):
    for j in range(test_dataset_ac_df.shape[0]):
        if np.array_equal(test_dataset_in_df.Morgan2FP[i],test_dataset_ac_df.Morgan2FP[j]):
            print('Test',i,j)
            ind_ac_te.append(j)
            ind_in_te.append(i)

test_dataset_ac_df = test_dataset_ac_df.drop(ind_ac_te).reset_index(drop=True)
test_dataset_in_df = test_dataset_in_df.drop(ind_in_te).reset_index(drop=True)

'''
8. Concatenate the actives and inactives to form final training and test datasets.
9. Do a final check between the actives and inactives to remove duplicates
'''

training_set = pd.concat([train_dataset_ac_df,train_dataset_in_df]).reset_index(drop=True)
ext_test_set = pd.concat([test_dataset_ac_df,test_dataset_in_df]).reset_index(drop=True)

print('Initial Training set',training_set['Activity'].value_counts())

source=[]
for i in range(training_set.shape[0]):
    for j in range(ext_test_set.shape[0]):
        if np.array_equal(training_set.Morgan2FP[i],ext_test_set.Morgan2FP[j]):
            print('Public',i,j)
            source.append(i)

training_set = training_set.drop(source).reset_index(drop=True)
print('Final training set',training_set['Activity'].value_counts())
print('Ext Test set',ext_test_set['Activity'].value_counts())

training_set[['SMILES','Activity']].to_csv('../Data/Training/M1_training_set.csv', index=False)
ext_test_set[['SMILES','Activity']].to_csv('../Data/Test/M1_test_scaffold_split.csv', index=False)