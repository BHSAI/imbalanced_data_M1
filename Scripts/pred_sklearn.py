import pickle
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem, PandasTools
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
import xgboost as xgb

'''
Choose the ML model out of Naive Bayes, Random forest and XGBoost
'''

clf = BernoulliNB()
#clf = RandomForestClassifier()
#lf = xgb.XGBClassifier()

model = 'nb'

'''
Calculate 1024-bit Morgan fingerprint
'''

def computeMorganFP(mol, depth=2, nBits=1024):
    a = np.zeros(nBits)
    try:
      DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol,depth,nBits),a)
    except:
      return None
    return a

'''
Load the training set
'''

training_set = pd.read_csv('../Data/Training/M1_training_set.csv', index_col=None)
PandasTools.AddMoleculeColumnToFrame(frame=training_set, smilesCol='SMILES', molCol='Molecule')
training_set['Morgan2FP'] = training_set['Molecule'].map(computeMorganFP)
X_train_ext = training_set.Morgan2FP
y_train_ext = training_set.Activity

'''
Model building and calculating 10-fold stratified cross-validation results
'''

folds = []
validation = pd.DataFrame(columns=['Sentitivity','Specificity','MCC','ROC-AUC','G-Mean'])

skf = StratifiedKFold(n_splits=10, shuffle = True, random_state=125)
labels = [0,1]

for i, (train_index, test_index) in enumerate(skf.split(X_train_ext, y_train_ext)):
    folds.append({'train':train_index,'test':test_index})

for i in range (len(folds)):
    X_train = pd.DataFrame(X_train_ext).iloc[folds[i]['train']].to_numpy().ravel()
    X_test = pd.DataFrame(X_train_ext).iloc[folds[i]['test']].to_numpy().ravel()
    y_train = pd.DataFrame(y_train_ext).iloc[folds[i]['train']].to_numpy().ravel()
    y_test = pd.DataFrame(y_train_ext).iloc[folds[i]['test']].to_numpy().ravel()

    clf.fit(list(X_train), y_train)
    y_pred = clf.predict(list(X_test))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=labels).ravel()

    sensitivity = tp/(tp + fn)
    specificity = tn/(tn + fp)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    g_mean = np.sqrt(sensitivity*specificity)

    validation.loc[i,:] = [sensitivity,specificity,mcc,roc_auc,g_mean]

validation.to_csv('../Data/Results/{}_validation.csv'.format(model))

with open('../Data/Models/{}.pkl'.format(model),'wb') as f:
    pickle.dump(clf,f)