import pickle
import pandas as pd
from rdkit.Chem import PandasTools
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,matthews_corrcoef

import logging
import argparse
from typing import Tuple

from utils import computeMorganFP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_classifier(model_name: str):
    if model_name == 'nb':
        return BernoulliNB()
    elif model_name == 'rf':
        return RandomForestClassifier(random_state=42)
    elif model_name == 'xg':
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Evaluate a classification model and return accuracy and ROC AUC."""
    y_pred = model.predict(list(X_test))
    acc = accuracy_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    sen = tp/(tp + fn)
    spec = tn/(tn + fp)
    g_mean = np.sqrt(sen*spec)

    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    mcc = matthews_corrcoef(y_test, y_pred)

    logging.info(f"Evaluation -> Accuracy: {acc:.2f}, Sensitivity: {sen:.2f}, Specificity: {spec:.2f}, MCC: {mcc:.2f}, ROC AUC: {roc:.2f}, G-Mean: {g_mean:.2f}")
    return acc, sen, spec, mcc, roc, g_mean

def main(model: str):
    clf = get_classifier(model)

    training_set = pd.read_csv('../Data/Training/M1/M1_training_set.csv', index_col=None)
    PandasTools.AddMoleculeColumnToFrame(frame=training_set, smilesCol='SMILES', molCol='Molecule')
    training_set['Morgan2FP'] = training_set['Molecule'].map(computeMorganFP)

    X = training_set['Morgan2FP'].to_list()
    y = training_set['Activity'].to_numpy()

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=125)
    validation = pd.DataFrame(columns=['Accuracy', 'Sensitivity', 'Specificity', 'MCC', 'ROC-AUC', 'G-Mean'])

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = [X[i] for i in train_index]
        X_test = [X[i] for i in test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        clf.fit(X_train, y_train)
        validation.loc[i] = evaluate_model(clf, X_test, y_test)

    validation.to_csv('../Data/Results/{}_validation.csv'.format(model))
    with open(f"../Data/Model/{model}.pkl", "wb") as f:
          pickle.dump(clf, f)

    logging.info(f"Model '{model}' trained and saved. Validation results saved as well.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model with 10-fold CV on M1 dataset.")
    parser.add_argument("--model", required=True, choices=["nb", "rf", "xgb"], help="Model type: nb (Naive Bayes), rf (Random Forest), xgb (XGBoost)")
    args = parser.parse_args()
    main(args.model)