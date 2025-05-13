import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import Sequential, layers, optimizers, callbacks
from tensorflow.keras.layers import Dense, Dropout

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.AUC(curve='ROC', name='auc'),
]

def metrics_calc(data):
    data = data.astype(float)
    data['Sensitivity'] = data['TP'] / (data['TP'] + data['FN'])
    data['Specificity'] = data['TN'] / (data['TN'] + data['FP'])
    data['MCC'] = ((data['TP'] * data['TN']) - (data['FP'] * data['FN'])) / np.sqrt ( 
        (data['TP'] + data['FP']) * (data['TP'] + data['FN']) * (data['TN'] + data['FP']) * (data['TN'] + data['FN']))
    data['ROC-AUC'] = data['AUC']
    data['G-Mean'] = np.sqrt(data['Sensitivity'] * data['Specificity'])
    return(data)

def build_dnn(input_dim, dropout=0.25, lr=0.001, n_hidden1=1000, n_hidden2=500):
    model = Sequential([
        layers.Dense(n_hidden1, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(dropout),
        layers.Dense(n_hidden2, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=[METRICS])
    return model

def run_dnn(X_train, y_train, X_val, y_val, model_path, epochs=2000, batch_size=64, patience=50):
    model = build_dnn(X_train.shape[1])
    cb_list = [
        callbacks.ModelCheckpoint(model_path, save_best_only=True),
        callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=cb_list, verbose=0)
    return model

def run_transfer_learning(base_model_path, X_train, y_train, X_val, y_val, output_path, epochs=2000, batch_size=64, patience=50):
    base_model = tf.keras.models.load_model(base_model_path)
    model = Sequential(base_model.layers)
    for layer in model.layers[:-3]:
        layer.trainable = False
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[METRICS])
    cb_list = [
        callbacks.ModelCheckpoint(output_path, save_best_only=True),
        callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=cb_list, verbose=0)
    return model

def main():

    train_fp = pd.read_csv("../Data/Training/M1/M1_training_set_FPs.csv", index_col=0)
    test_fp = pd.read_csv("../Data/Test/M1_test_scaffold_split_FPs.csv", index_col=0)
    tl_fp = pd.read_csv("../Data/Training/M1/TL_GLASS_FPs.csv", index_col=0)

    y_train_full = np.asarray(train_fp.pop('Activity')).reshape(-1, 1)
    X_train_full = train_fp.to_numpy()
    y_test = np.asarray(test_fp.pop('Activity')).reshape(-1, 1)
    X_test = test_fp.to_numpy()
    y_tl = np.asarray(tl_fp.pop('Activity')).reshape(-1, 1)
    X_tl = tl_fp.to_numpy()

    # Train DNN on TL dataset first and save model
    X_train_tl, X_val_tl, y_train_tl, y_val_tl = train_test_split(X_tl, y_tl, test_size=0.1, random_state=42, shuffle=True)
    model_path_tl = "../Model/TL_model.keras"
    run_dnn(X_train_tl, y_train_tl, X_val_tl, y_val_tl, model_path_tl)

    dnn_validation, dnn_ext, tl_validation, tl_ext = [], [], [], []

    for _ in range(10):
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=None, shuffle=True)

        model_dnn = run_dnn(X_train, y_train, X_val, y_val, "../Model/best_model_DNN.keras")
        dnn_validation.append(model_dnn.evaluate(X_val, y_val, verbose=0))
        dnn_ext.append(model_dnn.evaluate(X_test, y_test, verbose=0))

        model_tl = run_transfer_learning(model_path_tl, X_train, y_train, X_val, y_val, "data/models/best_model_TL.keras")
        tl_validation.append(model_tl.evaluate(X_val, y_val, verbose=0))
        tl_ext.append(model_tl.evaluate(X_test, y_test, verbose=0))

    columns = ['Loss', 'TP', 'FP', 'TN', 'FN', 'AUC']
    pd.DataFrame(metrics_calc(pd.DataFrame(dnn_validation, columns=columns)))[['Sensitivity', 'Specificity', 'MCC', 'ROC-AUC', 'G-Mean']].to_csv("../Data/Results/dnn_validation.csv")
    pd.DataFrame(metrics_calc(pd.DataFrame(dnn_ext, columns=columns)))[['Sensitivity', 'Specificity', 'MCC', 'ROC-AUC', 'G-Mean']].to_csv("../Data/Results/dnn_ext.csv")
    pd.DataFrame(metrics_calc(pd.DataFrame(tl_validation, columns=columns)))[['Sensitivity', 'Specificity', 'MCC', 'ROC-AUC', 'G-Mean']].to_csv("../Data/Results/tl_validation.csv")
    pd.DataFrame(metrics_calc(pd.DataFrame(tl_ext, columns=columns)))[['Sensitivity', 'Specificity', 'MCC', 'ROC-AUC', 'G-Mean']].to_csv("../Data/Results/tl_ext.csv")

if __name__ == "__main__":
    main()
