import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers 


'''
Define the metrics to be calculated
'''

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

'''
Inactive range options:
10: Original DNN
2: Boundary adjustment
10_RNN: RNN added training set
'''

inactive_range = '10'

'''
Load the training and test sets, and GLASS database for transfer learning as Morgan fingerprints
'''

training_set = pd.read_csv('../Data/Training/M1_training_set_1_{}_FPs.csv'.format(inactive_range), index_col=0)
train_labels = np.asarray(training_set.pop('Activity')).reshape(training_set.shape[0],1)
train_features = np.array(training_set)

ext_test_set =  pd.read_csv('../Data/Test/M1_test_scaffold_split_FPs.csv'.format(inactive_range), index_col=0)
ext_test_set_labels = np.asarray(ext_test_set.pop('Activity')).reshape(ext_test_set.shape[0],1)
ext_test_set_features = np.array(ext_test_set)

gpcr_tl = pd.read_csv('../Data/Training/TL_GLASS_1_10_FPs.csv', index_col=0)
gpcr_tl_labels = np.asarray(gpcr_tl.pop('Activity')).reshape(gpcr_tl.shape[0],1)
gpcr_tl_features = np.array(gpcr_tl)

'''
Define the DNN parameters
'''

batch_size = 64
n_epochs = 2000
patience = 50
drop_rate = 0.25
n_hidden1 =1000
n_hidden2 = 500
learning_rate = 0.001

'''
Run the DNN with two hidden layers with additional conditions for dropout and early stopping.
Save the DNN model run on the GLASS database for transfer learning.
'''
def dnn_normal():
    model = Sequential()
    model.add(Dense(n_hidden1, activation="relu", input_shape=(1024,)))
    model.add(Dropout(drop_rate))
    model.add(Dense(n_hidden2, activation="relu"))
    model.add(Dropout(drop_rate))
    model.add(Dense(1, activation="sigmoid"))
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=[METRICS])
    checkpoint_cb = keras.callbacks.ModelCheckpoint("../Data/Models/best_model_DNN.keras")
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_valid, y_valid),
                         callbacks=[checkpoint_cb, early_stopping_cb])
    if refs == 2:
        model.save("../Data/Models/TL_model.keras")
    validation_results = model.evaluate(X_valid, y_valid,
                                  batch_size=batch_size, verbose=0)
    ext_test_results = model.evaluate(ext_test_set_features, ext_test_set_labels,
                                  batch_size=batch_size, verbose=0)
    return validation_results, ext_test_results
        
'''
Transfer layers from the DNN run on GLASS database.
'''

def dnn_tl2():
    model_m = keras.models.load_model("../Data/Models/TL_model.keras")
    model_clone = keras.models.clone_model(model_m)
    model_clone.set_weights(model_m.get_weights())
    model_new = keras.models.Sequential(model_clone.layers)
    for layer in model_new.layers[:-3]:
        layer.trainable = False

    adam = optimizers.Adam(learning_rate=learning_rate)
    model_new.compile(loss = "binary_crossentropy", optimizer = adam, metrics=[METRICS])
    checkpoint_cb =keras.callbacks.ModelCheckpoint("best_model_TL.keras")
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    history = model_new.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_valid, y_valid),
                          callbacks=[checkpoint_cb, early_stopping_cb])
    validation_results = model_new.evaluate(X_valid, y_valid,
                                  batch_size=batch_size, verbose=0)
    ext_test_results = model_new.evaluate(ext_test_set_features, ext_test_set_labels,
                                  batch_size=batch_size, verbose=0)
    return validation_results, ext_test_results

'''
Split the GPCR data into training and validation sets and run the model. First run the GLASS database model and save it.
'''

X_train, X_valid, y_train, y_valid = train_test_split(gpcr_tl_features, gpcr_tl_labels, test_size=0.10, shuffle=True)
refs = 2
dnn_normal()

dnn_results = []
tl_results = []

dnn_validation = pd.DataFrame(columns=['Loss','TP','FP','TN','FN','AUC'])
tl_validation = pd.DataFrame(columns=['Loss','TP','FP','TN','FN','AUC'])
dnn_ext_results = pd.DataFrame(columns=['Loss','TP','FP','TN','FN','AUC'])
tl_ext_results = pd.DataFrame(columns=['Loss','TP','FP','TN','FN','AUC'])

'''
Build DNN (1024:1000:500:1) with 90:10 train:test of the smaller dataset, and then perform TL 10 times
'''

for repeats in range(10):
    X_train, X_valid, y_train, y_valid = train_test_split(train_features, train_labels, test_size=0.10, shuffle=True)

    refs = 0
    dnn_results.append(dnn_normal())
    dnn_validation.loc[repeats,:] = dnn_results[repeats][0]
    dnn_ext_results.loc[repeats,:] = dnn_results[repeats][1]    

    tl_results.append(dnn_tl2())
    tl_validation.loc[repeats,:] =  tl_results[repeats][0]
    tl_ext_results.loc[repeats,:] = tl_results[repeats][1]

pd.DataFrame(metrics_calc(dnn_validation))[['Sensitivity','Specificity','MCC','ROC-AUC','G-Mean']].to_csv('../Data/Results/dnn_validation_1_{}.csv'.format(inactive_range))
pd.DataFrame(metrics_calc(dnn_ext_results))[['Sensitivity','Specificity','MCC','ROC-AUC','G-Mean']].to_csv('../Data/Results/dnn_ext_1_{}.csv'.format(inactive_range)) 
pd.DataFrame(metrics_calc(tl_validation))[['Sensitivity','Specificity','MCC','ROC-AUC','G-Mean']].to_csv('../Data/Results/tl_validation_1_{}.csv'.format(inactive_range))
pd.DataFrame(metrics_calc(tl_ext_results))[['Sensitivity','Specificity','MCC','ROC-AUC','G-Mean']].to_csv('../Data/Results/tl_ext_1_{}.csv'.format(inactive_range))

