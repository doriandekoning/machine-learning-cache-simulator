import numpy as np
import pandas
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense,Activation, LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import functools
import keras.backend as K

#https://github.com/keras-team/keras/issues/2115
#def weighted_categorical_crossentropy(y_true, y_pred, weights):
#    nb_cl = len(weights)
#    final_mask = K.zeros_like(y_pred[:, 0])
#    y_pred_max = K.max(y_pred, axis=1)
#    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
#    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
#    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
#        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
#    return K.categorical_crossentropy(y_pred, y_true) * final_mask

if len(sys.argv) < 3:
    print("Not enough arguments, format is: [input_csv] [outputlocation]")
    exit(1)



#Import data
df = pandas.read_csv(sys.argv[1])
df = df.replace("r", 0)
df = df.replace("w", 1)
df = df.replace("nop", 2)
print(df)

trainingData =      df.values[:2800000]
validationData =    df.values[2800000:2810000]
print(trainingData.shape)
print(validationData.shape)
scaler = MinMaxScaler(feature_range=(0, 1)).fit(df.values[:, 0:1])
trainingData[:, 0:1] = scaler.transform(trainingData[:, 0:1])
validationData[:, 0:1] = scaler.transform(validationData[:, 0:1])

trainingInputData = np.array(trainingData[: , 0:2]).reshape(280, 10000, 2)
trainingOutputData = np.array(trainingData[: , 2:]).reshape(280, 10000, 1)
print(trainingData[:, 1:2])
sample_weights = trainingData[:, 1:2].reshape(280*10000)
print("Before")
print(sample_weights)
weights = {0:1, 1:67, 2:121}
#sample_weights[sample_weights == 0] = 1.
#sample_weights[sample_weights == 1] = 67.
#sample_weights[sample_weights == 2] = 121.
sample_weights = np.array([weights[xi] for xi in sample_weights]).reshape(280, 10000)
print("After")
print(sample_weights)
trainingOutputData = to_categorical(trainingOutputData)

validationInputData = np.array(validationData[:, 0:2]).reshape(1, 10000, 2)
validationOutputData = np.array(validationData[:, 2:]).reshape(1, 10000, 1)
validationOutputData = to_categorical(validationOutputData)
print("Finished importing data")

#Setup model

print(sample_weights)

model = Sequential()
model.add(LSTM(3, input_shape=(10000,2), return_sequences=True, stateful=True, batch_size=14))
for x in range(32):
    model.add(LSTM(100, return_sequences=True, stateful=True))
model.add(Dense(3, kernel_initializer='normal', activation='softmax'))

model.summary()

opt = Adam(lr=0.00000001)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['categorical_accuracy'], sample_weight_mode="temporal")
model.fit(trainingInputData, trainingOutputData, epochs=5, batch_size=14, verbose=1, shuffle=False, validation_split=0.1, sample_weight=sample_weights) # Because our input data is ordered we do not want to shuffle the validation data
model.save(sys.argv[2])


