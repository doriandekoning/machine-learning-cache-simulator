import numpy as np
import pandas
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense,Activation, LSTM
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
if len(sys.argv) < 3:
    print("Not enough arguments, format is: [input_csv] [model]")
    exit(1)


#Import data
df = pandas.read_csv(sys.argv[1])
df = df.replace("r", 0)
df = df.replace("w", 1)
df = df.replace("nop", 2)
print(df)
allData =    df.values[0:140000]
inputData = np.array(allData[: , 0:2]).reshape(14, 10000, 2)
outputData = np.array(allData[: , 2:]).reshape(14, 10000, 1)

outputData = to_categorical(outputData)


print("Finished importing data")

#Setup model

model = load_model(sys.argv[2])
result = model.predict_classes(inputData, batch_size=14)
#results = model.evaluate(inputData, outputData, batch_size=14)
#print(results)
print(result)
print("AND THIS")
print(outputData)

#print(confusion_matrix(outputData.argmax(axis=1) , result.argmax(axis=1) ))
#print(accuracy_score(outputData, result, normalize=False))
