from __future__ import absolute_import, division, print_function
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import tensorflow as tf 
import numpy as np


bc_data = pd.read_csv("data/breast_cancer_wisconsin.csv")

y = bc_data["diagnosis"]
bc_data = bc_data.drop("diagnosis", axis=1)
bc_data = bc_data.drop(bc_data.columns[-1], axis=1)

#train
ytrain = y[:int(len(y) * .7)]
bc_data_train = bc_data[:int(len(bc_data) * .7)]

#test 
ytest = y[int(len(y) * .7):]
bc_data_test = bc_data[int(len(bc_data) * .7):]


#create model
model = LogisticRegression().fit(bc_data_train, ytrain)


predictions = model.predict(bc_data_test)

print(predictions)
print(ytest)

#neural network
nn_model = tf.keras.Sequential([
	#input layer
	tf.keras.layers.Dense(len(bc_data_train.iloc[0]))
	#hidden layer
	tf.keras.layers.Dense(10, activation='relu')
	
])





