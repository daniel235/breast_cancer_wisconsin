from __future__ import absolute_import, division, print_function
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import tensorflow as tf 
import numpy as np


bc_data = pd.read_csv("data/breast_cancer_wisconsin.csv")

y = bc_data["diagnosis"]
bc_data = bc_data.drop("diagnosis", axis=1)
bc_data = bc_data.drop(bc_data.columns[-1], axis=1)

for n in range(len(y)):
	if y[n] == "M":
		y[n] = 1
	else:
		y[n] = 0


print("y ", y)
#train
ytrain = y[:int(len(y) * .7)]
bc_data_train = bc_data[:int(len(bc_data) * .7)]

#test 
ytest = y[int(len(y) * .7):]
bc_data_test = bc_data[int(len(bc_data) * .7):]


#create model
#model = LogisticRegression().fit(bc_data_train, ytrain)

#predictions = model.predict(bc_data_test)


#neural network
nn_model = tf.keras.Sequential([
	#input layer
	tf.keras.layers.Dense(len(bc_data_train.iloc[0])),
	#hidden layer
	tf.keras.layers.Dense(10, activation='relu'),
	#second hidden layer
	tf.keras.layers.Dense(8, activation='relu'),
	#third hidden layer
	tf.keras.layers.Dense(8, activation='relu'),
	#output layer
	tf.keras.layers.Dense(2)
])

bc_train = bc_data_train.to_numpy()
predictions = nn_model(bc_train).numpy()

tf.nn.softmax(predictions).numpy()

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print(loss_function(ytrain, predictions).numpy())

#print(predictions)

nn_model.compile(optimizer='adam',
				loss=loss_function,
				metrics=['accuracy']
)

print(bc_train)
ytrain = ytrain.to_numpy().astype(np.float32)
print(ytrain)
nn_model.fit(bc_train, ytrain, epochs=50)





