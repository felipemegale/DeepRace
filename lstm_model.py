import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utility import *

print('Loading data')
x, y, vocabulary, vocabulary_inv = load_data(avg_len=False, load_saved_data=False, load_testdata=False)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

sequence_length = x.shape[1]
print('sequence length = {}'.format(sequence_length))
vocabulary_size = len(vocabulary_inv)
embedding_dim = 32
filter_sizes = [3, 4, 5]
num_filters = 512
drop = 0.5
epochs = 16
batch_size = 4

# model = keras.Sequential()
# model.add(layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length))

# # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
# model.add(layers.GRU(256, return_sequences=True))

# model.add(layers.LSTM(128))

# model.add(layers.Dense(2))

# model.summary()

adam = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# print("X_train shape", X_train.shape)
# print("len( X_train[0] )", len( X_train[0] ))
# print("y_train shape", y_train.shape)
# print("len( y_train[0] )", len( y_train[0] ))
# model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)  # starts training

# # Storing the model for future use
# # serialize model to JSON
# model_json = model.to_json()
# with open("lstm_model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("lstm_model.h5")
# print("Saved model to disk")

# load json and create model
json_file = open('lstm_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("lstm_model.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

score = loaded_model.evaluate(X_test, y_test, verbose=1)
print("score",score)

print(len(file_names))

# for i, x in enumerate(file_names):
#     print("x",x)
#     print("i",i)
    #print(loaded_model.predict(X_test[i:i+1]))

y_pred = loaded_model.predict(X_test[12:30])
print("y_pred",y_pred)

from keras import backend as K
# with a Sequential model
get_3rd_layer_output = K.function([loaded_model.layers[0].input], [loaded_model.layers[1].output, 
                                                                   loaded_model.layers[2].output,
                                                                   loaded_model.layers[3].output])

sample = 10

# predict the sample
y_pred = loaded_model.predict(np.array([X_test[sample]]))
print('classifier prediction: {}'.format(y_pred))

predicted_label = 0 if (y_pred[0][0] > y_pred[0][1]) else 1
print('predicted label: {}'.format(predicted_label))

# getting the intermediate output
conv_output1 = (get_3rd_layer_output([np.array([X_test[sample]])])[0][0]).T
conv_output2 = (get_3rd_layer_output([np.array([X_test[sample]])])[1][0]).T
conv_output3 = (get_3rd_layer_output([np.array([X_test[sample]])])[2][0]).T
print("shape1",conv_output1.shape)
print("shape2",conv_output2.shape)
print("shape3",conv_output3.shape)
# conv_output_concat = np.concatenate((conv_output1, conv_output2, conv_output3))
# print(conv_output_concat.shape)

# get the weights of the last layer
# last_layer_w = loaded_model.get_weights()[-2]
# print(last_layer_w.shape) # (1536, 2)

# print(last_layer_w[1,1])
# print(conv_output_concat[1,:].shape)