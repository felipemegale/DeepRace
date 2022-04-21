'''
Y_PRED [[0.8921792 0.1078208]]
'''
from operator import mod
from random import randint
from keras.layers import Input, Dense, Embedding, Conv1D
from keras.layers import Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
import os
import numpy as np

from utility import *

cwd = os.getcwd()

def relu_fun(list_):
    return (abs(list_) + list_) / 2

datasets=["OMP_Critical", "OMP_Private", "POSIX"]

for folder_name in datasets:
    logger = setup_logger(f'{folder_name}_seq_model', f'{folder_name}_seq_model')
    logger.info(f'Loading data from {folder_name}')
    x, y, vocabulary, vocabulary_inv = load_data(avg_len=False, load_saved_data=False, load_testdata=False, folder_name=folder_name)
    # X_test = x
    # y_test = y
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=randint(1,100))

    #print(X_train.shape)
    #sys.exit("Breaking")
    sequence_length = x.shape[1]
    logger.info('sequence length = {}'.format(sequence_length))
    vocabulary_size = len(vocabulary_inv)
    logger.info("vocabulary size",vocabulary_size)
    embedding_dim = 32
    filter_sizes = [3, 4, 5]
    num_filters = 512
    drop = 0.5
    epochs = 24
    batch_size = 4
    # this returns a tensor
    '''
    #--------------------------------------------------------------------------------------------------------------------
    print("Creating Model...")
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)

    conv_list = list()
    for ftr in filter_sizes:
        conv = Conv1D(num_filters, kernel_size=ftr, padding='same', kernel_initializer='normal', activation='relu')(embedding)
        maxpool = MaxPooling1D(pool_size=(sequence_length - ftr + 1), padding='valid')(conv)
        conv_list.append(maxpool)

    concatenated_tensor = Concatenate(axis=1)(conv_list)

    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=2, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)

    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # Model summary
    print(model.summary())
    '''
    #--------------------------------------------------------------------------------------------------------------------
    logger.info("Creating Model...")

    model = models.Sequential()
    model.add(Input(shape=(sequence_length,), dtype='int32'))
    model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length))
    model.add(Conv1D(64, kernel_size=3, padding='same', kernel_initializer='normal', activation='relu'))
    model.add(MaxPooling1D(pool_size=3, padding='valid'))
    model.add(Conv1D(128, kernel_size=4, padding='same', kernel_initializer='normal', activation='relu'))
    model.add(MaxPooling1D(pool_size=4, padding='valid'))
    model.add(Conv1D(256, kernel_size=5, padding='same', kernel_initializer='normal', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, padding='valid'))
    #model.add(Conv1D(128, kernel_size=6, padding='same', kernel_initializer='normal', activation='relu'))
    #model.add(MaxPooling1D(pool_size=6, padding='valid'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(units=256,activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))
    #inputs = Input(shape=(sequence_length,), dtype='int32')
    #embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    '''
    conv_list = list()
    for ftr in filter_sizes:
        conv = Conv1D(num_filters, kernel_size=ftr, padding='same', kernel_initializer='normal', activation='relu')(embedding)
        maxpool = MaxPooling1D(pool_size=(sequence_length - ftr + 1), padding='valid')(conv)
        conv_list.append(maxpool)

    concatenated_tensor = Concatenate(axis=1)(conv_list)

    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=2, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)
    '''
    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    adam = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # Model summary
    logger.info(model.summary())
    #sys.exit("TESTING")


    logger.info("Traning Model...")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)  # starts training


    # Storing the model for future use
    # serialize model to JSON
    model_json = model.to_json()
    with open(f"cnn_sequential_model_{folder_name}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"cnn_sequential_model_{folder_name}.h5")
    logger.info("Saved model to disk")

    # load json and create model
    json_file = open(f'cnn_sequential_model_{folder_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f"cnn_sequential_model_{folder_name}.h5")
    logger.info("Loaded model from disk")

    loaded_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    score = loaded_model.evaluate(X_test, y_test, verbose=1)
    logger.info("EVALUATE",score)

    logger.info("LEN FILE NAMES", len(file_names))

    # for i, x in enumerate(file_names):
    #     print(x)
    #     print(i)
    #     print(loaded_model.predict(X_test[i:i+1]))

    y_pred = loaded_model.predict(X_test[12:30])
    logger.info("Y_PRED",y_pred)

    # Import the modules from `sklearn.metrics`
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

    # Confusion matrix
    logger.info("confusion matrix:")
    logger.info(confusion_matrix(y_test[:, 0], y_pred[:, 0]))
    # Precision 
    logger.info("precision:")
    logger.info(precision_score(y_test[:, 0], y_pred[:, 0]))
    # Recall
    logger.info("recall:")
    logger.info(recall_score(y_test[:, 0], y_pred[:, 0]))
    # F1 score
    logger.info("f1:")
    logger.info(f1_score(y_test[:, 0], y_pred[:, 0]))
    # Cohen's kappa
    logger.info("cohen kappa:")
    logger.info(cohen_kappa_score(y_test, y_pred))

    y_pred[:, :] = y_pred[:, :] > 0.5
    cm = confusion_matrix(y_test[:, 0], y_pred[:, 0])
    logger.info("confusion matrix:")
    logger.info(cm)


    from keras import backend as K
    # with a Sequential model
    get_3rd_layer_output = K.function([loaded_model.layers[0].input], [loaded_model.layers[5].output])
    logger.info(loaded_model.layers[5].output)
    sample = 10

    # predict the sample
    y_pred = loaded_model.predict(np.array([X_test[sample]]))
    logger.info('classifier prediction: {}'.format(y_pred))

    predicted_label = 0 if (y_pred[0][0] > y_pred[0][1]) else 1
    logger.info('predected label: {}'.format(predicted_label))


    # getting the intermediate output
    conv_output1 = (get_3rd_layer_output([np.array([X_test[sample]])])[0][0]).T
    #conv_output2 = (get_3rd_layer_output([np.array([X_test[sample]])])[1][0]).T
    #conv_output3 = (get_3rd_layer_output([np.array([X_test[sample]])])[2][0]).T
    # print(conv_output1.shape)
    # print(conv_output2.shape)
    # print(conv_output3.shape)
    #conv_output_concat = np.concatenate((conv_output1, conv_output2, conv_output3))
    # print(conv_output_concat.shape)

    # get the weights of the last layer
    last_layer_w = loaded_model.get_weights()[-2]
    # print(last_layer_w.shape) # (1536, 2)

    # print(last_layer_w[1,1])
    # print(conv_output_concat[1,:].shape)

    # generate the heat map
    heat_map = np.zeros(sequence_length)

    #for i in range(num_filters * len(filter_sizes)):
    #    heat_map += last_layer_w[i,predicted_label] * conv_output_concat[i,:] # here should be the convolutional layer that this wieght related to the maxpooling value related to it.

    heat_map = last_layer_w[i, predicted_label] * conv_output1[i,:]

    # show the heat map
    logger.info(relu_fun(heat_map).tolist()[:500])


    # keep track of the line of token in AST and also its impact probability
    ast_intrst_lines = {}

    nums = relu_fun(heat_map).tolist()
    for i in range(len(nums)):
        if nums[i] > 0:
            ast_intrst_lines[i] = nums[i]
            logger.info(str(i) + '\n')
            
    # reading the untouched AST 
    ast = read_ast(file_names[sample])

    #this dictionary keeps the source code line and its impact value on classification
    source_lines_probability = create_source_lines_values(ast, ast_intrst_lines, method="maximum")
    highlight_source_code(file_names[sample], source_lines_probability, method="MAX")

    source_lines_probability = create_source_lines_values(ast, ast_intrst_lines, method="average")
    highlight_source_code(file_names[sample], source_lines_probability, method="AVG")

    source_lines_probability = create_source_lines_values(ast, ast_intrst_lines, method="only_parent")
    highlight_source_code(file_names[sample], source_lines_probability, method="PARENT")


    def softmax(x):
        '''
        Compute softmax values for each sets of scores in x.
        '''
        return np.exp(x) / np.sum(np.exp(x), axis=0) * 1000


    logger.info(softmax(relu_fun(heat_map)).tolist()[:500])