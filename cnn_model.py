'''
16 epochs: Y_PRED [[0.5133009  0.48669901]]
24 epochs: Y_PRED [[0.5231215 0.4768785]]
28 epochs: Y_PRED [[0.52662987 0.47337016]]
32 epochs: Y_PRED [[0.5300586  0.46994132]]
48 epochs: Y_PRED [[0.41416934 0.5858306 ]] (overfitted at 37 epochs)
64 epochs: Y_PRED [[0.22040138 0.7795986 ]] (overfitted at 35 epochs)

original values:
embedding_dim = 32
filter_sizes = [3, 4, 5]
num_filters = 512
drop = 0.5
epochs = 16
batch_size = 4
'''
from random import randint
from keras.layers import Input, Dense, Embedding, Conv1D
from keras.layers import Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling1D
from tensorflow.keras.optimizers import Adam
from keras.models import Model, model_from_json
from sklearn.model_selection import train_test_split
import os
import numpy as np

from utility import *

cwd = os.getcwd()

def relu_fun(list_):
    return (abs(list_) + list_) / 2

params = {
    'embedding_dim':[16,32,64],
    'filter_sizes': [[1,2,3],[2,3,4],[3,4,5]],
    'num_filters': [128,256,512],
    'drop': [0.1,0.25,0.5],
    'epochs': [16,24],
    'batch_size': [2,4,8]
}

datasets=["OMP_Critical", "OMP_Private", "POSIX"]
logger = setup_logger(f'orig_model', f'orig_model')

for folder_name in datasets:
    logger.info(f'{folder_name}: Loading data')
    x, y, vocabulary, vocabulary_inv = load_data(avg_len=False, load_saved_data=False, load_testdata=False, folder_name=folder_name)
    # X_test = x
    # y_test = y
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=randint(1,100))
    # 64, [3, 4, 5], 512, 0.25, 24, 4
    sequence_length = x.shape[1]
    logger.info(f'{folder_name}: sequence length = {sequence_length}')
    vocabulary_size = len(vocabulary_inv)
    embedding_dim = 64
    filter_sizes = [3,4,5]
    num_filters = 512
    drop = 0.25
    epochs = 24
    batch_size = 4

    # this returns a tensor
    logger.info(f"{folder_name}: Creating Model")
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
    adam = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # Model summary
    logger.info(f'{folder_name} model summary:')
    logger.info(model.summary())

    logger.info(f"{folder_name}: Traning Model")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)  # starts training

    # Storing the model for future use
    # serialize model to JSON
    model_json = model.to_json()
    with open(f"cnn_model_{folder_name}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"cnn_model_{folder_name}.h5")
    logger.info(f"{folder_name}: Saved model to disk")

    # load json and create model
    json_file = open(f'cnn_model_{folder_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f"cnn_model_{folder_name}.h5")
    logger.info(f"{folder_name}: Loaded model from disk")

    loaded_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    score = loaded_model.evaluate(X_test, y_test, verbose=1)
    logger.info(f"{folder_name}: EVALUATE: {score}")

    logger.info(f"{folder_name}: LEN FILE NAMES: {len(file_names)}")

    # for i, x in enumerate(file_names):
    #     print(x)
    #     print(i)
        # print(loaded_model.predict(X_test[i:i+1]))

    y_pred = loaded_model.predict(X_test[12:30])
    logger.info(f"{folder_name} => Y_PRED: {y_pred}")

    # Import the modules from `sklearn.metrics`
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

    # Confusion matrix
    logger.info(f"{folder_name}: confusion matrix:")
    logger.info(confusion_matrix(y_test[:, 0], y_pred[:, 0]))
    # Precision 
    logger.info(f"{folder_name}: precision:")
    logger.info(precision_score(y_test[:, 0], y_pred[:, 0]))
    # Recall
    logger.info(f"{folder_name}: recall:")
    logger.info(recall_score(y_test[:, 0], y_pred[:, 0]))
    # F1 score
    logger.info(f"{folder_name}: f1:")
    logger.info(f1_score(y_test[:, 0], y_pred[:, 0]))
    # Cohen's kappa
    logger.info(f"{folder_name}: cohen kappa:")
    logger.info(cohen_kappa_score(y_test, y_pred))

    y_pred[:, :] = y_pred[:, :] > 0.5
    cm = confusion_matrix(y_test[:, 0], y_pred[:, 0])
    logger.info(f"{folder_name}: confusion matrix:")
    logger.info(cm)


    from keras import backend as K
    # with a Sequential model
    get_3rd_layer_output = K.function([loaded_model.layers[0].input], [loaded_model.layers[2].output, 
                                                                    loaded_model.layers[3].output,
                                                                    loaded_model.layers[4].output])

    logger.info(f'{folder_name} loaded model layers[5] output')
    logger.info(loaded_model.layers[5].output)
    sample = 10

    # predict the sample
    y_pred = loaded_model.predict(np.array([X_test[sample]]))
    logger.info(f'{folder_name}: y_pred: {y_pred}')

    predicted_label = 0 if (y_pred[0][0] > y_pred[0][1]) else 1
    logger.info(f'{folder_name}: predicted label: {predicted_label}')


    # getting the intermediate output
    conv_output1 = (get_3rd_layer_output([np.array([X_test[sample]])])[0][0]).T
    conv_output2 = (get_3rd_layer_output([np.array([X_test[sample]])])[1][0]).T
    conv_output3 = (get_3rd_layer_output([np.array([X_test[sample]])])[2][0]).T
    logger.info(f"{folder_name}: shape1: {conv_output1.shape}")
    logger.info(f"{folder_name}: shape2: {conv_output2.shape}")
    logger.info(f"{folder_name}: shape3: {conv_output3.shape}")
    conv_output_concat = np.concatenate((conv_output1, conv_output2, conv_output3))
    # print(conv_output_concat.shape)

    # get the weights of the last layer
    last_layer_w = loaded_model.get_weights()[-2]
    # print(last_layer_w.shape) # (1536, 2)

    # print(last_layer_w[1,1])
    # print(conv_output_concat[1,:].shape)

    # generate the heat map
    heat_map = np.zeros(sequence_length)

    for i in range(num_filters * len(filter_sizes)):
        heat_map += last_layer_w[i,predicted_label] * conv_output_concat[i,:] # here should be the convolutional layer that this wieght related to the maxpooling value related to it.

    # show the heat map
    # print(relu_fun(heat_map).tolist()[:500])

    # keep track of the line of token in AST and also its impact probability
    ast_intrst_lines = {}

    nums = relu_fun(heat_map).tolist()
    for i in range(len(nums)):
        if nums[i] > 0:
            ast_intrst_lines[i] = nums[i]
            # print(str(i) + '\n')
            
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

    logger.info(f"{folder_name}: Printing Softmax")
    logger.info(softmax(relu_fun(heat_map)).tolist()[:500])
