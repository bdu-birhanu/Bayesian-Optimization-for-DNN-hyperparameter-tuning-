#new deephyper to select the best hyper-parameter with 10% of my test data
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Permute, Flatten, Masking, \
    GaussianNoise, Reshape, Lambda, TimeDistributed, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, MaxPool2D
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
from data_loader import num_class, maxlen
from data_loader import x_train, y_train, x_train_length, y_train_length, x_val, y_val, x_val_length, y_val_length

# import matplotlib.pyplot as plt

def ctc_lambda_func(args):
    '''

    '''
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def hist_ocr_model(config: dict, n_components: int = 5, verbose: bool = 0, num_class=310, img_row=32, img_col=200):
    tf.keras.utils.set_random_seed(2)

    default_config = {
        "rnn_size": 128,
        "feature_map_1": 64,
        "feature_map_2": 128,
        "activation": "relu",
        "batch_size": 32,
        "epoch": 10,
        "kernel": 3,
        "dropout": 0.25,

    }
    default_config.update(config)
    '''
    if you use the full datset you could increase the batch_size and epo
    '''
    k = 3
    inputs_data = Input(shape=(img_row, img_col, 1))
    conv_1 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(inputs_data)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 1), strides=2)(conv_1)
    conv_2 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 1))(conv_2)  # we remove the strides here
    conv_3 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(pool_2)
    conv_4 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_3 = MaxPool2D(pool_size=(2, 1))(conv_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(pool_3)
    conv_5 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(batch_norm_5)

    conv_6 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(conv_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    conv_7 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"])(batch_norm_6)

    # conv2=(int(conv1[2]),int(conv1[1]*int(conv1[3])))
    r = Reshape((int(conv_7.shape[2]), int(conv_7.shape[1]) * int(conv_7.shape[3])))(conv_7)
    # [ sample, timesteps, features]
    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(default_config["rnn_size"], return_sequences=True, dropout=0.25))(r)
    blstm_2 = Bidirectional(LSTM(default_config["rnn_size"], return_sequences=True, dropout=0.25))(blstm_1)
    outputs = Dense(num_class + 1, activation='softmax')(blstm_2)

    pred_model = Model(inputs_data, outputs)

    labels = Input(name='the_labels', shape=[46], dtype='float32')  # 46 is the max size of text length
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

    model = Model(inputs=[inputs_data, labels, input_length, label_length], outputs=loss_out)
    # lrate = 0.01
    # decay = lrate / epoch
    # sgd = SGD(learning_rate=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    hist = model.fit(x=[x_train, y_train, x_train_length, y_train_length], y=np.zeros(len(y_train)),
                     batch_size=default_config["batch_size"], epochs=default_config["epoch"],
                     validation_data=([x_val, y_val, x_val_length, y_val_length], [np.zeros(len(y_val))]),
                     verbose=1)

    return model, pred_model, hist

# from deephyper.problem import HpProblem
#
# problem = HpProblem()
# problem.add_hyperparameter((10, 256), "rnn_size", default_value=128)
# problem.add_hyperparameter(["sigmoid", "tanh", "relu"], "activation", default_value="relu")
# problem.add_hyperparameter((2, 4), "batch_size", default_value=2)
# problem.add_hyperparameter((10, 100), "epochs", default_value=20)
# problem


# def run(config):
#     # important to avoid memory exploision
#     tf.keras.backend.clear_session()
#
#     _, _, hist = hist_ocr_model(config, n_components=5, verbose=0)
#
#     return -hist.history["val_loss"][-1]
#
#
# from deephyper.search.hps import CBO
#
# search = CBO(problem, run, initial_points=[problem.default_configuration], log_dir="cbo-results", random_state=2)
# results = search.search(max_evals=4)
#
# i_max = results.objective.argmax()
# best_config = results.iloc[i_max][:-4].to_dict()
# best_config
with open('./model/best_hyp_615k.txt') as f:
    data = f.read()
best_config = eval(data)# toget dictionary from the string which is save in the isd

best_model,best_pred_model, best_history = hist_ocr_model(best_config, n_components=5, verbose=1)

best_pred_model.save('./model/tr1_25_BBO_615k.hdf5')
np.save('./model/tr1_history1_25_BBO_615k.npy', best_history.history)
print("Training is successfully completed and now your model  and history are stored to your disk")
print("optimized and best model is successfully completed and now your models are stored to your disk")