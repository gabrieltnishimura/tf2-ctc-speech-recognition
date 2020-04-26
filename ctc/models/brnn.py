from tensorflow.keras.layers import Input, TimeDistributed, Dense, Dropout
from tensorflow.keras.layers import Bidirectional, SimpleRNN, Lambda, Masking
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

# Architecture from Baidu Deep speech: Scaling up end-to-end speech recognition (https://arxiv.org/pdf/1412.5567.pdf)


def brnn(units, input_dim=26, output_dim=29, dropout=0.2, numb_of_dense=3,
         n_layers=1):
    """
    :param units: Hidden units per layer
    :param input_dim: Size of input dimension (number of features), default=26
    :param output_dim: Output dim of final layer of model (input to CTC layer), default=29
    :param dropout: Dropout rate, default=0.2
    :param numb_of_dense: Number of fully connected layers before recurrent, default=3
    :param n_layers: Number of bidirectional recurrent layers, default=1
    :return: network_model: brnn

    Default model contains:
     1 layer of masking
     3 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     1 layer of BRNN
     1 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     1 layer of softmax
    """

    # Input data type
    dtype = 'float32'
    # Kernel and bias initializers for fully connected dense layers
    kernel_init_dense = 'random_normal'
    bias_init_dense = 'random_normal'

    # Kernel and bias initializers for recurrent layer
    kernel_init_rnn = 'glorot_uniform'
    bias_init_rnn = 'zeros'

    # ---- Network model ----
    # x_input layer, dim: (batch_size * x_seq_size * features)
    input_data = Input(name='the_input', shape=(None, input_dim), dtype=dtype)

    # Masking layer
    x = Masking(mask_value=0., name='masking')(input_data)

    # Default 3 fully connected layers DNN ReLu
    # Default dropout rate 20 % at each FC layer
    for i in range(0, numb_of_dense):
        x = TimeDistributed(Dense(units=units,
                                  kernel_initializer=kernel_init_dense,
                                  bias_initializer=bias_init_dense,
                                  activation=clipped_relu), name='fc_'+str(i+1))(x)
        x = TimeDistributed(Dropout(dropout), name='dropout_'+str(i+1))(x)

    # Bidirectional RNN (with ReLu)
    for i in range(0, n_layers):
        x = Bidirectional(SimpleRNN(units, activation='relu',
                                    kernel_initializer=kernel_init_rnn,
                                    dropout=0.2,
                                    bias_initializer=bias_init_rnn,
                                    return_sequences=True),
                          merge_mode='concat', name='bi_rnn'+str(i+1))(x)

    # 1 fully connected layer DNN ReLu with default 20% dropout
    x = TimeDistributed(Dense(units=units,
                              kernel_initializer=kernel_init_dense,
                              bias_initializer=bias_init_dense,
                              activation='relu'), name='fc_4')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_4')(x)

    # Output layer with softmax
    y_pred = TimeDistributed(Dense(units=output_dim,
                                   kernel_initializer=kernel_init_dense,
                                   bias_initializer=bias_init_dense,
                                   activation='softmax'), name='softmax')(x)

    # ---- CTC ----
    # y_input layers (transcription data) for CTC loss
    # transcription data (batch_size * y_seq_size)
    labels = Input(name='the_labels', shape=[None], dtype=dtype)
    # unpadded len of all x_sequences in batch
    input_length = Input(name='input_length', shape=[1], dtype=dtype)
    # unpadded len of all y_sequences in batch
    label_length = Input(name='label_length', shape=[1], dtype=dtype)

    # Lambda layer with ctc_loss function due to Keras not supporting CTC layers
    loss_out = Lambda(function=ctc_lambda_func,
                      name='ctc',
                      output_shape=(1,))(
        [y_pred, labels, input_length, label_length])

    network_model = Model(
        inputs=[input_data, labels, input_length, label_length],
        outputs=loss_out)

    return network_model


def ctc_lambda_func(args):
    """Lambda implementation of CTC loss, using ctc_batch_cost from TensorFlow backend
    CTC implementation from Keras example found at https://github.com/keras-team/keras/blob/master/examples/image_ocr.py"""
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # print "y_pred_shape: ", y_pred.shape
    y_pred = y_pred[:, 2:, :]
    # print "y_pred_shape: ", y_pred.shape
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def clipped_relu(value):
    """Returns clipped relu, clip value set to 20."""
    return K.relu(value, max_value=20)
