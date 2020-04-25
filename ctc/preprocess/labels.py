import numpy as np
import tensorflow as tf
from utils.text_utils import text_to_int_sequence


def switch(character):
    return tf.case([
        (tf.equal(character, tf.constant(' ')), lambda: tf.constant(0)),
        (tf.equal(character, tf.constant('a')), lambda: tf.constant(1)),
        (tf.equal(character, tf.constant('b')), lambda: tf.constant(2)),
        (tf.equal(character, tf.constant('c')), lambda: tf.constant(3)),
        (tf.equal(character, tf.constant('d')), lambda: tf.constant(4)),
        (tf.equal(character, tf.constant('e')), lambda: tf.constant(5)),
        (tf.equal(character, tf.constant('f')), lambda: tf.constant(6)),
        (tf.equal(character, tf.constant('g')), lambda: tf.constant(7)),
        (tf.equal(character, tf.constant('h')), lambda: tf.constant(8)),
        (tf.equal(character, tf.constant('i')), lambda: tf.constant(9)),
        (tf.equal(character, tf.constant('j')), lambda: tf.constant(10)),
        (tf.equal(character, tf.constant('k')), lambda: tf.constant(11)),
        (tf.equal(character, tf.constant('l')), lambda: tf.constant(12)),
        (tf.equal(character, tf.constant('m')), lambda: tf.constant(13)),
        (tf.equal(character, tf.constant('n')), lambda: tf.constant(14)),
        (tf.equal(character, tf.constant('o')), lambda: tf.constant(15)),
        (tf.equal(character, tf.constant('p')), lambda: tf.constant(16)),
        (tf.equal(character, tf.constant('q')), lambda: tf.constant(17)),
        (tf.equal(character, tf.constant('r')), lambda: tf.constant(18)),
        (tf.equal(character, tf.constant('s')), lambda: tf.constant(19)),
        (tf.equal(character, tf.constant('t')), lambda: tf.constant(20)),
        (tf.equal(character, tf.constant('u')), lambda: tf.constant(21)),
        (tf.equal(character, tf.constant('v')), lambda: tf.constant(22)),
        (tf.equal(character, tf.constant('w')), lambda: tf.constant(23)),
        (tf.equal(character, tf.constant('x')), lambda: tf.constant(24)),
        (tf.equal(character, tf.constant('y')), lambda: tf.constant(25)),
        (tf.equal(character, tf.constant('z')), lambda: tf.constant(26)),
        (tf.equal(character, tf.constant('\'')), lambda: tf.constant(27)),
    ], exclusive=True)


def test(label):
    splitted = tf.strings.bytes_split(label)
    mapped = tf.map_fn(switch, splitted, dtype=(tf.int32))
    tf.print(mapped)
    return mapped


def process(labels):
    """
    Converts and pads transcripts from text to int sequences
    :param labels: transcripts
    :return: y_data: numpy array with transcripts converted to a sequence of ints and zero-padded
             label_length: numpy array with length of each sequence before padding
    """

    len_y_seq = []
    y = []

    tf.print(labels)
    y.append(tf.map_fn(test, labels, dtype=(tf.int32)))
    tf.print(y)

    # for i in range(0, len(labels)):
    #     y_int = text_to_int_sequence(labels[i])
    #     len_y_seq.append(len(y_int))
    #     y.append(y_int)

    y = tf.keras.preprocessing.sequence.pad_sequences(
        y, dtype='float', padding='post', truncating='post')

    label_length = np.array(len_y_seq)

    return y, label_length
