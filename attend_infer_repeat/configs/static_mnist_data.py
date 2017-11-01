import tensorflow as tf
from attrdict import AttrDict

from attend_infer_repeat.data import load_data as _load_data, tensors_from_data as _tensors

flags = tf.flags

tf.flags.DEFINE_string('train_path', 'mnist_train.pickle', '')
tf.flags.DEFINE_string('valid_path', 'mnist_validation.pickle', '')

axes = {'imgs': 0, 'labels': 0, 'nums': 1}


def load(batch_size):

    f = tf.flags.FLAGS

    valid_data = _load_data(f.valid_path)
    train_data = _load_data(f.train_path)

    train_tensors = _tensors(train_data, batch_size, axes, shuffle=True)
    valid_tensors = _tensors(valid_data, batch_size, axes, shuffle=False)

    data_dict = AttrDict(
        train_img=train_tensors['imgs'],
        valid_img=valid_tensors['imgs'],
        train_num=train_tensors['nums'],
        valid_num=valid_tensors['nums'],
        train_tensors=train_tensors,
        valid_tensors=valid_tensors,
        train_data=train_data,
        valid_data=valid_data,
        axes=axes
    )

    return data_dict