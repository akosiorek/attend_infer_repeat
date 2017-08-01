import os
import numpy as np
import itertools
import cPickle as pickle

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.examples.tutorials.mnist import input_data

from scipy.misc import imresize


_MNIST_PATH = os.path.join(os.path.dirname(__file__), 'MNIST_data')


def create_mnist(partition='train', canvas_size=(50, 50), obj_size=(20, 20), n_objects=(0, 2), n_samples=None,
                 dtype=np.uint8, expand_nums=True, with_overlap=False):

    mnist = input_data.read_data_sets(_MNIST_PATH, one_hot=False)
    mnist_data = getattr(mnist, partition)

    n_templates = mnist_data.num_examples
    if n_samples is None:
        n_samples = n_templates

    n_objects = nest.flatten(n_objects)
    n_objects.sort()
    max_objects = n_objects[-1]

    imgs = np.zeros((n_samples,) + tuple(canvas_size), dtype=dtype)
    labels = np.zeros((n_samples, n_objects[-1]), dtype=np.uint8)
    nums = np.random.randint(max_objects + 1, size=n_samples, dtype=np.uint8)

    templates = np.reshape(mnist_data.images, (-1, 28, 28))
    resize = (lambda x: x) if templates.shape[1:] == obj_size else (lambda x: imresize(x, obj_size))
    obj_size = np.asarray(obj_size, dtype=np.int32)
    position_range = np.asarray(canvas_size) - obj_size
    make_p = lambda: np.round(np.random.rand(n) * position_range).astype(np.int32)

    occupancy = np.zeros(canvas_size, dtype=bool)

    i = 0
    n_tries = 5
    while i < n_samples:
        tries = 0
        retry = False
        n = nums[i]
        if n > 0:
            indices = np.random.choice(n_templates, n, replace=False)

            occupancy[...] = False
            for j in xrange(n):
                idx = indices[j]
                labels[i, j] = mnist_data.labels[idx]
                template = resize(templates[idx])

                p = make_p()
                if not with_overlap:
                    while occupancy[p[0]:p[0]+obj_size[0], p[1]:p[1]+obj_size[1]].any() and tries < n_tries:
                        p = make_p()
                        tries += 1
                    if tries == n_tries:
                        retry = True
                        break

                imgs[i, p[0]:p[0]+obj_size[0], p[1]:p[1]+obj_size[1]] = template
                occupancy[p[0]:p[0]+obj_size[0], p[1]:p[1]+obj_size[1]] = True

        if not retry:
            i += 1
        else:
            imgs[i, ...] = 0.

    if expand_nums:
        expanded = np.zeros((max_objects + 1, n_samples, 1), dtype=np.uint8)
        for i, n in enumerate(nums):
            expanded[:n, i] = 1
        nums = expanded

    return dict(imgs=imgs, labels=labels, nums=nums)


def load_data(path):
    with open(path) as f:
        data = pickle.load(f)

    data['imgs'] = data['imgs'].astype(np.float32) / 255.
    data['nums'] = data['nums'].astype(np.float32)
    return data


def tensors_from_data(data_dict, batch_size, axes=None, shuffle=False):
    keys = data_dict.keys()
    if axes is None:
        axes = {k: 0 for k in keys}

    key = keys[0]
    ax = axes[key]
    n_entries = data_dict[key].shape[ax]

    if shuffle:
        def idx_fun():
            return np.random.choice(n_entries, batch_size)

    else:

        def idx_fun():
            start = next(itertools.cycle(xrange(0, n_entries, batch_size)))
            end = start + batch_size
            return np.arange(start, end)

    def data_fun():
        idx = idx_fun()
        minibatch = []
        for k in keys:
            item = data_dict[k]
            minibatch_item = item.take(idx, axes[k])
            minibatch.append(minibatch_item)
        return minibatch

    minibatch = data_fun()
    types = [getattr(tf, str(m.dtype)) for m in minibatch]

    tensors = tf.py_func(data_fun, [], types)
    for t, m in zip(tensors, minibatch):
        t.set_shape(m.shape)

    tensors = {k: v for k, v in zip(keys, tensors)}
    return tensors


if __name__ == '__main__':
    partitions = ['train', 'test']
    nums = [60000, 1000]

    for p, n in zip(partitions, nums):
        data = create_mnist(p, n_samples=n)
        filename = 'mnist_{}.pickle'.format(p)
        with open(filename, 'w') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)