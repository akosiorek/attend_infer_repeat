import os
import numpy as np
from scipy.misc import imresize

from tensorflow.python.util import nest
from tensorflow.examples.tutorials.mnist import input_data


_MNIST_PATH = os.path.join(os.path.dirname(__file__), 'MNIST_data')


def create_mnist(parition='train', canvas_size=(50, 50), obj_size=(20, 20), n_objects=(0, 2), n_samples=None,
             dtype=np.uint8, expand_nums=True):

    mnist = input_data.read_data_sets(_MNIST_PATH, one_hot=False)
    mnist_data = getattr(mnist, parition)

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

    for i in xrange(n_samples):
        n = nums[i]
        if n > 0:
            indices = np.random.choice(n_templates, n, replace=False)
            pos = np.round(np.random.rand(n, 2) * position_range[np.newaxis]).astype(np.int32)
            for j in xrange(n):
                idx = indices[j]
                p = pos[j]
                labels[i, j] = mnist_data.labels[idx]
                template = resize(templates[idx])
                imgs[i, p[0]:p[0]+obj_size[0], p[1]:p[1]+obj_size[1]] = template

    imgs = imgs.astype(np.float32) / 255.

    if expand_nums:
        expanded = np.zeros((max_objects + 1, n_samples, 1), dtype=np.uint8)
        for i, n in enumerate(nums):
            expanded[:n, i] = 1
        nums = expanded

    return dict(imgs=imgs, labels=labels, nums=nums)


if __name__ == '__main__':
    d = create_mnist(n_samples=10)
    for k, v in d.iteritems():
        print k, v.shape