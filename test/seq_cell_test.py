import unittest
import tensorflow as tf
import sonnet as snt

from attrdict import AttrDict

from attend_infer_repeat.seq_model import SeqAIRCell
from attend_infer_repeat.modules import *


def make_modules():

    return dict(
        transition=snt.GRU(3),
        input_encoder=(lambda: Encoder(5)),
        glimpse_encoder=(lambda: Encoder(7)),
        transform_estimator=(lambda x: StochasticTransformParam(13, x)),
        steps_predictor=(lambda: StepsPredictor(17))
    )


class SeqCellTest(unittest.TestCase):

    def test_instantiate(self):
        learning_rate = 1e-4
        batch_size = 10
        img_size = (5, 7)
        crop_size = (2, 2)
        n_what = 13
        n_steps_per_image = 3
        n_timesteps = 1

        x = tf.placeholder(tf.float32, (n_timesteps, batch_size,) + img_size, name='inpt')

        modules = make_modules()
        air = SeqAIRCell(n_steps_per_image, img_size, crop_size, n_what, **modules)
        initial_state = air.initial_state(x[0])

        outputs, state = tf.nn.dynamic_rnn(air, x, initial_state=initial_state, time_major=True)
        outputs = AttrDict({k: v for k, v in zip(air.output_names, outputs)})

        print 'test:'
        for k, v in outputs.iteritems():
            print k, v.get_shape().as_list()


        loss = sum(map(tf.reduce_mean, outputs.values()))

        opt = tf.train.AdamOptimizer(learning_rate)
        train_step = opt.minimize(loss)

        print 'Constructed model'

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        xx = np.random.rand(*x.get_shape().as_list())
        res, l = sess.run([outputs, loss], {x: xx})

        for k, v in res.iteritems():
            print k, v.shape

        print res

        print 'loss = {}'.format(l)
        print 'Done'