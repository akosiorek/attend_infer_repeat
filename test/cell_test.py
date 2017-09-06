import unittest
import tensorflow as tf
import sonnet as snt

from attend_infer_repeat.cell import AIRCell
from attend_infer_repeat.modules import *


def make_modules():

    return dict(
        transition=snt.GRU(3),
        input_encoder=(lambda: Encoder(5)),
        glimpse_encoder=(lambda: Encoder(7)),
        glimpse_decoder=(lambda x: Decoder(11, x)),
        transform_estimator=(lambda x: StochasticTransformParam(13, x)),
        steps_predictor=(lambda: StepsPredictor(17))
    )


class CellTest(unittest.TestCase):

    def test_instantiate(self):
        learning_rate = 1e-4
        batch_size = 10
        img_size = (3, 3)
        crop_size = (2, 2)
        n_latent = 10
        n_steps = 3

        x = tf.placeholder(tf.float32, (batch_size,) + img_size, name='inpt')

        # transition = snt.GRU(n_latent)
        modules = make_modules()
        air = AIRCell(img_size, crop_size, n_latent, **modules)
        initial_state = air.initial_state(x)

        dummy_sequence = tf.zeros((n_steps, batch_size, 1), name='dummy_sequence')
        outputs, state = tf.nn.dynamic_rnn(air, dummy_sequence, initial_state=initial_state, time_major=True)
        canvas, crop, what, what_loc, what_scale, where, where_loc, where_scale, presence_prob, presence = outputs

        canvas = tf.reshape(canvas, (n_steps, batch_size,) + tuple(img_size))
        final_canvas = canvas[-1]

        loss = tf.nn.l2_loss(x - final_canvas)

        opt = tf.train.AdamOptimizer(learning_rate)
        train_step = opt.minimize(loss)

        print 'Constructed model'

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        xx = np.random.rand(*x.get_shape().as_list())
        res, l = sess.run([outputs, loss], {x: xx})

        for r in res:
            print r.shape

        print res

        print 'loss = {}'.format(l)
        print 'Done'