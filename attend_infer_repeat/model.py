import numpy as  np
import tensorflow as tf
import sonnet as snt


class TransformParam(snt.AbstractModule):

    def __init__(self, n_param):
        super(TransformParam, self).__init__(self.__class__.__name__)
        self._n_param = n_param

    def _build(self, inpt):

        flat = snt.BatchFlatten()
        linear = snt.Linear(self._n_param)
        return linear(flat(inpt))


class ConvEncoder(snt.AbstractModule):
    def __init__(self):
        super(ConvEncoder, self).__init__(self.__class__.__name__)

    def _build(self, inpt):

        if len(inpt.get_shape().as_list()) == 3:
            inpt = inpt[..., tf.newaxis]

        conv1 = snt.Conv2D(32, (3, 3), 2)
        conv2 = snt.Conv2D(32, (3, 3), 2)
        seq = snt.Sequential((conv1, tf.nn.elu, conv2, tf.nn.elu))
        return seq(inpt)


class Encoder(snt.AbstractModule):

    def __init__(self, n_latent):
        super(Encoder, self).__init__(self.__class__.__name__)
        self._n_latent = n_latent

    def _build(self, inpt):
        flat = snt.BatchFlatten()
        linear1 = snt.Linear(100)
        linear2 = snt.Linear(self._n_latent)
        seq = snt.Sequential([flat, linear1, tf.nn.elu, linear2, tf.nn.elu])
        return seq(inpt)


class Decoder(snt.AbstractModule):

    def __init__(self, output_size):
        super(Decoder, self).__init__(self.__class__.__name__)
        self._output_size = output_size

    def _build(self, inpt):
        n = np.prod(self._output_size)
        linear1 = snt.Linear(100)
        linear2 = snt.Linear(n)
        reshape = snt.BatchReshape(self._output_size)
        seq = snt.Sequential([linear1, tf.nn.elu, linear2, reshape])
        return seq(inpt)


class SpatialTransformer(snt.AbstractModule):

    def __init__(self, img_size, crop_size, constraints=None, inverse=False):
        super(SpatialTransformer, self).__init__(self.__class__.__name__)

        with self._enter_variable_scope():
            self._warper = snt.AffineGridWarper(img_size, crop_size, constraints)
            if inverse:
                self._warper = self._warper.inverse()

    def _build(self, img, transform_params):
        if len(img.get_shape()) == 3:
            img = img[..., tf.newaxis]

        grid_coords = self._warper(transform_params)
        return snt.resampler(img, grid_coords)


class AIRCell(snt.RNNCore):
    _n_transform_param = 4

    def __init__(self, img_size, crop_size, n_latent, transition):
        super(AIRCell, self).__init__(self.__class__.__name__)
        self._img_size = img_size
        self._n_pix = np.prod(self._img_size)
        self._crop_size = crop_size
        self._n_latent = n_latent
        self._transition = transition

        with self._enter_variable_scope():

            self._canvas = snt.TrainableVariable(self._img_size, dtype=tf.float32, name='initial_canvas')

            transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
            self._transform_param = TransformParam(self._n_transform_param)

            self._conv_encoder = ConvEncoder()
            self._spatial_transformer = SpatialTransformer(img_size, crop_size, transform_constraints)
            self._input_encoder = Encoder(self._transition.output_size[0])
            self._encoder = Encoder(n_latent)
            self._decoder = Decoder(crop_size)
            self._inverse_transformer = SpatialTransformer(img_size, crop_size, transform_constraints, inverse=True)

    @property
    def state_size(self):
        return [
            np.prod(self._img_size),        # image
            np.prod(self._img_size),        # canvas
            self._n_latent,                 # what
            self._n_transform_param,        # where
            self._transition.state_size,    # hidden state of the rnn
            1,                              # presence logit
        ]

    @property
    def output_size(self):
        return [np.prod(self._img_size), np.prod(self._crop_size), self._n_latent, self._n_transform_param, 1]

    def initial_state(self, img):
        batch_size = img.get_shape().as_list()[0]
        hidden_state = self._transition.initial_state(batch_size, tf.float32, trainable=True)

        # where_code = np.asarray([1, 0, 1, 0], dtype=np.float32).reshape((1, 4))
        # where_code = tf.get_variable('where_init', initializer=where_code, dtype=tf.float32)

        where_code = tf.get_variable('where_init', shape=[1, self._n_transform_param], dtype=tf.float32)

        what_code = tf.get_variable('what_init', shape=[1, self._n_latent], dtype=tf.float32)
        flat_canvas = tf.reshape(self._canvas(), (1, self._n_pix))
        where_code, what_code, flat_canvas = (tf.tile(i, (batch_size, 1)) for i in (where_code, what_code, flat_canvas))

        flat_img = tf.reshape(img, (batch_size, self._n_pix))
        flat_canvas = tf.tile(tf.reshape(self._canvas(), (1, self._n_pix)), (batch_size, 1))
        init_presence = tf.convert_to_tensor(np.ones((batch_size, 1), dtype=np.float32) * 1e3)
        return [flat_img, flat_canvas, what_code, where_code, hidden_state, init_presence]

    def _build(self, inpt, state):

        img_flat, canvas_flat, what_code, where_code, hidden_state, presence_logit = state
        img = tf.reshape(img_flat, (-1,) + tuple(self._img_size))

        inpt_encoding = self._conv_encoder(img)
        inpt_encoding = self._input_encoder(inpt_encoding)

        rnn_inpt = tf.concat((inpt_encoding, what_code, where_code), -1)
        rnn_inpt = snt.Linear(self._transition.output_size[0])(rnn_inpt)
        hidden_output, hidden_state = self._transition(rnn_inpt, hidden_state)

        where_code = self._transform_param(hidden_output)
        cropped = self._spatial_transformer(img, where_code)

        presence_inpt = tf.concat([hidden_output, tf.nn.sigmoid(presence_logit)], -1)
        presence_logit = snt.Linear(1)(presence_inpt)

        what_code = self._conv_encoder(cropped)
        what_code = self._encoder(what_code)

        decoded = self._decoder(what_code)
        inversed = self._inverse_transformer(decoded, where_code)

        inversed_flat = tf.reshape(inversed, (-1, self._n_pix))

        canvas_flat = canvas_flat + tf.nn.sigmoid(presence_logit) * inversed_flat
        decoded_flat = tf.reshape(decoded, (-1, np.prod(self._crop_size)))
        output = [canvas_flat, decoded_flat, what_code, where_code, presence_logit]
        state = [img_flat, canvas_flat, what_code, where_code, hidden_state, presence_logit]
        return output, state


if __name__ == '__main__':
    learning_rate = 1e-4
    batch_size = 10
    img_size = 50, 50
    crop_size = 20, 20
    n_latent = 10
    n_steps = 3

    x = tf.placeholder(tf.float32, (batch_size,) + img_size, name='inpt')

    transition = snt.GRU(n_latent)
    air = AIRCell(img_size, crop_size, n_latent, transition)
    initial_state = air.initial_state(x)

    dummy_sequence = tf.zeros((n_steps, batch_size, 1), name='dummy_sequence')
    outputs, state = tf.nn.dynamic_rnn(air, dummy_sequence, initial_state=initial_state, time_major=True)
    canvas, crop, what, where, presence_logit = outputs

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