import tensorflow as tf
import sonnet as snt

from model import AIRModel
from modules import BaselineMLP, MLP, Encoder, Decoder, StochasticTransformParam


class AIRonMNIST(AIRModel):

    def __init__(self, obs, nums,
                 inpt_encoder_hidden=[256]*2,
                 glimpse_encoder_hidden=[256]*2,
                 glimpse_decoder_hidden=[252]*2,
                 transform_estimator_hidden=[256]*2,
                 baseline_hidden=[256, 128],
                 *args, **kwargs):

        self.baseline = BaselineMLP(baseline_hidden)

        super(AIRonMNIST, self).__init__(
            *args,
            obs=obs,
            nums=nums,
            glimpse_size=(20, 20),
            n_appearance=50,
            transition=snt.LSTM(256),
            input_encoder=(lambda: Encoder(inpt_encoder_hidden)),
            glimpse_encoder=(lambda: Encoder(glimpse_encoder_hidden)),
            glimpse_decoder=(lambda x: Decoder(glimpse_decoder_hidden, x)),
            transform_estimator=(lambda x: StochasticTransformParam(transform_estimator_hidden, x)),
            output_std=.3,
            **kwargs
        )