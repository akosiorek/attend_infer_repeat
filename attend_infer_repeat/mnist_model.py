import tensorflow as tf
import sonnet as snt

from model import AIRModel
from modules import BaselineMLP, Encoder, Decoder, StochasticTransformParam, StepsPredictor


class AIRonMNIST(AIRModel):

    def __init__(self, obs, nums,
                 inpt_encoder_hidden=[256]*2,
                 glimpse_encoder_hidden=[256]*2,
                 glimpse_decoder_hidden=[252]*2,
                 transform_estimator_hidden=[256]*2,
                 steps_pred_hidden=[50]*1,
                 baseline_hidden=[256, 128]*1,
                 transform_var_bias=-2.,
                 step_bias=0.,
                 *args, **kwargs):

        self.baseline = BaselineMLP(baseline_hidden)

        def _make_transform_estimator(x):
            est = StochasticTransformParam(transform_estimator_hidden, x, scale_bias=transform_var_bias)
            return est

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
            transform_estimator=_make_transform_estimator,
            steps_predictor=(lambda: StepsPredictor(steps_pred_hidden, step_bias)),
            output_std=.3,
            **kwargs
        )