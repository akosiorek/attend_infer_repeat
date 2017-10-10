
# coding: utf-8

# In[1]:

from os import path as osp
import numpy as np
import tensorflow as tf
import sonnet as snt
from attrdict import AttrDict

from evaluation import make_fig, make_logger

from data import load_data, tensors_from_data
from mnist_model import AIRonMNIST

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[2]:

learning_rate = 1e-4
n_steps = 3

results_dir = '../results'
run_name = 'multi_mnist'

logdir = osp.join(results_dir, run_name)
checkpoint_name = osp.join(logdir, 'model.ckpt')
axes = {'imgs': 0, 'labels': 0, 'nums': 1}


# In[3]:

batch_size = 64
use_prior = True

num_steps_prior = AttrDict(
    anneal='exp',
    init=1. - 1e-15,
    final=1e-7,
    steps_div=1e4,
    steps=1e5,
    hold_init=1e3,
)

appearance_prior = AttrDict(loc=0., scale=1.)
where_scale_prior = AttrDict(loc=0., scale=1.)
where_shift_prior = AttrDict(loc=0., scale=1.)

use_reinforce = True
sample_presence = True
step_bias = .75
transform_var_bias = .5
output_multiplier = .5

init_explore_eps = 1e-3

l2_weight = 0.


# In[4]:

valid_data = load_data('mnist_validation.pickle')
train_data = load_data('mnist_train.pickle')


# In[5]:

tf.reset_default_graph()
train_tensors = tensors_from_data(train_data, batch_size, axes, shuffle=True)
valid_tensors = tensors_from_data(valid_data, batch_size, axes, shuffle=False)
x, valid_x = train_tensors['imgs'], valid_tensors['imgs']
y, test_y = train_tensors['nums'], valid_tensors['nums']
    
n_hidden = 32 * 8
n_layers = 2
n_hiddens = [n_hidden] * n_layers
    
air = AIRonMNIST(x, y,
                max_steps=n_steps, 
                explore_eps=init_explore_eps,
                inpt_encoder_hidden=n_hiddens,
                glimpse_encoder_hidden=n_hiddens,
                glimpse_decoder_hidden=n_hiddens,
                transform_estimator_hidden=n_hiddens,
                steps_pred_hidden=[128, 64],
                baseline_hidden=[256, 128],
                transform_var_bias=transform_var_bias,
                step_bias=step_bias,
                output_multiplier=output_multiplier
)


# In[6]:

train_step, global_step = air.train_step(learning_rate, l2_weight, appearance_prior, where_scale_prior,
                            where_shift_prior, num_steps_prior)


# In[7]:

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
    
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
all_summaries = tf.summary.merge_all()


# In[8]:

summary_writer = tf.summary.FileWriter(logdir, sess.graph)
saver = tf.train.Saver()


# In[9]:

train_batches = train_data['imgs'].shape[0] // batch_size
valid_batches = valid_data['imgs'].shape[0] // batch_size
log = make_logger(air, sess, summary_writer, train_tensors, train_batches, valid_tensors, valid_batches)


# In[ ]:

train_itr = sess.run(global_step)
print 'Starting training at iter = {}'.format(train_itr)

if train_itr == 0:
    log(0)

while train_itr < 3 * 1e5:
        
    train_itr, _ = sess.run([global_step, train_step])
    
    if train_itr % 1000 == 0:
        summaries = sess.run(all_summaries)
        summary_writer.add_summary(summaries, train_itr)
        
    if train_itr % 10000 == 0:
        log(train_itr)
        
    if train_itr % 5000 == 0:
        saver.save(sess, checkpoint_name, global_step=train_itr)
        make_fig(air, sess, logdir, train_itr)    
