
# coding: utf-8

# In[ ]:

from os import path as osp
import tensorflow as tf

from evaluation import make_fig, make_logger
from experiment_tools import load, init_checkpoint, parse_flags, print_flags, set_flags


# In[ ]:

import sys
sys.path.append('../')


# In[ ]:

# Define flags
flags = tf.flags

flags.DEFINE_string('data_config', 'configs/static_mnist_data.py', '')
flags.DEFINE_string('model_config', 'configs/imp_weighted_nvil.py', '')
flags.DEFINE_string('results_dir', '../checkpoints', '')
flags.DEFINE_string('run_name', 'test_run', '')

flags.DEFINE_integer('batch_size', 64, '')

flags.DEFINE_integer('summary_every', 1000, '')
flags.DEFINE_integer('log_every', 5000, '')
flags.DEFINE_integer('save_every', 5000, '')
flags.DEFINE_integer('max_train_iter', int(3 * 1e5), '')
flags.DEFINE_boolean('restart', False, '')

flags.DEFINE_float('eval_size_fraction', .01, '')

# Parse flags
parse_flags()
F = flags.FLAGS

# Prepare enviornment
logdir = osp.join(F.results_dir, F.run_name)
logdir, flags, restart_checkpoint = init_checkpoint(logdir, F.data_config, F.model_config, F.restart)
checkpoint_name = osp.join(logdir, 'model.ckpt')


# In[ ]:

# Build the graph
tf.reset_default_graph()
data_dict = load(F.data_config, F.batch_size)
air, train_step, global_step = load(F.model_config, img=data_dict.train_img, num=data_dict.train_num)

print_flags()


# In[ ]:

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# In[ ]:

sess.run(tf.global_variables_initializer())


# In[ ]:

saver = tf.train.Saver()
if restart_checkpoint is not None:
    print "Restoring from '{}'",format(restart_checkpoint)
    saver.restore(sess, restart_checkpoint)


# In[ ]:

summary_writer = tf.summary.FileWriter(logdir, sess.graph)
all_summaries = tf.summary.merge_all()


# In[ ]:

# Logging
ax = data_dict['axes']['imgs']
factor = F.eval_size_fraction
train_batches, valid_batches = [int(data_dict[k]['imgs'].shape[ax] * factor) for k in ('train_data', 'valid_data')]
log = make_logger(air, sess, summary_writer, data_dict.train_tensors,
                  train_batches, data_dict.valid_tensors, valid_batches)


# In[ ]:

train_itr = sess.run(global_step)
print 'Starting training at iter = {}'.format(train_itr)

if train_itr == 0:
    log(0)
    make_fig(air, sess, logdir, train_itr)

while train_itr < F.max_train_iter:

    train_itr, _ = sess.run([global_step, train_step])

    if train_itr % F.summary_every == 0:
        summaries = sess.run(all_summaries)
        summary_writer.add_summary(summaries, train_itr)

    if train_itr % F.log_every == 0:
        log(train_itr)

    if train_itr % F.save_every == 0:
        saver.save(sess, checkpoint_name, global_step=train_itr)
        make_fig(air, sess, logdir, train_itr)


# In[ ]:

make_fig(air, sess, n_samples=64)

