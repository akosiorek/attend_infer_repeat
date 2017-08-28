import os.path as osp
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from tf_tools.eval import make_expr_logger


def rect(bbox, c=None, facecolor='none', label=None, ax=None):
    r = Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2],
                  edgecolor=c, facecolor=facecolor, label=label)

    if ax is not None:
        ax.add_patch(r)
    return r


def rect_stn(ax, width, height, w, c=None):
    sx, tx, sy, ty = w
    x = width * (1. - sx + tx) / 2
    y = height * (1. - sy + ty) / 2
    bbox = [y - .5, x - .5, height * sy, width * sx]
    rect(bbox, c, ax=ax)


def make_fig(air, sess, checkpoint_dir=None, global_step=None):
    n_steps = air.max_steps

    xx, pred_canvas, pred_crop, prob, pres, w = sess.run(
        [air.obs, air.canvas, air.glimpse, air.num_steps_distrib.prob()[..., 1:], air.presence, air.where])
    height, width = xx.shape[1:]

    max_imgs = 10
    bs = min(max_imgs, air.batch_size)
    scale = 1.5
    figsize = scale * np.asarray((bs, 2 * n_steps + 1))
    fig, axes = plt.subplots(2 * n_steps + 1, bs, figsize=figsize)

    for i, ax in enumerate(axes[0]):
        ax.imshow(xx[i], cmap='gray', vmin=0, vmax=1)

    for i, ax_row in enumerate(axes[1:1 + n_steps]):
        for j, ax in enumerate(ax_row):
            ax.imshow(pred_canvas[i, j], cmap='gray', vmin=0, vmax=1)
            if pres[i, j, 0] > .5:
                rect_stn(ax, width, height, w[i, j], 'r')

    for i, ax_row in enumerate(axes[1 + n_steps:]):
        for j, ax in enumerate(ax_row):
            ax.imshow(pred_crop[i, j], cmap='gray')  # , vmin=0, vmax=1)
            ax.set_title('{:d} with p({:d}) = {:.02f}'.format(int(pres[i, j, 0]), i + 1, prob[j, i].squeeze()),
                         fontsize=4 * scale)

    for ax in axes.flatten():
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if checkpoint_dir is not None:
        fig_name = osp.join(checkpoint_dir, 'progress_fig_{}.png'.format(global_step))
        fig.savefig(fig_name, dpi=300)
        plt.close('all')


def make_logger(air, sess, summary_writer, train_tensor, train_batches, test_tensor, test_batches):
    exprs = {
        'loss': air.loss.value,
        'rec_loss': air.rec_loss,
        'num_step_acc': air.num_step_accuracy,
        'num_step': air.num_step
    }

    if air.use_prior:
        exprs['prior_loss'] = air.prior_loss.value
        if air.num_steps_prior is not None:
            exprs['num_steps_prior_loss'] = air.num_steps_prior_loss

        if air.appearance_prior is not None:
            exprs['appearance_prior_loss'] = air.appearance_prior_loss
            exprs['where_kl'] = air.where_kl

    if air.use_reinforce:
        exprs['baseline_loss'] = air.baseline_loss
        exprs['reinforce_loss'] = air.reinforce_loss
        exprs['imp_weight'] = tf.reduce_mean(air.importance_weight)

    if air.l2_weight > 0:
        exprs['l2_loss'] = air.l2_loss

    train_log = make_expr_logger(sess, summary_writer, train_batches / air.batch_size, exprs, name='train')

    data_dict = {
        train_tensor['imgs']: test_tensor['imgs'],
        train_tensor['nums']: test_tensor['nums']
    }
    test_log = make_expr_logger(sess, summary_writer, test_batches / air.batch_size, exprs, name='test',
                                data_dict=data_dict)

    def log(train_itr):
        train_log(train_itr)
        test_log(train_itr)
        print

    return log