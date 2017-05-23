# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1
import os
import time
import math
import my_txtutils as txt
tf.set_random_seed(0)

# Full comments in rnn_train.py
# This file implements the exact same model but using the state_is_tuple=True
# option in tf.nn.rnn_cell.MultiRNNCell. This option is enabled by default.
# It produces faster code (by ~10%) but handling the state as a tuple is bit
# more cumbersome. Search for comments containing "state_is_tuple=True" for
# details.

SEQLEN = 30
BATCHSIZE = 100
ALPHASIZE = txt.ALPHASIZE
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001  # fixed learning rate

# load data, either shakespeare, or the Python source of Tensorflow itself
shakedir = "shakespeare/*.txt"
# shakedir = "../tensorflow/**/*.py"
codetext, valitext, bookranges = txt.read_data_files(shakedir, validation=False)

# display some stats on the data
epoch_size = len(codetext) // (BATCHSIZE * SEQLEN)
txt.print_data_stats(len(codetext), len(valitext), epoch_size)

#
# the model
#
lr = tf.placeholder(tf.float32, name='lr')  # learning rate
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
X = tf.placeholder(tf.uint8, [None, None], name='X')    # [ BATCHSIZE, SEQLEN ]
Xo = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)                 # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
# expected outputs = same sequence shifted by 1 since we are trying to predict the next character
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)               # [ BATCHSIZE, SEQLEN, ALPHASIZE ]

cells = [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
multicell = rnn.MultiRNNCell(cells, state_is_tuple=True)

# When using state_is_tuple=True, you must use multicell.zero_state
# to create a tuple of  placeholders for the input states (one state per layer).
# When executed using session.run(zerostate), this also returns the correctly
# shaped initial zero state to use when starting your training loop.
zerostate = multicell.zero_state(BATCHSIZE, dtype=tf.float32)

Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=zerostate)
# Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
# H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence

H = tf.identity(H, name='H')  # just to give it a name

# Softmax layer implementation:
# Flatten the first two dimension of the output [ BATCHSIZE, SEQLEN, ALPHASIZE ] => [ BATCHSIZE x SEQLEN, ALPHASIZE ]
# then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
# From the readout point of view, a value coming from a cell or a minibatch is the same thing

Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
Ylogits = layers.linear(Yflat, ALPHASIZE)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# stats for display
seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

# Init Tensorboard stuff. This will save Tensorboard information into a different
# folder at each run named 'log/<timestamp>/'.
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1)

# for display: init the progress bar
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN
progress = txt.Progress(DISPLAY_FREQ, size=111+2, msg="Training on next "+str(DISPLAY_FREQ)+" batches")

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

# training loop
istate = sess.run(zerostate)  # initial zero input state (a tuple)
for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=1000):

    # train on one minibatch
    feed_dict = {X: x, Y_: y_, lr: learning_rate, batchsize: BATCHSIZE}
    # This is how you add the input state to feed dictionary when state_is_tuple=True.
    # zerostate is a tuple of the placeholders for the NLAYERS=3 input states of our
    # multi-layer RNN cell. Those placeholders must be used as keys in feed_dict.
    # istate is a tuple holding the actual values of the input states (one per layer).
    # Iterate on the input state placeholders and use them as keys in the dictionary
    # to add actual input state values.
    for i, v in enumerate(zerostate):
        feed_dict[v] = istate[i]
    _, y, ostate, smm = sess.run([train_step, Y, H, summaries], feed_dict=feed_dict)

    # save training data for Tensorboard
    summary_writer.add_summary(smm, step)

    # display a visual validation of progress (every 50 batches)
    if step % _50_BATCHES == 0:
        feed_dict = {X: x, Y_: y_, batchsize: BATCHSIZE}  # no dropout for validation
        for i, v in enumerate(zerostate):
            feed_dict[v] = istate[i]
        y, l, bl, acc = sess.run([Y, seqloss, batchloss, accuracy], feed_dict=feed_dict)
        txt.print_learning_learned_comparison(x[:5], y, l, bookranges, bl, acc, epoch_size, step, epoch)

    # save a checkpoint (every 500 batches)
    if step // 10 % _50_BATCHES == 0:
        saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)

    # display progress bar
    progress.step(reset=step % _50_BATCHES == 0)

    # loop state around
    istate = ostate
    step += BATCHSIZE * SEQLEN
