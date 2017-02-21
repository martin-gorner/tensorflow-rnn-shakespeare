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
import numpy as np
import my_txtutils

# these must match what was saved !
ALPHASIZE = my_txtutils.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512

# Data files can be downloaded from the following locations:
#    - Fully trained on Shakespeare or Tensorflow Python source:
#      https://drive.google.com/file/d/0B5njS_LX6IsDQ1laeDJ6dktSb3M/view?usp=sharing
#    - Partially trained, to see how they make progress in training:
#      https://drive.google.com/file/d/0B5njS_LX6IsDc2Y0X1VWc1pVTE0/view?usp=sharing

shakespeareC0 = "./rnn_test_minibatchseq_1477684737-0"      # random
shakespeareC1 = "./rnn_test_minibatchseq_1477684737-150000"  # mostly lower case
shakespeareC2 = "./rnn_test_minibatchseq_1477684737-300000"  # words, paragraphs
shakespeareC3 = "./rnn_test_minibatchseq_1477684737-450000"  # structure of a play
shakespeareC4 = "./rnn_test_minibatchseq_1477684737-600000"  # better structure of a play, 3-letter words in correct English
shakespeareC5 = "./rnn_test_minibatchseq_1477684737-900000"  # character names wrong but in correct place, correct 4-letter words start appearing
shakespeareC6 = "./rnn_test_minibatchseq_1477684737-5100000"  # good names, even when invented (ex: SIR NATHANIS LORD OF SYRACUSE), correct 6-8 letter words
shakespeareC7 = "./rnn_test_minibatchseq_1477684737-10083000"  # ACT V SCENE IV, [Re-enter KING JOHN with MARDIAN] but some still badly closed, DON ADRIANO DRAGHAMONE <- invented! most words correct
shakespeareC8 = "./rnn_test_minibatchseq_1477684737-20049000"  # most scenic indications correct: [Stabs aside] [Enter FERDINAND] [Dies] [Exit ROSALIND] [To COMINIUS with me]
shakespeareC9 = "./rnn_test_minibatchseq_1477684737-47997000"  # [Enter CYMBELINE and LUCETTA] [Enter PRINCE HENRY and Patroclus] [Exeunt all but two Englishman] [Enter PRINCE HENRY, and Attendants]
shakespeareB10 = "./rnn_test_minibatchseq_1477670023-174939000"  # [Re-enter LAUNIS'S] [Enter MARK ANTONY and TREBIAGO, DIONA, and CAPULET and Lords, and Soldiers]

pythonA0 = "./rnn_test_minibatchseq_1477832845-150000"  # gibberish
pythonA1 = "./rnn_test_minibatchseq_1477832845-300000"  # some == and (
pythonA2 = "./rnn_test_minibatchseq_1477832845-1050000"  # correct function calls with parameters and ()
pythonA3 = "./rnn_test_minibatchseq_1477832845-4050000"  # looks like Tensorflow Python, nested () and [] ok
pythonB10 = "./rnn_test_minibatchseq_1477834023-138609000"  # can even recite the Apache license

# use topn=10 for all but the last which works with topn=2 for Shakespeare and topn=3 for Python
author = shakespeareB10

ncnt = 0
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./rnn_test_minibatchseq_1477670023-174939000.meta')
    new_saver.restore(sess, author)
    x = my_txtutils.convert_from_alphabet(ord("K"))
    x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    # initial values
    y = x
    h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    for i in range(1000000000):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

        # If sampling is be done from the topn most likely characters, the generated text
        # is more credible and more "english". If topn is not set, it defaults to the full
        # distribution (ALPHASIZE)

        # Recommended: topn = 10 for intermediate checkpoints, topn=2 for fully trained checkpoints

        c = my_txtutils.sample_from_probabilities(yo, topn=2)
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        c = chr(my_txtutils.convert_to_alphabet(c))
        print(c, end="")

        if c == '\n':
            ncnt = 0
        else:
            ncnt += 1
        if ncnt == 100:
            print("")
            ncnt = 0


#         TITUS ANDRONICUS
#
#
# ACT I
#
#
#
# SCENE III	An ante-chamber. The COUNT's palace.
#
#
# [Enter CLEOMENES, with the Lord SAY]
#
# Chamberlain	Let me see your worshing in my hands.
#
# LUCETTA	I am a sign of me, and sorrow sounds it.
#
# [Enter CAPULET and LADY MACBETH]
#
# What manner of mine is mad, and soon arise?
#
# JULIA	What shall by these things were a secret fool,
# That still shall see me with the best and force?
#
# Second Watchman	Ay, but we see them not at home: the strong and fair of thee,
# The seasons are as safe as the time will be a soul,
# That works out of this fearful sore of feather
# To tell her with a storm of something storms
# That have some men of man is now the subject.
# What says the story, well say we have said to thee,
# That shall she not, though that the way of hearts,
# We have seen his service that we may be sad.
#
# [Retains his house]
# ADRIANA	What says my lord the Duke of Burgons of Tyre?
#
# DOMITIUS ENOBARBUS	But, sir, you shall have such a sweet air from the state,
# There is not so much as you see the store,
# As if the base should be so foul as you.
#
# DOMITIUS ENOY	If I do now, if you were not to seek to say,
# That you may be a soldier's father for the field.
#
# [Exit]
