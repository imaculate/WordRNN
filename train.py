'''
This file trains a character-level multi-layer RNN on text data

Code is based on implementation in
https://github.com/larspars/word-rnn but written in python,
The 'word-rnn' is in turn based on https://github.com/oxford-cs-ml-2015/practical6
but was modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)
'''

from util import OneHot, GloveEmbedding, misc, SharedDropout, Zoneout, LayerNormalization, LookupTableFixed

