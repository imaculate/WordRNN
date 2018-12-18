from __future__ import print_function
import os
import random
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ModelOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Fiction generation with Word-RNN')
        parser.add_argument('--mode', type=str, default='train', help='Running mode [train, test, sample]')
        parser.add_argument('--model', type=str, default='lstm',
                            help='Model type [lstm, gru, rnn, irnn] (default: lstm)')
        parser.add_argument('--dataset', type=str, default='tinyshakespeare', help='The name of the dataset (default: tinyshakespeare)')
        parser.add_argument('--rnn_size', type=int, default='128', help='size of LSTM internal state (default:128)')
        parser.add_argument('--embedding_size', type=int, default=200, help='size of word embeddings')
        parser.add_argument('--learn_embeddings', type=str2bool, default=True, help='True to learn embeddings, False to keep embeddings fixed (default: True)')
        parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the LSTM (default: 2)')
        parser.add_argument('--num_fixed', type=int, default=0, help='number of recurrent layers to remain fixed (untrained), pretrained (LSTM only) (default: 0)')
        parser.add_argument('--lsuv_int', type=str2bool, default=False, help='use layer-sequential unit-variance (LSUV) initialization (default: false)')
        parser.add_argument('--multiplicative_integration', type=str2bool, default=False, help='turns on multiplicative integration (as opposed to simply summing states) (default: false)')
        parser.add_argument('--learning_rate', type=float, default=2e-3, help='learning (default: 2e-3)')
        parser.add_argument('--learning_rate_decay', type=float, default=1, help='learning rate decay - rmsprop only (default: 1)')
        parser.add_argument('--learning_rate_decay_after', type=int, default=0, help='in number of epochs, when to start decaying the learning rate (default: 0)')
        parser.add_argument('--learning_rate_decay_by_val_loss', type=str2bool, default=False, help='if True, learning rate is decayed when a validation loss is not smaller than the previous (default: False)')
        parser.add_argument('--learning_rate_decay_wait', type=int, default=0, help='the minimum number of epochs the learning rate is kept after decaying it because of validation loss (default: 0)')
        parser.add_argument('--decay_rate', type=float, default=0.5, help='decay rate for rmsprop (default: 0.5)')
        parser.add_argument('--dropout', type=float, default=0, help='dropout for regularization, used after each RNN hidden layer. 0 = no dropout (default: 0)')
        parser.add_argument('--recurrent_dropout', type=float, default=0, help='dropout for regularization, used on recurrent connections. 0 = no dropout (default: 0)')
        parser.add_argument('--zoneout', type=float, default=0, help='zoneout for regularization, used on recurrent connections. 0 = no zoneout (default: 0)')
        parser.add_argument('--zoneout_c', type=float, default=0, help='zoneout on the lstm cell. 0 = no zoneout (default: 0)')
        parser.add_argument('--recurrent_depth', type=int, default=0, help='the number of additional h2h matrices, when the model is an SDRNN (default:0)')
        parser.add_argument('--gradient_noise', type=float, default=0, help='amount of gradient noise for regularization (will be decayed over time t, as b/t^0.55 ) (default: 0)')
        parser.add_argument('--activation_clamp', type=float, default=0, help='clamp activations at this value (sdrnn only) (default: 0)')
        parser.add_argument('--activation_l2', type=float, default=0, help='amount of l2 penalization to apply to the activations (sdrnn only) (default: 0)')
        parser.add_argument('--l2', type=float, default=0, help='amount of l2 weight decay to regularize the model with (default: 0)')
        parser.add_argument('--activation_l1', type=float, help='amount of l1 weight decay to regularize the model with (rnn & dfarnn only) (default: 0)')
        parser.add_argument('--batch_normalization', type=int, default=0, help='whether to apply batch normalization [0=no BN, 1=vertical BN, 2=vertical, horizontal BN] (default: 0)')
        parser.add_argument('--;', type=str2bool, default=False, help='whether to apply layer normalization (default: False)')
        parser.add_argument('--seq_length', type=int, default=50, help='number of timesteps to unroll for (default: 50)')
        parser.add_argument('--batch_size', type=int, default=50, help='number of sequences to train on parallel (default: 50)')
        parser.add_argument('--max_epochs', type=int, default=50, help='number of full passes through the training data (default: 50)')
        parser.add_argument('--grad_clip', type=int, default=5, help='clip gradients at this value (default: 5)')
        parser.add_argument('--max_norm', type=float, default=0, help='make sure gradient norm does not exceed this value (default: 0)')
        parser.add_argument('--train_frac', type=float, default=0.95, help='fraction of data that goes into train set (default: 0.95)')
        parser.add_argument('--val_frac', type=float, default=0.05, help='fraction of data that goes into validation set test_frac will be computed as (1 - train_frac - val_frac)(default: 0.05)')
        parser.add_argument('--init_from', default=None, help='initialize network parameters from checkpoint at this path (default: None)' )
        parser.add_argument('--random_crops', type=str2bool, default=True, help='use a random crop of the training data per epoch when it does not evenly divide into the number of batches (default: True)')
        parser.add_argument('--word_level', type=str2bool, default=False,
                            help='whether to operate on the word level, instead of character level [False: use chars, True: use words ] (default: False)')
        parser.add_argument('--threshold', type=int, default=0,
                            help='minimum number of occurences a token must have to be included (ignored if --word-level is False) (default: 0)')
        parser.add_argument('--glove', type=str2bool, default=False,
                            help='whether or not to use GloVe embeddings (default: False)')
        parser.add_argument('--non_glove_embedding', type=str2bool, default=False,
                            help='use embedding with random initialization (default: False)')
        parser.add_argument('--optimizer', type=str, default='rmsprop',
                            help='which optimizer to use: adam or rmsprop (default: rmsprop)')

        ## bookeeping
        parser.add_argument('--seed', type=int, default=123, help='manual random number generator seed (default: 123)')
        parser.add_argument('--print_every', type=int, default=1, help='how many steps/minibatches between printing out the loss (default: 1)')
        parser.add_argument('--eval_val_every', type=int, default=1000, help='every how many iterations should we evaluate on validation data (default: 1000)')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='output directory where checkpoints get written (default: checkpoint )')
        parser.add_argument('--data_dir', type=str, default='./data', help='input directory where dataset is stored (default: data )')
        parser.add_argument('--accurate_gpu_timing', default=False, help='set this flag to get precise timings when using GPU. Might make code bit slower but reports accurate timings (default: False)')
        #GPU /CPU
        parser.add_argument('--gpuid', type=int, default=0, help='which gpu to use , -1 to use CPU (default: 0)')
        parser.add_argument('--opencl', type=str2bool, default=False, help='use OpenCL instead of CUDA (default: False)')

        self._parser = parser

    def parse(self):
        opt = self._parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

        opt.color_space = opt.color_space.upper()

        if opt.seed == 0:
            opt.seed = random.randint(0, 2**31 - 1)

        if opt.data_dir == './data':
            opt.data_dir += ('/' + opt.dataset)

        if opt.checkpoints_dir == './checkpoints':
            opt.checkpoints_dir += ('/' + opt.dataset)

        return opt
