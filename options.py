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
        parser = argparse.ArgumentParser(description='Fiction generation with Wod-RNN')
        parser.add_argument('--dataset', type=str, default='tinyshakespeare', metavar='D', help='The name of the dataset (default: tinyshakespeare)')
        parser.add_argument('--rnn_size', type=int, default='128', help='size of LSTM internal state (default:128)')
        parser.add_argument('--embedding_size', type=int, default=200, help='size of word embeddings')
        parser.add_argument('--learn_embeddings', type=str2bool, default=True, help='True to learn embeddings, False to keep embeddings fixed (default: True)')
        parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the LSTM (default: 2)')
        parser.add_argument('--num_fixed', type=int, default=0, help='number of recurrent layers to remain fixed (untrained), pretrained (LSTM only) (default: 0)')
        parser.add_argument('--model', type=str, default='lstm', help='Model type [lstm, gru, rnn, irnn] (default: lstm)')
        parser.add_argument('--lsuv_int', type=str2bool, default=False, help='use layer-sequential unit-variance (LSUV) initialization (default: false)')
        parser.add_argument('--multiplicative_integration', type=str2bool, default=False, help='turns on multiplicative integration (as opposed to simply summing states) (default: false)')
        parser.add_argument('--learning_rate', type=float, default=2e-3, help='learning (default: 2e-3)')
        parser.add_argument('--learning_rate_decay', type=float, default=1, help='learning rate decay - rmsprop only (default: 1)')
        parser.add_argument('--learning_rate_decay_after', type=int, default=0, help='in number of epochs, when to start decaying the learning rate (default: 0)')
    '''
    cmd:option('-learning_rate_decay_by_val_loss',0,'if 1, learning rate is decayed when a validation loss is not smaller than the previous')
    cmd:option('-learning_rate_decay_wait',0,'the minimum number of epochs the learning rate is kept after decaying it because of validation loss')
    cmd:option('-decay_rate',0.5,'decay rate for rmsprop')
    cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
    cmd:option('-recurrent_dropout',0,'dropout for regularization, used on recurrent connections. 0 = no dropout')
    cmd:option('-zoneout',0,'zoneout for regularization, used on recurrent connections. 0 = no zoneout')
    cmd:option('-zoneout_c',0,'zoneout on the lstm cell. 0 = no zoneout')
    cmd:option('-recurrent_depth', 0, 'the number of additional h2h matrices, when the model is an SDRNN')
    cmd:option('-gradient_noise',0,'amount of gradient noise for regularization (will be decayed over time t, as b/t^0.55 )')
    cmd:option('-activation_clamp',0,'clamp activations at this value (sdrnn only)')
    cmd:option('-activation_l2',0,'amount of l2 penalization to apply to the activations (sdrnn only)')
    cmd:option('-l2',0,'amount of l2 weight decay to regularize the model with')
    cmd:option('-activation_l1',0,'amount of l1 weight decay to regularize the model with (rnn & dfarnn only)')
    cmd:option('-batch_normalization',0,'whether to apply batch normalization (0=no BN, 1=vertical BN, 2=vertical and horizontal BN)')
    cmd:option('-layer_normalization',0,'whether to apply layer normalization')
    '''


        ''''
        parser.add_argument('--mode', default=0, help='run mode [0: train, 1: evaluate, 2: test] (default: 0)')
        parser.add_argument('--dataset', type=str, default='places365', help='the name of dataset [places365, cifar10, historybw] (default: places365)')
        parser.add_argument('--dataset-path', type=str, default='./dataset', help='dataset path (default: ./dataset)')
        parser.add_argument('--checkpoints-path', type=str, default='./checkpoints', help='models are saved here (default: ./checkpoints)')
        parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training (default: 16)')
        parser.add_argument('--color-space', type=str, default='lab', help='model color space [lab, rgb] (default: lab)')
        parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 30)')
        parser.add_argument('--lr', type=float, default=3e-4, metavar='LR', help='learning rate (default: 3e-4)')
        parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='learning rate exponentially decay rate (default: 0.1)')
        parser.add_argument('--lr-decay-steps', type=float, default=5e5, help='learning rate exponentially decay steps (default: 5e5)')
        parser.add_argument('--beta1', type=float, default=0, help='momentum term of adam optimizer (default: 0)')
        parser.add_argument("--l1-weight", type=float, default=100.0, help="weight on L1 term for generator gradient (default: 100.0)")
        parser.add_argument('--augment', type=str2bool, default=True, help='True for augmentation (default: True)')
        parser.add_argument('--label-smoothing', type=str2bool, default=False, help='True for one-sided label smoothing (default: False)')
        parser.add_argument('--acc-thresh', type=float, default=2.0, help="accuracy threshold (default: 2.0)")
        parser.add_argument('--kernel-size', type=int, default=4, help="default kernel size (default: 4)")
        parser.add_argument('--save', type=str2bool, default=True, help='True for saving (default: True)')
        parser.add_argument('--save-interval', type=int, default=1000, help='how many batches to wait before saving model (default: 1000)')
        parser.add_argument('--sample', type=str2bool, default=True, help='True for sampling (default: True)')
        parser.add_argument('--sample-size', type=int, default=8, help='number of images to sample (default: 8)')
        parser.add_argument('--sample-interval', type=int, default=1000, help='how many batches to wait before sampling (default: 1000)')
        parser.add_argument('--validate', type=str2bool, default=True, help='True for validation (default: True)')
        parser.add_argument('--validate-interval', type=int, default=0, help='how many batches to wait before validating (default: 0)')
        parser.add_argument('--log', type=str2bool, default=False, help='True for logging (default: True)')
        parser.add_argument('--log-interval', type=int, default=10, help='how many iterations to wait before logging training status (default: 10)')
        parser.add_argument('--visualize', type=str2bool, default=False, help='True for accuracy visualization (default: False)')
        parser.add_argument('--visualize-window', type=int, default=100, help='the exponentially moving average window width (default: 100)')
        parser.add_argument('--test-size', type=int, default=100, metavar='N', help='number of Turing tests (default: 100)')
        parser.add_argument('--test-delay', type=int, default=0, metavar='N', help='number of seconds to wait when doing Turing test, 0 for unlimited (default: 0)')
        parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        '''


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
