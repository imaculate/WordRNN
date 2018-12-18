from options import ModelOptions
import tensorflow as tf
import numpy as np
import random


def main(options):
    # reset tensorflow graph
    tf.reset_default_graph()

    # initialize random seed
    tf.set_random_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)

    # create a session environment
    with tf.Session() as sess:
        ## train / val / test spli
        test_frac = max(0, 1 - (options.train_frac + options.val_frac))
        split_sizes = [options.train_frac, options.val_frac, test_frac]

        if options.mode == 'train':
            ## create embeddings then  models

            # create embeddings
            if options.glove:
                if options.learn_embeddings:
                    embedding = GloVeEmbedding(vocab, options.embedding_size, options.data_dir, "", True)
                else:
                    embedding = GloVeEmbeddingFixed(vocab, options.embedding_size, options.data_dir, "", True)

            elif options.non_glove_embedding:
                if options.learn_embeddings:
                    embedding = nn.LookupTable(vocab_size, opt.embedding_size)
                else:
                    print("using fixed embeddings")
                    embedding = LookupTableFixed(vocab_size, opt.embedding_size)

            if options.model == 'gru':
                model = GRU(vocab_size, options.rnn_size, options.num_layers, options.dropout, embedding)
            elif options.model == 'rnn':
                model = RNN(vocab_size, options.rnn_size, options.num_layers, options.dropout, embedding)
            elif options.model == 'irnn':
                model, h2hs = IRNN(vocab_size, options.rnn_size, options.num_layers, options.dropout, embedding)
            ## default lstm
            elif options.model == 'sdrnn':
                model = SDRNN(input_size, vocab_size, options.rnn_size, options.num_layers, options.dropout, embedding)
            else:
                model = LSTM(vocab_size, options.rnn_size, options.num_layers, options.dropout, options.recurrent_dropout,
                                   embedding, options.num_fixed)

if __name__ == "__main__":
    main(ModelOptions().parse())