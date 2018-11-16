from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np

def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.

    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {str:int}
    num_features    # Word vector dimensionality
    min_word_count  # Minimum word count
    context         # Context window size
    """
    model_dir = 'word2vec_models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        #print 'Loading existing Word2Vec model \'%s\'' % split(model_name)[-1]
    else:
        # Set values for various parameters
        num_workers = 2       # Number of threads to run in parallel
        downsampling = 1e-3   # Downsample setting for frequent words

        # Initialize and train the model
        print( "Training Word2Vec model...")
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers, \
                                            size=num_features, min_count = min_word_count, \
                                            window = context, sample = downsampling)

        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print ('Saving Word2Vec model' +  str(split(model_name)[-1]))
        embedding_model.save(model_name)

    #  add unknown words
    embedding_weights = [np.array([embedding_model[w] if w in embedding_model\
                                   else np.random.uniform(-0.25,0.25,embedding_model.vector_size)\
                                   for w in vocabulary_inv])]
    return embedding_weights


def load_word2vec(params):
    """
    loads Word2Vec model
    Returns initial weights for embedding layer.

    inputs:
    model_type      # GoogleNews / glove
    vocabulary_inv  # dict {str:int}
    num_features    # Word vector dimensionality
    """

    model_dir = 'word2vec_models'

    if params.model_type == 'GoogleNews':
        model_name = join(model_dir, 'GoogleNews-vectors-negative300.bin.gz')
        assert(params.num_features == 300)
        assert(exists(model_name))
        print('Loading existing Word2Vec model (GoogleNews-300)')
        embedding_model = word2vec.Word2Vec.load_word2vec_format(model_name, binary=True)

    elif params.model_type == 'glove':
        model_name = join(model_dir, 'glove.6B.%dd.txt' % (params.num_features))
        assert(exists(model_name))
        print('Loading existing Word2Vec model (Glove.6B.%dd)' % (params.num_features))

        # dictionary, where key is word, value is word vectors
        embedding_model = {}
        for line in open(model_name, 'r'):
            tmp = line.strip().split()
            word, vec = tmp[0], map(float, tmp[1:])
            assert(len(vec) == params.num_features)
            if word not in embedding_model:
                embedding_model[word] = vec
        assert(len(embedding_model) == 400000)

    else:
        raise ValueError('Unknown pretrain model type: %s!' % (params.model_type))

    embedding_weights = [embedding_model[w] if w in embedding_model
                         else np.random.uniform(-0.25, 0.25, params.num_features)
                         for w in params.vocabulary_inv]
    embedding_weights = np.array(embedding_weights).astype('float32')

    return embedding_weights


if __name__=='__main__':
    import data_helpers
    print("Loading data...")
    x, _, _, params.vocabulary_inv = data_helpers.load_data()
    w = train_word2vec(x, params.vocabulary_inv)

