import numpy as np
import os
import re
import itertools
import scipy.sparse as sp
import cPickle as pickle
from collections import Counter
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")


def clean_str(string):
    # remove stopwords
    # string = ' '.join([word for word in string.split() if word not in cachedStopWords])
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def pad_sentences(sentence_sets, padding_word="<PAD/>", max_length=500):
    sequence_length = 0
    for sentences in sentence_sets:
        sequence_length = max(min(max(len(x) for x in sentences), max_length), sequence_length)
    
    padded_sentence_sets = []
    for sentences in sentence_sets:
        padded_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            if len(sentence) < max_length:
                num_padding = sequence_length - len(sentence)
                new_sentence = sentence + [padding_word] * num_padding
            else:
                new_sentence = sentence[:max_length]
            padded_sentences.append(new_sentence)
        padded_sentence_sets.append(padded_sentences)
    return padded_sentence_sets, sequence_length


def load_data_and_labels(data, M=0, N=0):
    x_text = [clean_str(doc['text']) for doc in data]
    x_text = [s.split(" ") for s in x_text]
    labels = [doc['catgy'] for doc in data]
    row_idx, col_idx, val_idx = [], [], []
    for i in xrange(len(labels)):
        l_list = list(set(labels[i])) # remove duplicate cateories to avoid double count
        for y in l_list:
            row_idx.append(i)
            col_idx.append(y)
            val_idx.append(1)
    m = max(row_idx) + 1
    n = max(col_idx) + 1
    if(M and N):
    	if(N > n):
       		#y_te = y_te.resize((np.shape(y_te)[0], np.shape(y_tr)[1]))
	    	Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, N))
    	elif(N < n):
		Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, N))
        	Y = Y[:, :N]
    else:
	Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
    return [x_text, Y, m, n]


def build_vocab(sentences, params, vocab_size=50000):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    # append <UNK/> symbol to the vocabulary
    vocabulary['<UNK/>'] = len(vocabulary)
    vocabulary_inv.append('<UNK/>')
    vocabulary[params.go_token] = len(vocabulary)
    vocabulary_inv.append(params.go_token)
    vocabulary[params.end_token] = len(vocabulary)
    vocabulary_inv.append(params.end_token)

    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    x = np.array([[vocabulary[word] if word in vocabulary else vocabulary['<UNK/>'] for word in sentence] for sentence in sentences])
    #x = np.array([[vocabulary[word] if word in vocabulary else len(vocabulary) for word in sentence] for sentence in sentences])
    return x


def load_data(params, max_length=500, vocab_size=50000):
    # Load and preprocess data
    with open(os.path.join(params.data_path), 'rb') as fin:
        [train, test, vocab, catgy] = pickle.load(fin)

    # dirty trick to prevent errors happen when test is empty
    if len(test) == 0:
        test[:5] = train[:5]

    trn_sents, Y_trn, m, n = load_data_and_labels(train)
    tst_sents, Y_tst, m, n = load_data_and_labels(test, M=m, N=n)
    sents_padded_sets, params.sequence_length = pad_sentences([trn_sents, tst_sents] , padding_word=params.pad_token, max_length=max_length)
    # tst_sents_padded = pad_sentences(tst_sents, padding_word=params.pad_token, max_length=max_length)
    vocabulary, vocabulary_inv = build_vocab(sents_padded_sets[0] + sents_padded_sets[1], params, vocab_size=vocab_size)
    X_trn = build_input_data(sents_padded_sets[0], vocabulary)
    X_tst = build_input_data(sents_padded_sets[1], vocabulary)
    return X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv, params
    # return X_trn, Y_trn, vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
