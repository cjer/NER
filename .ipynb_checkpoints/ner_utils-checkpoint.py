import numpy as np
import pandas as pd
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Flatten, Concatenate, Conv1D, SpatialDropout1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_accuracy
from keras_contrib.losses import crf_loss
import pickle
import tensorflow as tf
import random as rn
import matplotlib.pyplot as plt
import seaborn as sns
from conlleval import evaluate
import os

sns.set_context('paper')
sns.set_style('white')

# https://universaldependencies.org/tagset-conversion/he-conll-uposf.html

yap_to_ud = {
    'DEF': 'DET',
     'DT': 'DET',
     'DTT': 'DET',
     'REL': 'DET',
     'QW': 'DET',
     'INTJ': 'INTJ',
     'PRP': 'PRON',
     'CDT': 'NUM',
     'CD': 'NUM',
     'NCD': 'NUM',
     'ADVERB': 'ADV',
     'RB': 'ADV',
     'EX': 'ADV',
     'PREPOSITION': 'ADP',
     'IN': 'ADP',
     'NN': 'NOUN',
     'NNT': 'NOUN',
     'BNT': 'NOUN',
     'yySCLN': 'PUNCT',
     'yyCLN': 'PUNCT',
     'yyLRB': 'PUNCT',
     'yyRRB': 'PUNCT',
     'yyDOT': 'PUNCT',
     'yyQUOT': 'PUNCT',
     'yyQM': 'PUNCT',
     'yyEXCL': 'PUNCT',
     'yyCM': 'PUNCT',
     'NNP': 'PROPN',
     'CC': 'CCONJ',
     'CONJ': 'SCONJ',
     'TEMP': 'SCONJ',
     'VB': 'VERB',
     'COP': 'VERB',
     'MD': 'VERB',
     'AT': 'PART',
     'S_PRN': 'PART',
     'POS': 'PART',
     'P': 'PART',
     'JJ': 'ADJ',
     'JJT': 'ADJ',
     'BN': 'VERB',
     'yyDASH': 'PUNCT',
     'yyELPS': 'PUNCT',
     'ZVL': 'X',
     'NEG': 'RB',
     'TTL': 'NOUN',
     'NNPT': 'PROPN',
     ' PREPOSITION': 'ADP', 
     '??': 'X', 
     'DUMMY_AT': 'PART',
     'PREPOSITIONIN': 'ADP',
     'S_ANP': 'X',
    }


embed_dim=300


def initialize_random_seeds(seed=42): 
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(seed)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(seed)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/

    #session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
    #                             inter_op_parallelism_threads=1)

    from keras import backend as K

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


embedding_paths = {
    #wikipedia YAP form
    'yap_w2v_sg':   '../wordembedding-hebrew/vectors_orig_tok/wikipedia.yap_form.word2vec_skipgram.txt',
    'yap_w2v_cbow': '../wordembedding-hebrew/vectors_orig_tok/wikipedia.yap_form.word2vec_cbow.txt',
    'yap_glove':    '../wordembedding-hebrew/vectors_orig_tok/wikipedia.yap_form.glove.txt',
    'yap_ft_sg':    '../wordembedding-hebrew/vectors_orig_tok/wikipedia.yap_form.fasttext_skipgram.model.vec',
    'yap_ft_cbow':  '../wordembedding-hebrew/vectors_orig_tok/wikipedia.yap_form.fasttext_cbow.model.vec',
    # wikipedia.tokenized
    'token_w2v_sg':   '../wordembedding-hebrew/vectors_orig_tok/wikipedia.tokenized.word2vec_skipgram.txt',
    'token_w2v_cbow': '../wordembedding-hebrew/vectors_orig_tok/wikipedia.tokenized.word2vec_cbow.txt',
    'token_glove':    '../wordembedding-hebrew/vectors_orig_tok/wikipedia.tokenized.glove.txt',
    'token_ft_sg':    '../wordembedding-hebrew/vectors_orig_tok/wikipedia.tokenized.fasttext_skipgram.model.vec',
    'token_ft_cbow':  '../wordembedding-hebrew/vectors_orig_tok/wikipedia.tokenized.fasttext_cbow.model.vec',
    # pretrained
    'pretrained_token_ft':    '../fasttext/wiki.he.vec',
    'pretrained_token_ft_cc':    '../fasttext/cc.he.300.vec',

    #wikipedia alternative tokenization YAP form
    'alt_tok_yap_w2v_sg':   '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.word2vec_skipgram.txt',
    'alt_tok_yap_w2v_cbow': '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.word2vec_cbow.txt',
    'alt_tok_yap_glove':    '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.glove.txt',
    'alt_tok_yap_ft_sg':    '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.fasttext_skipgram.model.vec',
    'alt_tok_yap_ft_cbow':  '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.fasttext_cbow.model.vec',
    #wikipedia.tokenized alternative tokenization
    'alt_tok_token_w2v_sg':   '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.tokenized.word2vec_skipgram.txt',
    'alt_tok_token_w2v_cbow': '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.tokenized.word2vec_cbow.txt',
    'alt_tok_token_glove':    '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.tokenized.glove.txt',
    'alt_tok_token_ft_sg':    '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.tokenized.fasttext_skipgram.model.vec',
    'alt_tok_token_ft_cbow':  '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.tokenized.fasttext_cbow.model.vec',
    #wikipedia alternative tokenization YAP form deduped and tuned
    'alt_tok_tuned_yap_ft_sg':    '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.fasttext_skipgram.tuned.model.vec',
    'alt_tok_tuned_yap_ft_cbow':  '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.fasttext_cbow.tuned.model.vec',
    'alt_tok_tuned_yap_w2v_sg':   '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.word2vec_skipgram.tuned.txt',
    'alt_tok_tuned_yap_w2v_cbow': '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.word2vec_cbow.tuned.txt',
    'alt_tok_tuned_yap_glove':    '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.glove.tuned.txt',

}


def get_embedding_matrix(path, word2idx, embed_dim=300, MAX_NB_WORDS=200000, lower_case=False):
    #load embeddings
    print('loading word embeddings:', path)
    embeddings_index = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0].strip('_')
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('found %s word vectors' % len(embeddings_index))
    #embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    nb_words = min(MAX_NB_WORDS, len(word2idx))
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word2idx.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word.strip('_'))
        if lower_case and embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    print("sample words not found: ", np.random.choice(words_not_found, 10))
    return embedding_matrix


from gensim.models.fasttext import load_facebook_model

def get_embedding_matrix_from_fasttext_model(path, word2idx, embed_dim=300, MAX_NB_WORDS=200000):
    model = load_facebook_model(path)
    nb_words = min(MAX_NB_WORDS, len(word2idx))

    embedding_matrix = np.zeros((nb_words, embed_dim))

    for word, i in word2idx.items():
        if i >= nb_words:
            continue
        embedding_vector = model.wv[word.strip('_')]
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def plot_histories(histories, crf=False, **kwargs):
    for h in histories:
        plt.figure()
        if crf:
            plt.plot(h["crf_accuracy"])
            plt.plot(h["val_crf_accuracy"]) 
        else:   
            plt.plot(h["acc"])
            plt.plot(h["val_acc"])
        plt.show()
        
def predict_test_sentence(splits, models, words, i, idx2word, idx2tag, use_word=True, use_pos=True, use_char=False, **kwargs):
    for split, model in zip(splits, models):
        if use_char:
            split, char_split = split
            X_char_tr, X_char_te, _, _ = char_split
        X_tr, X_te, y_tr, y_te, pos_tr, pos_te = split
        params = []
        if use_word:
            params.append(np.array([X_te[i]]))
        if use_pos:
            params.append(np.array([pos_te[i]]))
        if use_char:
            params.append(np.array([X_char_te[i]]))
        p = model.predict(params)
        p = np.argmax(p, axis=-1)
        t = np.argmax(y_te[i], axis=-1)
        print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
        for w, pred, tr in zip(X_te[i], p[0], t):
            if idx2word[w]!="PAD":
                print("{:15} ({:5}): {}".format(idx2word[w], idx2tag[tr], idx2tag[pred]))
                
def predict_on_splits(splits, models, words, idx2word, idx2tag, use_word=True, use_pos=False, use_char=False, predict_on_train=False, **kwargs):
    all_cat_preds = []
    all_cat_y_te = []
    all_words_flat = []
    for split, model in zip(splits, models):
        split, char_split = split
        X_char_tr, X_char_te, _, _ = char_split
        X_tr, X_te, y_tr, y_te, pos_tr, pos_te = split
        
        if predict_on_train:
            X_te, y_te, pos_te = X_tr, y_tr, pos_tr
        params = []
        if use_word:
            params.append(np.array(X_te))
        if use_pos:
            params.append(np.array(pos_te))
        if use_char:
            params.append(np.array(X_char_te))
        preds = model.predict(params)
        preds = np.argmax(preds, axis=-1)
        cat_preds = []
        cat_y_te = []
        words_flat = []
        y_te_num = np.argmax(y_te, axis=-1)
        for ws, s, t in zip(X_te, preds, y_te_num):
            for w, pred, tr in zip(ws, s, t):
                if idx2word[w]!="PAD":
                    words_flat.append(idx2word[w])
                    cat_preds.append(idx2tag[pred].replace('_', '-'))
                    cat_y_te.append(idx2tag[tr].replace('_', '-'))

        all_cat_preds.append(cat_preds)
        all_cat_y_te.append(cat_y_te)
        all_words_flat.append(words_flat)
        
    return (all_cat_preds, all_cat_y_te, all_words_flat)

def predict_only(model, data, idx2word, idx2tag, use_word=True, use_pos=False, use_char=False, **kwargs):
    X_te, pos_te, X_char_te = data
    params = []
    if use_word:
        params.append(np.array(X_te))
    if use_pos:
        params.append(np.array(pos_te))
    if use_char:
        params.append(np.array(X_char_te))
    preds = model.predict(params)
    preds = np.argmax(preds, axis=-1)
    flat_preds = []
    flat_words = []
    all_words = []
    all_preds = []
    for ws, s in zip(X_te, preds):
        s_words = []
        s_preds = []
        for w, pred in zip(ws, s):
            if idx2word[w]!="PAD":
                s_words.append(idx2word[w])
                s_preds.append(idx2tag[pred].replace('_', '-'))
                flat_words.append(idx2word[w])
                flat_preds.append(idx2tag[pred].replace('_', '-'))
        all_words.append(s_words)
        all_preds.append(s_preds)

    return (flat_preds, flat_words, all_words, all_preds)


def replace_pad_with_o(ll):
    new_ll = ['O' if label=='PAD' else label for label in ll]
    return new_ll


def create_model(words, chars, max_len, n_words, n_tags, max_len_char, n_pos, 
                 n_chars, 
                 embedding_mats,
                 use_word=True, use_pos=False, embedding_matrix=None, 
                 embed_dim=70, trainable=True, input_dropout=False, stack_lstm=1,
                 epochs=100, early_stopping=True, patience=20, min_delta=0.0001,
                 use_char=False, crf=False, add_random_embedding=True, pretrained_embed_dim=300,
                 stack_cross=False, stack_double=False, rec_dropout=0.1,
                 validation_split=0.1, output_dropout=False, optimizer='rmsprop', pos_dropout=None,
                 char_dropout=False, all_spatial_dropout=True,
                 print_summary=True, verbose=2):
    X_tr, X_te, y_tr, y_te, pos_tr, pos_te = words
    X_char_tr, X_char_te, _, _ = chars
    all_input_embeds = []
    all_inputs = []
    train_data = []
    if use_word and not add_random_embedding and embedding_matrix is None:
        raise ValueError('Cannot use word without embedding')
    if use_word:
        input = Input(shape=(max_len,))
        if add_random_embedding:
            input_embed = Embedding(input_dim=n_words+2, output_dim=embed_dim, input_length=max_len)(input)
            all_input_embeds.append(input_embed)
        if embedding_matrix is not None:
            input_embed = Embedding(input_dim=n_words+2, output_dim=pretrained_embed_dim, input_length=max_len, 
                                weights=[embedding_mats[embedding_matrix]], trainable=trainable)(input)
            all_input_embeds.append(input_embed)
        all_inputs.append(input)
        train_data.append(X_tr)
    if use_pos:
        pos_input = Input(shape=(max_len,))
        pos_embed = Embedding(input_dim=n_pos+1, output_dim=10, input_length=max_len)(pos_input)
        if pos_dropout is not None:
            pos_embed = Dropout(pos_dropout)(pos_embed)
        all_input_embeds.append(pos_embed)
        all_inputs.append(pos_input)
        train_data.append(pos_tr)
    if use_char:
        # input and embeddings for characters
        char_in = Input(shape=(max_len, max_len_char,))
        emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=20,
                                   input_length=max_len_char))(char_in)
        # character LSTM to get word encodings by characters
        char_enc = TimeDistributed(Bidirectional(LSTM(units=10, return_sequences=False,
                                        recurrent_dropout=0.5)))(emb_char)
        if char_dropout:
            char_enc = SpatialDropout1D(0.3)(char_enc)
        all_input_embeds.append(char_enc)
        all_inputs.append(char_in)
        train_data.append(np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char)))
    if len(all_inputs)>1:
        model = Concatenate()(all_input_embeds)
        if (use_char and all_spatial_dropout):
            model = SpatialDropout1D(0.3)(model)
    else: 
        model = all_input_embeds[0]
        all_input_embeds = all_input_embeds[0]
        all_inputs = all_inputs[0]
        train_data = train_data[0]

    if input_dropout:
        model = Dropout(0.1)(model)

    if stack_double:
        front = LSTM(units=100, return_sequences=True, recurrent_dropout=rec_dropout)(model)
        front = LSTM(units=100, return_sequences=True, recurrent_dropout=rec_dropout)(front)
        back = LSTM(units=100, return_sequences=True, recurrent_dropout=rec_dropout, go_backwards=True)(model)
        model = LSTM(units=100, return_sequences=True, recurrent_dropout=rec_dropout, go_backwards=True)(back)
    if stack_cross:
        front = LSTM(units=100, return_sequences=True, recurrent_dropout=rec_dropout)(model)
        front = LSTM(units=100, return_sequences=True, recurrent_dropout=rec_dropout)(front)
        back = LSTM(units=100, return_sequences=True, recurrent_dropout=rec_dropout, go_backwards=True)(model)
        back = LSTM(units=100, return_sequences=True, recurrent_dropout=rec_dropout, go_backwards=True)(back)
        model = concatenate([back, front])
    for i in range(stack_lstm):
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=rec_dropout))(model)
        
    if output_dropout:
        model = Dropout(0.1)(model)
        
    if crf:
        model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
        crf = CRF(n_tags+1)
        loss = crf_loss
        metric = crf_accuracy
        monitor = 'val_crf_accuracy'
        out = crf(model)
    else:
        out = TimeDistributed(Dense(n_tags+1, activation="softmax"))(model)  # softmax output layer
        loss = "categorical_crossentropy"
        metric = 'accuracy'
        monitor = 'val_acc'

    model = Model(all_inputs, out)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    if early_stopping:
        es = [EarlyStopping(monitor=monitor, mode='max', verbose=1, patience=patience, restore_best_weights=True, min_delta=min_delta)]
    else:
        es=None
    if print_summary:
        print(model.summary())
    history = model.fit(train_data, np.array(y_tr), batch_size=32, epochs=epochs, 
                        validation_split=validation_split, verbose=verbose, callbacks=es)
    hist = pd.DataFrame(history.history)
        
    return model, hist

base_configs_1 = [
            {'crf': True, 'use_pos': False},
            {'crf': True, 'use_pos': True},
            {'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': False, 'embed_dim': embed_dim},
            {'crf': True, 'use_pos': True, 'embedding_matrix': 'all', 'trainable': False, 'embed_dim': embed_dim},
            {'use_char': True, 'crf': True, 'use_pos': False},
            {'use_char': True, 'crf': True, 'use_pos': True},
            {'use_char': True, 'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': False, 'embed_dim': embed_dim},
            {'use_char': True, 'crf': True, 'use_pos': True, 'embedding_matrix': 'all', 'trainable': False, 'embed_dim': embed_dim},
          ]
base_configs = [
            {'crf': True, 'use_pos': False},
            {'crf': True, 'use_pos': True},
            {'add_random_embedding': False, 'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': True, 'embed_dim': embed_dim},
            {'add_random_embedding': False, 'crf': True, 'use_pos': True, 'embedding_matrix': 'all', 'trainable': True, 'embed_dim': embed_dim},
            {'use_char': True, 'crf': True, 'use_pos': False},
            {'use_char': True, 'crf': True, 'use_pos': True},
            {'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': True, 'embed_dim': embed_dim},
            {'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': True, 'embedding_matrix': 'all', 'trainable': True, 'embed_dim': embed_dim},
          ]

base_configs_fixed = [
            {'add_random_embedding': True, 'use_char': True, 'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': False, 'embed_dim': 100},
            {'add_random_embedding': True, 'use_char': True, 'crf': True, 'use_pos': True, 'embedding_matrix': 'all', 'trainable': False, 'embed_dim': 100},
          ]
base_configs_stack = [
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': False, 'stack_lstm': 2},
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': True, 'embedding_matrix': 'all',  'trainable': False, 'stack_lstm': 2},
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': True, 'stack_lstm': 2},
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': True, 'embedding_matrix': 'all', 'trainable': True, 'stack_lstm': 2},
          ]

base_configs_stack2 = [
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': True, 'stack_lstm': 2},
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': True, 'embedding_matrix': 'all', 'trainable': True, 'stack_lstm': 2},
          ]

base_configs_stack_freeze = [
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': False, 'stack_lstm': 2},
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': True, 'embedding_matrix': 'all',  'trainable': False, 'stack_lstm': 2},
          ]

base_configs_stack_freeze_input_dropout = [
            {'input_dropout': True, 'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': False, 'stack_lstm': 2},
            {'input_dropout': True, 'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': True, 'embedding_matrix': 'all',  'trainable': False, 'stack_lstm': 2},
          ]

base_configs_extra_pos_dropout = [
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': True, 'embedding_matrix': 'all',  'trainable': False, 'stack_lstm': 2, 'all_spatial_dropout': True, 'pos_dropout': 0.3},
        ]

base_configs_pos_char_dropout = [
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': True, 'embedding_matrix': 'all',  'trainable': False, 'stack_lstm': 2, 'all_spatial_dropout': False, 'pos_dropout': 0.3, 'char_dropout': True},
        ]

base_configs_stack_no_emb = [
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': True, 'use_char': True, 'crf': True, 'use_pos': False, 'trainable': True, 'stack_lstm': 2},
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': True, 'use_char': True, 'crf': True, 'use_pos': True, 'trainable': True, 'stack_lstm': 2},
          ]

predict_pos = [
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': True, 'use_char': True, 'crf': True, 'use_pos': False, 'trainable': True, 'stack_lstm': 2},
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': True, 'use_char': False, 'crf': True, 'use_pos': False, 'trainable': True, 'stack_lstm': 2},
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': False, 'stack_lstm': 2},
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': False, 'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': False, 'stack_lstm': 2},
          ]

predict_pos_with_emb = [
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': True, 'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': False, 'stack_lstm': 2},
            {'optimizer': 'adam', 'output_dropout': True, 'add_random_embedding': False, 'use_char': False, 'crf': True, 'use_pos': False, 'embedding_matrix': 'all', 'trainable': False, 'stack_lstm': 2},
          ]


def build_configs(configs, embedding_mats):
    new_configs = []
    for conf in configs:
        if 'embedding_matrix' in conf:
            emb = conf['embedding_matrix']
            if type(emb) is str and emb=='all':
                for emb_mat in embedding_mats:
                    new_conf = conf.copy()
                    new_conf['embedding_matrix'] = emb_mat
                    new_configs.append(new_conf)
            else:
                new_configs.append(conf)
        else:
            new_configs.append(conf)

    return new_configs


def run_models(configs, splits, splits_char, embedding_mats,
               words, max_len, n_words, 
               idx2word, idx2tag, n_tags, max_len_char, n_pos, n_chars, 
               epochs=100, run_name=None, out_folder=None, 
               validation_split=0.1,
               skip_if_model_exists=True,
               extra_predictions = None, print_summary=True,
               evaluate_preds = True):
    if run_name is None or out_folder is None:
        raise ValueError
        
    results = []
    preds = []
    histories = []
    
    for i, conf in enumerate(configs):
        model_output_path = os.path.join(out_folder, run_name+'-'+str(i)+'-model')
        if skip_if_model_exists and os.path.exists(model_output_path):
            print('skipping because', model_output_path, 'exists...')
            continue
        mh = [create_model(split, char, max_len, n_words, n_tags, max_len_char,
                           n_pos, n_chars, embedding_mats, epochs=epochs,
                           validation_split=validation_split, print_summary=print_summary,
                           **conf) 
              for split, char in zip(splits, splits_char)]
        hists = [h for m, h in mh]
        models = [m for m, h in mh]
        plot_histories(hists, **conf)
        all_cat_preds, all_cat_y_te, all_words_flat = predict_on_splits(zip(splits, splits_char), models, words, idx2word, idx2tag, **conf)
        all_cat_preds = [replace_pad_with_o(ll) for ll in all_cat_preds]
        if extra_predictions is not None:
            for k, data in enumerate(extra_predictions):
                curr_preds = []
                for model in models:
                    ex_preds = predict_only(model, data, idx2word, idx2tag, **conf)
                    curr_preds.append(ex_preds)
                with open(os.path.join(out_folder, run_name+'-'+str(i)+'-extra_preds-'+str(k)+'.pkl'), 'wb') as f:
                    pickle.dump(curr_preds, f)
                    
        res = []
        if evaluate_preds:
            for cat_y_te, cat_preds in zip(all_cat_y_te, all_cat_preds):
                res.append(evaluate(cat_y_te, cat_preds))
        with open(os.path.join(out_folder, run_name+'-'+str(i)+'-conf_res.pkl'), 'wb') as f:
            pickle.dump([conf, res], f)
        with open(os.path.join(out_folder, run_name+'-'+str(i)+'-conf_preds.pkl'), 'wb') as f:
            pickle.dump([conf, all_cat_preds], f)
        with open(os.path.join(out_folder, run_name+'-'+str(i)+'-conf_hists.pkl'), 'wb') as f:
            pickle.dump([conf, hists], f)
        results.append(res)
        preds.append(all_cat_preds)
        histories .append(hists)
        for j, model in enumerate(models):
            model.save(model_output_path+'-'+str(j)+'.h5') # creates a HDF5 file
            del model
        with open(os.path.join(out_folder, run_name+'_conf_res_preds_hist.pkl'), 'wb') as f:
            pickle.dump(list(zip(configs, results, preds, histories)), f)
    return configs, results, preds, histories


from sklearn.metrics import accuracy_score
    
    
def run_models_pos(configs, splits, splits_char, embedding_mats,
               words, max_len, n_words, 
               idx2word, idx2tag, n_tags, max_len_char, n_pos, n_chars, 
               epochs=100, run_name=None, out_folder=None, 
               validation_split=0.1,
               skip_if_model_exists=True,
               extra_predictions = None, print_summary=True,
               evaluate_preds = True):
    if run_name is None or out_folder is None:
        raise ValueError
        
    results = []
    preds = []
    histories = []
    
    for i, conf in enumerate(configs):
        model_output_path = os.path.join(out_folder, run_name+'-'+str(i)+'-model')
        if skip_if_model_exists and os.path.exists(model_output_path):
            print('skipping because', model_output_path, 'exists...')
            continue
        mh = [create_model(split, char, max_len, n_words, n_tags, max_len_char,
                           n_pos, n_chars, embedding_mats, epochs=epochs,
                           validation_split=validation_split, print_summary=print_summary,
                           **conf) 
              for split, char in zip(splits, splits_char)]
        hists = [h for m, h in mh]
        models = [m for m, h in mh]
        plot_histories(hists, **conf)
        all_cat_preds, all_cat_y_te, all_words_flat = predict_on_splits(zip(splits, splits_char), models, words, idx2word, idx2tag, **conf)
        #all_cat_preds = [replace_pad_with_o(ll) for ll in all_cat_preds]
        if extra_predictions is not None:
            for k, data in enumerate(extra_predictions):
                curr_preds = []
                for model in models:
                    ex_preds = predict_only(model, data, idx2word, idx2tag, **conf)
                    curr_preds.append(ex_preds)
                with open(os.path.join(out_folder, run_name+'-'+str(i)+'-extra_preds-'+str(k)+'.pkl'), 'wb') as f:
                    pickle.dump(curr_preds, f)
                    
        res = []
        if evaluate_preds:
            for cat_y_te, cat_preds in zip(all_cat_y_te, all_cat_preds):
                res.append(accuracy_score(cat_y_te, cat_preds))
        with open(os.path.join(out_folder, run_name+'-'+str(i)+'-conf_res.pkl'), 'wb') as f:
            pickle.dump([conf, res], f)
        with open(os.path.join(out_folder, run_name+'-'+str(i)+'-conf_preds.pkl'), 'wb') as f:
            pickle.dump([conf, all_cat_preds], f)
        with open(os.path.join(out_folder, run_name+'-'+str(i)+'-conf_hists.pkl'), 'wb') as f:
            pickle.dump([conf, hists], f)
        results.append(res)
        preds.append(all_cat_preds)
        histories .append(hists)
        for j, model in enumerate(models):
            model.save(model_output_path+'-'+str(j)+'.h5') # creates a HDF5 file
            del model
        with open(os.path.join(out_folder, run_name+'_conf_res_preds_hist.pkl'), 'wb') as f:
            pickle.dump(list(zip(configs, results, preds, histories)), f)
    return configs, results, preds, histories




