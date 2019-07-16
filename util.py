from gensim.models import Word2Vec, FastText
import os
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import pandas as pd
from tqdm import tqdm

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    '''
        Directly adopted from keras.preprocessing
    '''
    num_samples = len(sequences)
    lengths = []
    for x in sequences:
        lengths.append(len(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x



class Dictionary(object):
    def __init__(self, n_hid):
        self.word2idx = {0: '<unk>'}
        self.idx2word = ['<unk>']
        self.idx2vec  = [np.zeros(n_hid)]
        self.word_count = defaultdict(int)
        self.len = 1
    def add_word(self, word, w2v):
        word = word.lower()
        self.word_count[word] += 1
        if word not in self.word2idx and word in w2v.wv:
            self.word2idx[word] = self.len
            self.idx2word += [word]
            self.idx2vec  += [w2v.wv[word]]
            self.len += 1
            
    def __len__(self):
        return self.len
    
    def idx2sent(self, x):
        return ' '.join([self.idx2word[i] for i in x])
    def sent2idx(self, x):
        if isinstance(x, type('str')):
            x = x.lower().split()
        return [self.word2idx[w] if w in self.word2idx else 0 for w in x]

def load_training_corpus(w2v, corpus_dir, maxlen = 12, pad = 0, freq_lbound = 16, freq_ubound = 2 ** 16, cxt_lbound = 2, dictionary = None):
    '''
        Use the same word embedding model as Nounce2vec and A la Carte for fair comparison. 
        Note that during training, some of words in Wikitext-103 might not occur in this word embedding.
    '''
    if dictionary == None:
        dictionary = Dictionary(w2v.vector_size)
    else:
        dictionary.word_count = defaultdict(int)
    if 'wikitext' in corpus_dir: 
        c1 = [fi.lower().split()  for fi in tqdm(open(corpus_dir + 'train.txt').readlines(), desc='Load Dataset')]
        c2 = [fi.lower().split()  for fi in open(corpus_dir + 'valid.txt').readlines()]
        c3 = [fi.lower().split()  for fi in open(corpus_dir + 'test.txt').readlines()]
        corpus = c1 + c2 + c3
    elif 'chimeras' in corpus_dir: 
        corpus = []
        for k in [2,4,6]:
            with open(os.path.join(corpus_dir, 'data.l%d.txt' % k), 'r') as f:
                lines = f.readlines()
                for l in lines:
                    fields=l.rstrip('\n').split('\t')
                    corpus += [sent.replace('___', ' <unk> ').split() for sent in fields[1].split('@@')]
        corpus = np.unique(corpus)
    for sent in tqdm(corpus, desc='Build Dictionary'):
        for word in sent:
            dictionary.add_word(word, w2v)
       
    '''
        # test:
        x = "I like playing basketball"
        print(dictionary.idx2sent(dictionary.sent2idx(x)))
    '''
    freq = np.array([fi for fi in list(dictionary.word_count.values()) if fi > 0])
    remove_words = {}
    for word in dictionary.word2idx:
        if word in w2v.wv and word != '<unk>' and w2v.wv.vocab[word].count > freq_lbound and \
            w2v.wv.vocab[word].count < freq_ubound and dictionary.word_count[word] > cxt_lbound:
            '''
                Only Choose words with sufficient occurance (so that we can guarantee the groundtruth embedding)
                but not that many (so that it's not so common) as simulated OOV words
            '''
            remove_words[word] = True
    train_dataset  = {}
    valid_dataset  = {}
    for word, prob in zip(remove_words, np.random.random(len(remove_words))):
        if prob < 0.9:
            '''
                Use 90% for training and 10% for validation
            '''
            train_dataset[word] = [[], []]
        else:
            valid_dataset[word] = [[], []]
    for sent in tqdm(corpus, desc='Tokenizing Corpus'):
        words_valid = []
        words_train = []
        for idx, word in enumerate(sent):
            if word in valid_dataset:
                words_valid += [[word, idx]]
            elif word in train_dataset:
                words_train += [[word, idx]]
        if len(words_valid) > 0 or len(words_train) > 0:
            sent_ids = dictionary.sent2idx(sent)
            if len(words_valid) > 0:
                for word, idx in words_valid:
                    # Only choose those with at most half OOV as contexts
                    if np.count_nonzero(sent_ids[idx-maxlen: idx+1+maxlen]) > maxlen:
                        valid_dataset[word][0] += [sent_ids[idx-maxlen: idx]]
                        valid_dataset[word][1] += [sent_ids[idx+1:  idx+1+maxlen]]
            if len(words_train) > 0:
                for word, idx in words_train:
                    if np.count_nonzero(sent_ids[idx-maxlen: idx+1+maxlen]) > maxlen:
                        train_dataset[word][0] += [sent_ids[idx-maxlen: idx]]
                        train_dataset[word][1] += [sent_ids[idx+1:  idx+1+maxlen]]
    for word in valid_dataset:
        lefts  = pad_sequences(valid_dataset[word][0],  maxlen=maxlen, value=pad, padding='pre',   truncating='pre')
        rights = pad_sequences(valid_dataset[word][1],  maxlen=maxlen, value=pad, padding='post',  truncating='post')    
        valid_dataset[word] = np.concatenate((lefts, rights), axis=1)
    for word in train_dataset:
        lefts  = pad_sequences(train_dataset[word][0],  maxlen=maxlen, value=pad, padding='pre',   truncating='pre')
        rights = pad_sequences(train_dataset[word][1],  maxlen=maxlen, value=pad, padding='post',  truncating='post')    
        train_dataset[word] = np.concatenate((lefts, rights), axis=1)

    print("%d / %d Train words with %d context sentences" % \
          (len(train_dataset), len(freq), np.sum([len(train_dataset[word]) for word in train_dataset])))
    print("%d / %d Valid words with %d context sentences" % \
          (len(valid_dataset), len(freq), np.sum([len(valid_dataset[word]) for word in valid_dataset])))
    return train_dataset, valid_dataset, dictionary



def load_chimera(dictionary, base_w2v, chimera_dir, maxlen = 12, pad = 0):
    _vocab = {v: i+1 for v, i in zip('abcdefghijklmnopqrstuvwxyz', range(26))}
    correct = {}
    with open(os.path.join(chimera_dir, 'dataset.txt'), 'r', encoding='latin1') as f:
        ser = 0
        for line in f.readlines()[1:]:
            if ser % 2 == 0:
                nonce = line[:line.find('_')]
            else:
                correct[nonce] = line.split('\t')[5].split('_')
            ser += 1
    columns=['contexts', 'ground_truth_vector', 'target_word', 'character', 'probes', 'scores', 'text']
    chimera_data = {}
    for k in [2,4,6]:
        chimera_data[k] = {column: [] for column in columns}
        lefts, rights = [], []
        with open(os.path.join(chimera_dir, 'data.l%d.txt' % k), 'r') as f:
            lines = f.readlines()
            for l in lines:
                fields=l.rstrip('\n').split('\t')
                probe = fields[2].split(',')
                nonce = fields[0]
                score = np.array(fields[3].split(','), dtype=np.float)
                sents = [sent.replace('___', ' ___ ').split() for sent in fields[1].split('@@')]
                for sent in sents:
                    idx = sent.index('___')
                    lefts  += [dictionary.sent2idx(sent[:idx])]
                    rights += [dictionary.sent2idx(sent[idx+1:])]
                chimera_data[k]['ground_truth_vector'] += [base_w2v.wv[correct[nonce][0]]]
                chimera_data[k]['target_word'] += [correct[nonce][0]]
                chimera_data[k]['character'] += [[_vocab[v] for v in correct[nonce][0] if v in _vocab]]
                chimera_data[k]['probes'] += [probe]
                chimera_data[k]['scores'] += [score]
                chimera_data[k]['text'] += [sents]
            # end for l in lines:
        # with open(os.path.join(chimera_dir, 'data.l%d.txt' % k), 'r') as f:
        lefts  = pad_sequences(lefts,  maxlen=maxlen, value=pad, padding='pre',  truncating='pre')
        rights = pad_sequences(rights, maxlen=maxlen, value=pad, padding='post',  truncating='post')
        chimera_data[k]['contexts']  = list(np.concatenate((lefts, rights), axis=1).reshape(-1, k, maxlen*2))
        chimera_data[k]['character'] = list(pad_sequences(chimera_data[k]['character'], maxlen=maxlen))
    # end for k in [2,4,6]:
    chimera_data = {k: pd.DataFrame(chimera_data[k], columns=columns) for k in chimera_data}

    print('--------------')
    print('Baseline: Additive')
    for k in [2, 4, 6]:
        data = chimera_data[k]
        oov_cxt = np.array(list(data["contexts"]), dtype=np.int32)
        oov_prb = np.array(list(data["probes"]))
        oov_scr = np.array(list(data["scores"]))
        prov = [[base_w2v.wv[pi] for pi in probe] for probe in oov_prb]
        pred = [np.average([dictionary.idx2vec[pi] for pi in pp if pi != pad], axis=0) for pp in oov_cxt.reshape(oov_cxt.shape[0], -1)]
        cors = []
        for p1, p2, p3 in zip(pred, prov, oov_scr):
            cos = cosine_similarity([p1], p2)
            cor = spearmanr(cos[0], p3)[0]
            cors += [cor]
        r2 = np.average(cors)
        print(r2)
    return chimera_data