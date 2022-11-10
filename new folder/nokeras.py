import torch
from torch import nn
import pandas as pd


def read_inflection_data(inf_file):
    df = pd.read_csv(inf_file, sep = '\t', header = None)
    df.columns = ["words", "inflectional features", "sentences"]
    df['sentences'] = df['sentences'].apply(lambda x: x.lower())

    df['words'] = df['words'].apply(lambda x : 'START_' + x + '_END')

    sent_words_vocab = set()
    for sent in df['sentences']:
        for wrd in sent.split():
            if wrd not in sent_words_vocab:
                sent_words_vocab.add(wrd)

    words_vocab = set()
    for wrd in df['words']:
        if wrd not in words_vocab:
            words_vocab.add(wrd)

    for line in df['inflectional features']:
        for feat in line.split(';'):
            if feat not in words_vocab:
                words_vocab.add(feat)

    return (df, words_vocab,sent_words_vocab)

    # print(sent)

df , words_vocab,sent_words_vocab = read_inflection_data("../data/inf/eng.trn")
words_vocab = sorted(list(words_vocab))
sent_words_vocab = sorted(list(sent_words_vocab))

print(df.head)

# num_encoder_tokens = len(words_vocab)
# num_decoder_tokens = len(sent_words_vocab) + 1

# inp_token_index = dict([(word, i+1) for i, word in enumerate(words_vocab)])
# target_token_index = dict([(word, i+1) for i, word in enumerate(sent_words_vocab)])

# reverse_input_char_index = dict((i,word) for word,i in inp_token_index.items())
# reverse_target_char_index = dict((i,word) for word,i in target_token_index.items())

# X , y = df[]


