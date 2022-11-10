import torch
from torch import nn

import string
import numpy as np

import keras 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


def clean_sentence(sentence):
    # Lower case the sentence
    lower_case_sent = sentence.lower()
    # Strip punctuation
    string_punctuation = string.punctuation + "¡" + '¿'
    clean_sentence = lower_case_sent.translate(str.maketrans('', '', string_punctuation))
   
    return clean_sentence

def tokenize(sentences):
    # Create tokenizer
    text_tokenizer = Tokenizer()
    # Fit texts
    text_tokenizer.fit_on_texts(sentences)
    return text_tokenizer.texts_to_sequences(sentences), text_tokenizer

def read_inflection_data(inf_file):
    f = open(inf_file)
    inf_features = []
    vocab = []
    sent = []
    word_indices = []
    for line in f:
        if len(line.strip()) > 0:
            fields = line.split("\t")
            
            # fields = line.split()

            word = fields[0]
            if word not in vocab:
                vocab.append(word)

            feat = " ".join(fields[1: -1])
            feat = feat.split(";")
            feat.insert(0,word)
            inf_features.append(feat)
            sent.append(fields[-1])
            rep = []
            for s in sent:
                rep.append(s.replace("\n", ""))
            sent = rep

        word_indices.append(vocab.index(word))

    f.close()

    sentences = [clean_sentence(pair) for pair in sent]
    
    return (inf_features,vocab,sentences,word_indices)

    # print(sent)





inf_features,vocab,sentences,word_indices = read_inflection_data("/inf/eng.trn")

# print(sentences)

sent_tokenized, sent_tokenizer = tokenize(sentences)
max_sent_len = int(len(max(sent_tokenized,key=len)))

# inf_features = torch.FloatTensor(inf_features)
# # print(inf_features)

# emb_dim = len(word_indices)
# hidden_dim = 300
# encoder = Encoder(len(vocab), emb_dim, hidden_dim)

# initial_state = encoder.init_states(1)

# output, hidden_state = encoder.forward(inf_features, initial_state)

# print(output)



input_sequence = Input(shape=(len(inf_features[0]),))
embedding = Embedding(input_dim=len(vocab), output_dim=128,)(input_sequence)
encoder = LSTM(64, return_sequences=False)(embedding)

input_sequence = Input(shape=(len(inf_features[0]),))
embedding = Embedding(input_dim=len(vocab), output_dim=128,)(input_sequence)
encoder = LSTM(64, return_sequences=False)(embedding)
r_vec = RepeatVector(max_sent_len)(encoder)
decoder = LSTM(64, return_sequences=True, dropout=0.2)(r_vec)
logits = TimeDistributed(Dense(vocab))(decoder)


enc_dec_model = Model(input_sequence, Activation('softmax')(logits))
enc_dec_model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(1e-3),
              metrics=['accuracy'])
enc_dec_model.summary()


model_results = enc_dec_model.fit(inf_features, sentences, batch_size=30, epochs=100)

def logits_to_sentence(logits, tokenizer):

    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = '<empty>' 

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

index = 14
print("The english sentence is: {}".format(sentences[index]))
print("The spanish sentence is: {}".format(inf_features[index]))
print('The predicted sentence is :')
print(logits_to_sentence(enc_dec_model.predict(inf_features[index:index+1])[0], sent_tokenizer))

