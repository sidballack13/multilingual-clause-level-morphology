import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers)

    def forward(self, input, hidden_state):
        # batch_size = input.size(0)
        batch_size = len(input)
        encoded = self.encoder(input)
        output, hidden_state = self.rnn(encoded.view(
            1, batch_size, -1), hidden_state)
        return output, hidden_state

    def init_states(self, batch_size):
        # Return a all 0s initial states
        return (torch.zeros([batch_size, self.hidden_size]),
                torch.zeros([batch_size, self.hidden_size]))



class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output = self.decoder(input)
        return output

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

    return (inf_features,vocab,sent,word_indices)

    # print(sent)

inf_features,vocab,sent,word_indices = read_inflection_data("../data/inf/eng.trn")

inf_features = torch.FloatTensor(inf_features)
# print(inf_features)

emb_dim = len(word_indices)
hidden_dim = 300
encoder = Encoder(len(vocab), emb_dim, hidden_dim)

initial_state = encoder.init_states(1)

output, hidden_state = encoder.forward(inf_features, initial_state)

print(output)