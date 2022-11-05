import torch.nn as nn
from model.sublayers import PositionalEncoding, Feed_Forward
from model.sublayers import Feed_Forward
from model.multihead_attention import MultiHead_Attention

class Encoder_Block(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(Encoder_Block, self).__init__()

        self.mh_attention = MultiHead_Attention(embed_dim, num_heads, dropout_rate)
        self.norm1        = nn.LayerNorm(embed_dim)
        self.feed_forward = Feed_Forward(embed_dim=embed_dim)
        self.norm2        = nn.LayerNorm(embed_dim)
        self.dropout      = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):

        res1 = x
        x    = self.mh_attention(x, x, x, mask)
        x    = self.norm1(x + res1)
        x    = self.dropout(x)
        res2 = x
        x    = self.feed_forward(x)
        x    = self.norm2(x + res2)
        out  = self.dropout(x)
        return out


class Decoder_Block(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(Decoder_Block, self).__init__()
        
        self.mmh_attention = MultiHead_Attention(embed_dim, num_heads, dropout_rate)
        self.norm1         = nn.LayerNorm(embed_dim)
        self.mh_attention  = MultiHead_Attention(embed_dim, num_heads, dropout_rate)
        self.norm2         = nn.LayerNorm(embed_dim)
        self.feed_forward  = Feed_Forward(embed_dim=embed_dim)
        self.norm3         = nn.LayerNorm(embed_dim)
        self.dropout       = nn.Dropout(dropout_rate)

    def forward(self, x, encoder_outputs, target_mask=None, source_mask=None):

        res1 = x
        x    = self.mmh_attention(x, x, x, target_mask)
        x    = self.norm1(x + res1)
        x    = self.dropout(x)
        res2 = x
        x    = self.mh_attention(x, encoder_outputs, encoder_outputs, source_mask)
        x    = self.norm2(x + res2)
        x    = self.dropout(x)
        res3 = x 
        x    = self.feed_forward(x)
        x    = self.norm3(x + res3)
        out  = self.dropout(x)

        return out



class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding    = PositionalEncoding(embed_dim)
        self.encoder1        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder2        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder3        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.dropout         = nn.Dropout(dropout_rate)

    def forward(self, input, source_mask=None):

        x    = self.input_embedding(input)
        x    = self.pos_encoding(x)
        x    = self.encoder1(x, source_mask)
        x    = self.encoder2(x, source_mask)
        out  = self.encoder3(x, source_mask)

        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding    = PositionalEncoding(embed_dim)
        self.decoder1        = Decoder_Block(embed_dim, num_heads, dropout_rate)
        self.decoder2        = Decoder_Block(embed_dim, num_heads, dropout_rate)
        self.decoder3        = Decoder_Block(embed_dim, num_heads, dropout_rate)
        self.dropout         = nn.Dropout(dropout_rate)

    def forward(self, input, word_encoder_outputs, target_mask=None, source_mask=None):

        x   = self.input_embedding(input)
        x   = self.pos_encoding(x)
        x   = self.decoder1(x, word_encoder_outputs, target_mask, source_mask)
        x   = self.decoder2(x, word_encoder_outputs, target_mask, source_mask)
        out = self.decoder3(x, word_encoder_outputs, target_mask, source_mask)

        return out

