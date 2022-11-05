import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import Encoder, Decoder

class analysis_model(nn.Module):
    def __init__(self, input_vocab, output_vocab, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(analysis_model, self).__init__()

        self.surf = input_vocab
        self.feat = output_vocab

        self.encoder    = Encoder(len(input_vocab),  embed_dim, num_heads, dropout_rate)
        self.decoder    = Decoder(len(output_vocab), embed_dim, num_heads, dropout_rate)
        self.linear     = nn.Linear(embed_dim, len(output_vocab))


    def forward(self, input, target_in, target_out, epoch, trg_mask=None, src_mask=None):

        encoder_output = self.encoder(input, src_mask)
        decoder = self.decoder(target_in, encoder_output, trg_mask, src_mask)
        output = self.linear(decoder)
        _output = output.view(-1, output.size(-1))
        _target = target_out.contiguous().view(-1)
        loss = F.cross_entropy(_output, _target, ignore_index=0, reduction='none')
        return loss, self.accuracy(output, target_out, epoch), output
    
    
    def accuracy(self, outputs, targets, epoch):

        surf_vocab = self.surf
        feat_vocab = self.feat

        B = targets.size(0)
        softmax = nn.Softmax(dim=2)

        pred_tokens = torch.argmax(softmax(outputs),2)
        
        correct_tokens = (pred_tokens == targets) * (targets!=0) 
        wrong_tokens   = (pred_tokens != targets) * (targets!=0)

        num_correct = correct_tokens.sum().item() 
        num_wrong   = wrong_tokens.sum().item() 
        num_total   = num_correct + num_wrong 


        correct_predictions = []; wrong_predictions = []
        if (epoch % 5) == 0:
            for i in range(B):
                target  = ''.join([feat_vocab[seq.item()] for seq in targets[i]])
                pred  = ''.join([feat_vocab[seq.item()] for seq in pred_tokens[i]])
                if '</s>' not in target: 
                    continue
                target = target[:target.index('</s>')+4] 
                pred = pred[:len(target)]
                if target != pred:
                    wrong_predictions.append('target: %s pred: %s' % (target, pred))
                else:
                    correct_predictions.append('target: %s pred: %s' % (target, pred))

        return  num_correct, num_total, num_wrong, wrong_predictions, correct_predictions