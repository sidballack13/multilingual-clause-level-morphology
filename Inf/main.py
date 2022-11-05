import random, time, logging
from asyncio import tasks
import torch, logging, os, argparse
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from model.model import inf_model
from utils import *
from test import test


logger_file_name   = 'experiment4'        # Add ExpNUMBER !!!         
logger_folder_name = "EXPERIMENTS/exp4"   # Add ExpNUMBER !!!

def train(train_loader, val_loader, logger, args):
    scheduler = ReduceLROnPlateau(args.optimizer, mode='min', factor=0.8, patience=3, verbose=True)

    best_loss = 1e4
    start_time = time.time()
    trn_loss_values, trn_acc_values= [], []
    val_loss_values, val_acc_values= [], []
    for epoch in range(args.epochs):
        args.model.train()
        epoch_loss = 0; epoch_acc = 0; epoch_error = 0; epoch_num_tokens = 0;
        epoch_wrong_predictions, epoch_correct_predictions = [], []
        for i, (source_lemma, source_tag, source, target) in enumerate(train_loader):
            args.model.zero_grad()

            # source: (B, Tx), target: (B, Ty)
            source_lemma, source_tag, source, target = source_lemma.to(args.device), source_tag.to(args.device), source.to(args.device), target.to(args.device)
            #print(f"Source Lemma: {source_lemma.shape}")
            #print(f"Source Lemma: {source_tag.shape}")
            #print(f"Source: {source.shape}")
            #print(f"Target: {target.shape}\n")
            
            # (B, Ty)
            target_input  = target[:, :-1]
            target_output = target[:, 1:]
            
            # src_mask: (B, 1 ,Tx), tgt_mask: (B, Ty, Ty)
            src_mask, trg_mask = create_masks(source, target_input, args)
            src_lemma_mask, _  = create_masks(source_lemma, None, args)
            src_tag_mask, _    = create_masks(source_tag, None, args)
            #print(f"Source Lemma Mask: {src_lemma_mask.shape}")
            #print(f"Source Lemma Mask: {src_tag_mask.shape}")
            #print(f"Source Mask: {src_mask.shape}")
            #print(f"Target Mask: {trg_mask.shape}\n")


            loss, acc, output = args.model(source, target_input, target_output, epoch, trg_mask, src_lemma_mask, src_tag_mask, src_mask)

            batch_loss = loss.mean() # optimize for mean loss per token
            batch_loss.backward()
            args.optimizer.step()

            correct_tokens, num_tokens, wrong_tokens, wrong_predictions, correct_predictions = acc
            epoch_loss       += loss.sum().item()
            epoch_num_tokens += num_tokens
            epoch_acc        += correct_tokens
            epoch_error      += wrong_tokens
            epoch_wrong_predictions   += wrong_predictions
            epoch_correct_predictions += correct_predictions
            #print(f"\nBatch: {i+1}/{len(train_loader)} Loss: {batch_loss:.5f} Acc: {correct_tokens/num_tokens:.5f}\n")

        nll_train = epoch_loss / epoch_num_tokens #len(train_loader)
        ppl_train = np.exp(epoch_loss / epoch_num_tokens)
        acc_train = epoch_acc / epoch_num_tokens
        trn_loss_values.append(nll_train)
        trn_acc_values.append(acc_train)
        #print(f"Epoch: {epoch}/{args.epochs} | avg_train_loss: {nll_train:.7f} | perplexity: {ppl_train:.7f} | train_accuracy: {acc_train:.7f}")
        logger.info(f"Epoch: {epoch}/{args.epochs} | avg_train_loss: {nll_train:.7f} | perplexity: {ppl_train:.7f} | train_accuracy: {acc_train:.7f}")

        # File Operations
        if len(wrong_predictions) > 0:
            f1 = open(args.results_file_name + "/"+str(args.epochs)+"epochs_trn_wrong_predictions.txt", "w")
            f2 = open(args.results_file_name + "/"+str(args.epochs)+"epochs_trn_correct_predictions.txt", "w")
            for i in epoch_wrong_predictions:
                f1.write(i+'\n')
            for i in epoch_correct_predictions:
                f2.write(i+'\n')
            f1.close(); f2.close()

        # Validation
        args.model.eval()
        with torch.no_grad():
            nll_test, ppl_test, acc_test = test(val_loader, epoch, logger, args)
            loss = nll_test
        val_loss_values.append(nll_test)
        val_acc_values.append(acc_test)
        scheduler.step(nll_test)

        # Savings
        if loss < best_loss:
            logger.info('Update best val loss\n')
            best_loss = loss
            best_ppl = ppl_test
            torch.save(args.model.state_dict(), args.save_path)

        logging.info("\n")

    end_time = time.time()
    training_time = (abs(end_time - start_time))
    logger.info(f"\n\n---Final Results---")
    #print(f"Epochs: {args.epochs}, Batch Size: {args.batchsize}, lr: {args.lr}, train_loss: {nll_train:.4f}, val_loss: {nll_test:.4f}")
    logger.info(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, lr: {args.lr}, train_loss: {nll_train:.4f}")
    logger.info(f"Training Time: {training_time}\n")
    plot_curves(args.task, args.mname, args.fig, args.axs, trn_loss_values, val_loss_values, args.plt_style, 'loss')


class Parser:
    def __init__(self, part_seperator="\t", tag_seperator=";"):
        
        self.part_seperator = part_seperator
        self.tag_seperator  = tag_seperator 

    def parse_file(self, file):

        data = []
        for line in open(file):

            line =line.rstrip().split(self.part_seperator)
            src_lemma, src_feat, tgt = line[0], line[1], line[2]
            #print(f"Source Lemma: {src_lemma}, Source Feature: {src_feat}, Target: {tgt}\n")

            source_lemma = [char for char in src_lemma]
            source_feat  = src_feat.split(self.tag_seperator)

            source = source_lemma + source_feat
            target = [char for char in tgt]
            
            #print(f"Source: {source}, Target: {target}\n")
            data.append([source_lemma, source_feat, source, target])
        return data
            

class Vocab:
    def __init__(self, data, pad_to=-1, start_token="<s>", eos_token="</s>", pad_token="<p>",  unk_token="<unk>"):

        self.pad_to      = pad_to
        self.start_token = start_token
        self.eos_token   = eos_token
        self.pad_token   = pad_token
        self.unk_token   = unk_token

        default         = {pad_token : 0, start_token : 1, eos_token : 2, unk_token : 3}
        surf_encoder    = dict(**default); surf_decoder = dict();
        surf_decoder[0] = pad_token; surf_decoder[1] = start_token; surf_decoder[2] = eos_token; surf_decoder[3] = unk_token;
        feat_encoder    = dict(**default); feat_decoder = dict();
        feat_decoder[0] = pad_token; feat_decoder[1] = start_token; feat_decoder[2] = eos_token; feat_decoder[3] = unk_token;

        lemmas, tags = [], []
        for sentence in data:
            lemmas.extend(sentence[3])
            tags.extend(sentence[2])  
        
        for j, tag in enumerate(list(set(tags))):
            feat_encoder[tag] = j+4
            feat_decoder[j+4] = tag
            
        for j, surf in enumerate(list(set(lemmas))):
            surf_encoder[surf] = j+4
            surf_decoder[j+4]  = surf

        self.surf_encoder, self.surf_decoder = surf_encoder, surf_decoder
        self.feat_encoder, self.feat_decoder = feat_encoder, feat_decoder

        print(f"Feature Encoder: {self.feat_encoder}")
        print(f"Surface Decoder: {self.feat_decoder}")
        print(f"Surface Encoder: {self.surf_encoder}")
        print(f"Surface Decoder: {self.surf_decoder}\n")
        
        self.data = data

    def encode(self, x):

        src_lemma = []
        for i in self.handle_input(x[0]):
            if i in self.feat_encoder:
                src_lemma.append(self.feat_encoder[i])
            else:
                src_lemma.append(self.feat_encoder['<unk>'])

        src_tag = []
        for i in self.handle_input(x[1]):
            if i in self.feat_encoder:
                src_tag.append(self.feat_encoder[i])
            else:
                src_tag.append(self.feat_encoder['<unk>'])

        src = []
        for i in self.handle_input(x[2]):
            if i in self.feat_encoder:
                src.append(self.feat_encoder[i])
            else:
                src.append(self.feat_encoder['<unk>'])

        tgt = []
        for i in self.handle_input(x[3]):
            if i in self.surf_encoder:
                tgt.append(self.surf_encoder[i])
            else:
                tgt.append(self.surf_encoder['<unk>'])

        src_lemma, src_tag, src, tgt = torch.tensor(src_lemma), torch.tensor(src_tag), torch.tensor(src), torch.tensor(tgt)

        return src_lemma, src_tag, src, tgt

    def decode(self, x):
        return [self.feat_decoder[i] for i in x[2]], [self.surf_decoder[i] for i in x[3]]


    def handle_input(self, x):

        right_padding = self.pad_to - len(x) - 2
        return [self.start_token] + x + [self.eos_token] + [self.pad_token] * right_padding 

        
class WordLoader(Dataset):
    def __init__(self, data, pad_to=-1, start_token="<s>", eos_token="</s>", pad_token="<p>"):
        assert pad_to != -1
        
        self.vocab  = Vocab(data, pad_to=pad_to, start_token=start_token, eos_token=eos_token, pad_token=pad_token)
        self.data   = data

    def __getitem__(self, idx):
        return self.vocab.encode(self.data[idx])
            
    def __len__(self):
        return len(self.data)



# Set loggers 
if not os.path.exists(logger_folder_name):
    os.mkdir(logger_folder_name)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | | %(levelname)s | | %(message)s')
logger_file_name = os.path.join(logger_folder_name, logger_file_name)
file_handler = logging.FileHandler(logger_file_name,'w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info('Code started \n')
# set args
parser      = argparse.ArgumentParser(description='')
args        = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Configurations
args.task       = "task1"
args.pad_to     = 60
args.epochs     = 10
args.batch_size = 64
args.lr         = 5e-4


# Dataset
parser        = Parser()
train_data    = parser.parse_file("./inf/eng.trn")
val_data      = parser.parse_file("./inf/eng.dev")
train_dataset = WordLoader(train_data, pad_to=args.pad_to)
val_dataset   = WordLoader(val_data,   pad_to=args.pad_to)
train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

# Set val vocab same with train vocab
val_dataset.vocab = train_dataset.vocab 
surf_vocab        = train_dataset.vocab.surf_decoder
feature_vocab     = train_dataset.vocab.feat_decoder


# Model
args.mname   = 'Encoder_Decoder'
embed_dim    = 256
num_heads    = 16
dropout_rate = 0.15 
args.model   = inf_model(input_vocab=feature_vocab, output_vocab=surf_vocab, embed_dim=embed_dim, num_heads=num_heads, dropout_rate=dropout_rate)
args.model.to(args.device)

# Loss and optimizer
args.criterion = nn.CrossEntropyLoss(ignore_index=0)
args.optimizer = optim.Adam(args.model.parameters(), lr=args.lr, betas=(0.9, 0.95))

# Terminal operations
logger.info(f"\nUsing device: {str(args.device)}")
logger.info(f"Number of Epochs: {args.epochs}")
logger.info(f"Batch Size: {args.batch_size}")
logger.info(f"Learning rate: {args.lr}")
logger.info(f"Number of parameters {len(torch.nn.utils.parameters_to_vector(args.model.parameters()))}")
logger.info(f"Embedding Dimension: {embed_dim}")
logger.info(f"Number of heads in Attention: {num_heads}")
logger.info(f"Dropout rate: {dropout_rate}\n")

# File Operations
modelname              = args.mname+'/results/'+str(len(train_data))+'_instances'
args.results_file_name = os.path.join(logger_folder_name, modelname)
try:
    os.makedirs(args.results_file_name)
    print("Directory " , args.results_file_name,  " Created ")
except FileExistsError:
    print("Directory " , args.results_file_name,  " already exists")
args.save_path = args.results_file_name + str(args.epochs)+'epochs.pt'
fig_path       = args.results_file_name + str(args.epochs)+'epochs.png'

# Plotting
args.fig, args.axs = plt.subplots(1)
args.plt_style     = pstyle = '-'

# Training
train(train_loader, val_loader, logger, args)

# Save figure
plt.savefig(fig_path)