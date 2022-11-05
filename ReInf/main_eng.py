import torch, logging, os, argparse, random, time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from model.model import reinf_model
from training import train
import numpy as np
from utils import *
from test import test


logger_file_name   = 'experiment1'             
logger_folder_name = "EXPERIMENTS/exp1"   



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
        for i, (source, target) in enumerate(train_loader):
            args.model.zero_grad()

            # source: (B, Tx), target: (B, Ty)
            source, target = source.to(args.device), target.to(args.device)

            # (B, Ty)
            target_input  = target[:, :-1]
            target_output = target[:, 1:]
            
            # src_mask: (B, 1 ,Tx), tgt_mask: (B, Ty, Ty)
            src_mask, trg_mask = create_masks(source, target_input, args)

            loss, acc, output = args.model(source, target_input, target_output, epoch, trg_mask, src_mask)

            batch_loss = loss.mean() # optimize for mean loss per token
            batch_loss.backward()
            args.optimizer.step()

            correct_tokens, num_tokens, wrong_tokens, wrong_predictions, correct_predictions, _ = acc
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
            src_tags, src, tgt_tags, tgt = line[0], line[1], line[2], line[3]

            # Create Sperator for word modification
            src = src + ";"

            source_tags  = src_tags.split(self.tag_seperator)
            source_lemma = [char for char in src]
            target_tags  = tgt_tags.split(self.tag_seperator)
            target       = [char for char in tgt]

            source = source_tags + source_lemma + target_tags
            
            #print(f"Source: {source}, Target: {target}\n")
            data.append([source, target])
        return data
            

class Vocab:
    def __init__(self, data, pad_to=-1, start_token="<s>", eos_token="</s>", pad_token="<p>",  unk_token="<unk>"):

        self.pad_to      = pad_to
        self.start_token = start_token
        self.eos_token   = eos_token
        self.pad_token   = pad_token
        self.unk_token   = unk_token

        default           = {pad_token : 0, start_token : 1, eos_token : 2, unk_token : 3}
        source_encoder    = dict(**default); source_decoder = dict();
        source_decoder[0] = pad_token; source_decoder[1] = start_token; source_decoder[2] = eos_token; source_decoder[3] = unk_token;
        target_encoder    = dict(**default); target_decoder = dict();
        target_decoder[0] = pad_token; target_decoder[1] = start_token; target_decoder[2] = eos_token; target_decoder[3] = unk_token;

        sources, targets = [], []
        for sample in data:
            sources.extend(sample[0])
            targets.extend(sample[1])  
        
        for j, tag in enumerate(list(set(targets))):
            target_encoder[tag] = j+4
            target_decoder[j+4] = tag
            
        for j, surf in enumerate(list(set(sources))):
            source_encoder[surf] = j+4
            source_decoder[j+4]  = surf

        self.source_encoder, self.source_decoder = source_encoder, source_decoder
        self.target_encoder, self.target_decoder = target_encoder, target_decoder

        #print(f"Source Encoder: {self.source_encoder}")
        #print(f"Source Decoder: {self.source_decoder}")
        #print(f"Target Encoder: {self.target_encoder}")
        #print(f"Target Decoder: {self.target_decoder}\n")
        
        self.data = data

    def encode(self, x):

        src = []
        for i in self.handle_input(x[0]):
            if i in self.source_encoder:
                src.append(self.source_encoder[i])
            else:
                src.append(self.source_encoder['<unk>'])

        tgt = []
        for i in self.handle_input(x[1]):
            if i in self.target_encoder:
                tgt.append(self.target_encoder[i])
            else:
                tgt.append(self.target_encoder['<unk>'])

        src, tgt = torch.tensor(src), torch.tensor(tgt)

        return src, tgt

    def decode(self, x):
        return [self.source_decoder[i] for i in x[0]], [self.target_decoder[i] for i in x[1]]

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
args.task       = "task2"
args.pad_to     = 200
args.epochs     = 200
args.batch_size = 16
args.lr         = 1e-4


# Dataset
parser        = Parser()
train_data    = parser.parse_file("./reinf/eng.trn") # 10,000
val_data      = parser.parse_file("./reinf/eng.dev") # 1,000
train_dataset = WordLoader(train_data, pad_to=args.pad_to)
val_dataset   = WordLoader(val_data,   pad_to=args.pad_to)
train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
logger.info(f"Batch Length Train Loader: {len(train_loader)}")
logger.info(f"Batch Length Val Loader: {len(val_loader)}")

# Set val vocab same with train vocab
val_dataset.vocab = train_dataset.vocab 
source_vocab      = train_dataset.vocab.source_decoder
target_vocab      = train_dataset.vocab.target_decoder
print(f"surf: {len(source_vocab)}") # 35
print(f"feat: {len(target_vocab)}") # 129


# Model
args.mname   = 'Encoder_Decoder'
embed_dim    = 256
num_heads    = 16
dropout_rate = 0.2 
args.model   = reinf_model(input_vocab=source_vocab, output_vocab=target_vocab, embed_dim=embed_dim, num_heads=num_heads, dropout_rate=dropout_rate)
args.model.to(args.device)

# Loss and optimizer
args.criterion = nn.CrossEntropyLoss(ignore_index=0)
args.optimizer = optim.AdamW(args.model.parameters(), lr=args.lr, betas=(0.9, 0.95))

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