import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#from data_loader import *
from tensorboardX import SummaryWriter

import argparse
from graph import *
#import data_loader
import networkx as nx
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

#torch.backends.cudnn.enabled = False

device = torch.device("cpu")        #("cuda" if torch.cuda.is_available() else "cpu")
print device
torch.manual_seed(1)
np.random.seed(1)
if device == "cuda":
	torch.cuda.manual_seed(1)



"""
Choose most dissimilar node from the neighbors
of the present node as the target of the LSTM
and train the routing equation in an end-to-end
way.
// How to define the cost function??
getting the average neighborhood vector weighted
by dissimilarity coeff. as the average dissimilarity
vector and inner product of the neighbors with the
average dissimilarity vector, softmax to get the prob.
distt. among the neighbors to land on in next move.
This is the approximate target distribution.
Regular CrossEntropyLoss between target distribution
and LSTM predicted distribution.
"""
class GraphDataset(Dataset):
    def __init__(self):
        walks = np.load('walks_dt13th.npy').astype(int)
        self.len = walks.shape[0]
        self.x_data = walks[:,0:-1]
        self.y_data = walks[:,1:]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
def args_parse():
    parser = argparse.ArgumentParser("Implementation of LSTM based node embedding")
    parser.add_argument("--input",
                help = "Path to read graph from", default = 'karate.edgelist')
    parser.add_argument("--format",
                help = "format of the file to read graph from", default = 'edgelist') 
    parser.add_argument("--emb_dim", type = int,
                help = "Dimensions of Embeddings", default = 2)
    parser.add_argument("--path_length", type = int,
                help = "walk length", default = 80)
    parser.add_argument("--num_paths", type = int,
                help = "no. of walks per node", default = 10)
    parser.add_argument("--batch_size", type = int,
                help = "Batch Size for Training", default = 1)
    parser.add_argument("--hidden_size", type = int,
                help = "hidden Size for LSTM", default = 2)
    parser.add_argument("--num_workers", type = int,
                help = "Number of workers for DataLoader", default = 1)
    parser.add_argument("--n_layers", type = int,
                help = "n_layers for Training LSTM", default = 2)
    parser.add_argument("--epochs", type = int,
                help= "no. of epochs to train for", default = 100)
   
    return parser.parse_args()

def yield_next(G, prev, embedding_lookup):   #accepts graph in Networkx format
    if prev == 0:
        prev = 1
    neighbors = list(G[prev])
    neighbors = [neighbor - 1 for neighbor in neighbors]
    #print neighbors
    t = nn.Sigmoid()
    wts = 1 - t(torch.matmul(embedding_lookup[neighbors], embedding_lookup[prev]))
    #neigh = embedding_lookup[neighbors]*(1-t(torch.matmul(embedding_lookup[neighbors],embedding_lookup[prev].view(2,1))))
    #neigh = torch.sum(neigh, dim = 0)
    #print neigh
    #return neigh/len(neighbors), neighbors
    return wts, neighbors


def main(args):
    if args.format == 'mat':
        G = readfile(args.input)
    elif args.format == 'edgelist':
        G = from_edgelist(args.input)
    print G.nodes_number()
    walks_corpus = G.build_deepwalk_corpus(args.num_paths, args.path_length)
    train(G, G.nodes_number(), args.emb_dim, args.hidden_size, args.n_layers, args.batch_size, args.num_workers, args.epochs)


def bppt(data, i, bppt_len):   #for truncated Backpropagation
    inputs, targets = data
    inputs = inputs[:,i:i+bppt_len]
    targets = targets[:,i:i+bppt_len]
    return inputs,targets

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def loss(y_pred, true, targets, neighbors, embed, epoch):
    temp = 1
    s = nn.Softmax()
    ls = nn.LogSoftmax()
    l = nn.CrossEntropyLoss()
    ll = nn.KLDivLoss()
    #y_est = torch.matmul(embed[neighbors], true)
    #y_est = hand_conditioning(y_pred, s(true), neighbors)
    #y_pred_ = ls(y_pred/temp)
    #loss = -1*torch.sum(y_est.detach()*y_pred_)
    loss = l(y_pred.view(1,-1), targets) 
    return loss


def hand_conditioning(y_pred, y_est, neighbors):
    y_s = torch.zeros((y_pred.shape[0]))
    for a,b in zip(neighbors, y_est):
	y_s[a] = b
    return y_s
		
	
	

def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=40):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr*(0.1**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

class RNNmodel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, batch_size, nlayers, dropout = 0.5):
        super(RNNmodel, self).__init__()
        self.encoder = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size, nlayers, batch_first = True, dropout = dropout)
        self.decoder = nn.Linear(emb_dim, vocab_size)

        self.init_weights()
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.batch_size = batch_size
        self.decoder.weight = self.encoder.weight    #tying weights
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
    def forward(self,input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        return output, self.decoder, hidden   
       
    def init_hidden(self):
        return (torch.zeros( self.nlayers,self.batch_size, self.hidden_size).to(device),
            torch.zeros(self.nlayers,self.batch_size, self.hidden_size).to(device))
       
def load_data(batch_size, num_workers):
    dataset = GraphDataset()
    train_loader = DataLoader( dataset,batch_size = batch_size,shuffle = True, num_workers = num_workers, drop_last = True)
    return train_loader

def train(G,vocab_size, emb_dim, hidden_size, n_layers, batch_size, num_workers, epochs):
    model = RNNmodel(vocab_size, emb_dim, hidden_size, batch_size, n_layers).to(device)
    #model.load_state_dict(torch.load('tst.pt')) #, map_location = lambda storage, loc: storage))
    #emb = torch.load('dt26both.pt')['encoder.weight']
    #model.encoder.weight = nn.Parameter(emb)
    #for name,param in model.named_parameters():
        #print str(name) + str(param.data)
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    train_loader = load_data(batch_size, num_workers)
    #writer = SummaryWriter('rout/cuda_dt26KL')
    model.train()
    for epoch in range(epochs+1):
        l = 0
	#optimizer = exp_lr_scheduler(optimizer, epoch)
        for i,data in enumerate(train_loader,0):
            model.zero_grad()
            hidden = model.init_hidden()
            for _,j in enumerate(range(0,80, 20)):                     #20 is the bppt len
                #print j               
                inputs, targets = bppt(data, j, 20)
                inputs, targets = Variable(inputs.to(device)), Variable(targets.to(device))
                targets = targets[:,-1]
                hidden = repackage_hidden(hidden)
                embed_lookup = model.encoder.weight
                neigh, neighbors = yield_next(G, inputs[:,-1].item(), embed_lookup)
                scores, decoder,hidden = model(inputs, hidden)
                #scores = scores.contiguous().view(-1, emb_dim)[-1]
		scores = scores.contiguous()[:,-1, :]
                y_pred = decoder(scores)
                los = loss(y_pred, neigh, targets, neighbors, embed_lookup,epoch)
                l += los
                los.backward()
                optimizer.step()
        print str(epoch) + ':' + str(l.item())
        #writer.add_scalar('loss', l.item(), epoch)
        #writer.add_embedding(list(model.parameters())[0])
        #if epoch % 100 == 0:
            #torch.save(model.state_dict(), 'l'+ str(epoch) + '.pt')
    torch.save(model.state_dict(), 'tst.pt')


if __name__ == "__main__":
    args = args_parse()
    #l = loss(torch.rand(3), torch.rand(3))
    main(args)
