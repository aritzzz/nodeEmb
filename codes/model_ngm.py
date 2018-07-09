
_author_ = '@RajeevVerma'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#from data_loader import *
from tensorboardX import SummaryWriter
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import argparse
from graph import *
#import data_loader
import networkx as nx
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gc
gc.collect()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print 'Using' + str(device)
torch.manual_seed(1)
np.random.seed(1)
if device == "cuda":
	torch.cuda.manual_seed(1)
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

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



writer = SummaryWriter('KC')

class TrainDataset(Dataset):
    def __init__(self):
        walks = np.load('dataset/kc/walks_kc1.npy').astype(int)
        self.len = walks.shape[0]
        self.x_data = walks[:,0:-1]
        self.y_data = walks[:,1:]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
class ValidDataset(Dataset):
    def __init__(self):
        walks = np.load('dataset/bc/validation_bc.npy').astype(int)
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
                help= "no. of epochs to train for", default = 50)
    parser.add_argument("--bppt", type = int, default = 20,
		help = "No. of timestamps to unroll for Truncated Backpropagation")
    parser.add_argument("--rout", default=False, action="store_true",
		help="Set True to use the non-deterministic approximated target distribution")
  
    return parser.parse_args()


def main(args):
    if args.format == 'mat':
        G = readfile(args.input)
    elif args.format == 'edgelist':
        G = from_edgelist(args.input)
    print G.nodes_number()
    walks_corpus = G.build_deepwalk_corpus(args.num_paths, args.path_length)
    model = RNNmodel(G.nodes_number(), args.emb_dim, args.hidden_size, args.batch_size, args.n_layers).to(device)
    optimizer = optim.Adam(model.parameters(),  lr = 0.001)
    train_loader = load_data(TrainDataset(), args.batch_size, args.num_workers)
    #model.load_state_dict(torch.load('t.pt'))
    #model.encoder.weight = torch.nn.Parameter(torch.load('Rrtst_kc.pt')['encoder.weight'])
    #for name,param in model.named_parameters():
	#print str(name) + '\t' + str(param.data)
    #valid_loader = load_data(ValidDataset(), args.batch_size, args.num_workers)
    #loss_ = []
    #lr = np.linspace(0.0000001, 0.1, 10)
    for epoch in range(args.epochs):
	optimizer = exp_lr_scheduler(optimizer, epoch)
	#for param_group in optimizer.param_groups:
        	#param_group['lr'] = lr[epoch]
        loss = train(G, model, optimizer, train_loader, epoch, args.bppt, args.rout)
	#loss_.append(loss)
    	#evaluate(G,model, optimizer, valid_loader, epoch, args.rout)
    torch.save(model.state_dict(), 't.pt')
    #plt.plot(lr, loss_)
    #plt.show()
    

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


def loss(y_pred, y_est, targets, epoch, rout_flag): #pass neigh
    temp = 1
    s = nn.Softmax()
    ls = nn.LogSoftmax()
    l = nn.CrossEntropyLoss()
    ll = nn.KLDivLoss()
    #print y_pred
    #print y_est
    #y_est = s(torch.matmul(embed[neighbors], true)/temp)
    #y_est = hand_conditioning(y_pred, y_est, neighbors)
    #if epoch % 10 == 0:
         #print y_est
    if rout_flag:
	#y_est = hand_conditioning(y_pred, s(true), neighbors)
	y_pred_ = ls(y_pred/temp)
	#print y_est.type()
	#print y_pred_.type()
	#loss = -1*torch.sum(y_est.detach()*y_pred_)
	#loss = l(y_pred, targets) -1*torch.sum(y_est.detach()*y_pred_)
	loss = l(y_pred, targets) + ll(y_pred_, y_est.detach())
	#loss = ll(y_pred_, y_est.detach())
    else:
    	loss = l(y_pred, targets)    #- 10*torch.sum(y_est.detach()*y_pred_)
    #loss = ll(y_pred_, y_est.detach())
    #loss = l(y_pred.view(1,-1), targets)-10*(torch.sum(y_est.detach()*y_pred_))     #/(y_pred.shape[0])
    #loss = l(y_pred.view(1,-1), targets)      #+ ll(y_pred_, y_est.detach())
    return loss


       
   
   

def exp_lr_scheduler(optimizer, epoch, init_lr = 0.001,  lr_decay_epoch=20):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.5**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
    #lr = np.linspace(0.0000001, 0.1, 10).tolist()
    #i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #lr.pop(0)
    #for param_group in optimizer.param_groups:
	#print 'lr is set to ' + str(param_group['lr'])
    return optimizer

class RNNmodel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, batch_size, nlayers, dropout = 0.5):
        super(RNNmodel, self).__init__()
        self.encoder = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size, nlayers, batch_first = True, dropout = dropout)
	#self.drop = nn.Dropout()
        self.decoder = nn.Linear(emb_dim, vocab_size)

        self.init_weights()
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.batch_size = batch_size
	self.vocab_size = vocab_size
        self.decoder.weight = self.encoder.weight    #tying weights
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
    def forward(self,input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
	#output = self.drop(output)
        return output, self.decoder, hidden  
    def init_hidden(self):
        return (torch.zeros( self.nlayers,self.batch_size, self.hidden_size).to(device),
            torch.zeros(self.nlayers,self.batch_size, self.hidden_size).to(device))

    def yield_next(self, G, prev, embedding_lookup):   #Necessary changes to be made for Karate and Blog Catalog Graph
	#print prev	
	prevl = prev + 1
    	prevl = prevl.cpu().numpy().tolist()
    	neighbors =  map(G.get, prevl)          #list(G[prev]) 
    	neighbors = np.array([np.array(n) for n in neighbors]) - 1
	#print neighbors
    	t = nn.Sigmoid()
    	s = nn.Softmax()
    	output = torch.zeros((self.batch_size, self.vocab_size)).to(device)
   	for i in range(neighbors.shape[0]):
    		wts = 1 - t(torch.matmul(embedding_lookup[neighbors[i]], embedding_lookup[prev[i]]))
    		output[i,:] = self.conditioning(s(wts), neighbors[i])
    	return output

    def conditioning(self, y_est, neighbors):
        y_s = torch.zeros(self.vocab_size).to(device)
        for a,b in zip(neighbors, y_est):
    	    y_s[a] = b
        return y_s

      
def load_data(Dset, batch_size, num_workers):
    dataset = Dset
    train_loader = DataLoader( dataset,batch_size = batch_size,shuffle = True, num_workers = num_workers, drop_last = True)
    return train_loader

def train(G, model, optimizer, train_loader, epoch, bppt_len, rout):
    writer = SummaryWriter('KC/train7')
    model.train()
    l = 0
    for i,data in enumerate(train_loader,0):
        model.zero_grad()
        hidden = model.init_hidden()
        for _,j in enumerate(range(0,80, bppt_len)):                        
		inputs, targets = bppt(data, j, bppt_len)
		inputs, targets = Variable(inputs.to(device)), Variable(targets.to(device))
		targets = targets[:,-1]
		hidden = repackage_hidden(hidden)
		if rout:
			embed_lookup = model.encoder.weight
			y_est = model.yield_next(G, inputs[:,-1], embed_lookup)
		else:
			y_est = None
		scores, decoder,hidden = model(inputs, hidden)
		scores = scores.contiguous()[:,-1, :]
		y_pred = decoder(scores)
		los = loss(y_pred, y_est, targets, epoch, rout)
		l += los
		los.backward()
		optimizer.step()
		#if i % 100 == 0:
			#print str(epoch) + str(los.item())
    print 'Train Epoch {} :\t Loss {}'.format(epoch, l.item())   #/len(train_loader))
    writer.add_scalar('train_loss', l.item(), epoch)
    #writer.add_embedding(list(model.parameters())[0])  
    #torch.save(model.state_dict(), 'dt27_KCCE_tst.pt')
    #return l.item()

def evaluate(G, model, optimizer, valid_loader, epoch, rout):
    #writer2 = SummaryWriter('BC/valid')
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            hidden = model.init_hidden()
            for _,j in enumerate(range(0,80,20)):
                inputs, targets = bppt(data, j, 20)
                inputs, targets = Variable(inputs.to(device)), Variable(targets.to(device))
                targets = targets[:,-1]
                scores, decoder, hidden = model(inputs, hidden)
                scores = scores.contiguous()[:,-1,:]
                output = decoder(scores)
                los = loss(output, targets)
                total_loss += los
                hidden = repackage_hidden(hidden)
        print 'Valid Epoch {} :\t Loss {}'.format(epoch, total_loss.item())   #/len(valid_loader))       
        #writer2.add_scalar('valid_loss', total_loss.item()/len(valid_loader), epoch)


if __name__ == "__main__":
    args = args_parse()
    main(args)
