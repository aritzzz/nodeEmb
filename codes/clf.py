import torch
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
#from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


path_model = 'nodeEmb/bcTrains/'
path_labels = 'nodeEmb/dataset/bc/'

model = 'BC2.pt'
mlb = MultiLabelBinarizer(range(39))
def load_model(model):   #returns embeddings and labels
    emb = torch.load(path_model + model, map_location=lambda storage, loc:storage)['encoder.weight'].numpy()
    labels = np.load(path_labels + 'labels_bc.npy')
    labels = mlb.fit_transform(labels)
    return emb,labels

percent = 0.5

emb, labels = load_model(model)
#print labels[0]
indices = np.random.permutation(emb.shape[0])
n = int(emb.shape[0]*percent)
training_idx, test_idx = indices[:n], indices[n:]
training, test = emb[training_idx,:], emb[test_idx,:]
training_labels, test_labels = labels[training_idx,:], labels[test_idx,:]
#print training[0].shape
#print training_labels
clf = OneVsRestClassifier(LogisticRegression())
#clf = BinaryRelevance(LinearSVC())

clf.fit(training, training_labels)
#est =  clf.estimators_
#print clf.classes_
#print clf.multilabel_
#print clf.get_params()
#for e in est:
      #print e.predict(training[1,:].reshape(1,-1))
#print training_labels.sum(axis=0)
label_count = [test_labels.astype(bool)[i,:].sum() for i in range(test_labels.shape[0])]
print label_count[0:5]
predictions =  clf.predict(test[0].reshape(1,-1))
#print test_labels[0]
#print predictions
#print clf.decision_function(test[0].reshape(1,-1))
probs = clf.predict_proba(test)
tst_preds = []
for i, k in enumerate(label_count):
	prob = probs[i,:]
	l = clf.classes_[prob.argsort()[-k:]].tolist()
	tst_preds.append(l)
print len(tst_preds)
print test_labels.shape
pl = mlb.fit_transform(tst_preds)
print pl.shape
#print accuracy_score(test_labels, MultiLabelBinarizer().fit_transform(tst_preds))
print f1_score(test_labels, pl , average = 'macro')




