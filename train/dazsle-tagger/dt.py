#!/usr/bin/env python
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
#from subtlenet.backend.keras_objects import *
#from subtlenet.backend.losses import *
from keras.layers import Dense, BatchNormalization, Input, Dropout, Activation, concatenate, GRU
from keras.utils import np_utils
from keras.optimizers import Adam, Nadam, SGD
import keras.backend as K
from tensorflow.python.framework import graph_util, graph_io
import os, sys
import numpy as np
import pandas as pd
from collections import namedtuple

import subtlenet.utils as utils 
utils.set_processor('cpu')
VALSPLIT = 0.2 #0.7
MULTICLASS = False
REGRESSION = False
np.random.seed(5)

basedir = '/home/rbisnath/pkl_files/jet_level'
Nqcd = 1200000
Nsig = 1200000

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

class Sample(object):
    def __init__(self, name, base, max_Y):
        self.name = name 

        nrows = Nqcd if 'QCD' in name else Nsig
        
        self.Yhat = {} 
        if args.pkl:
            self.X = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'x')).values[:nrows]
            self.SS = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'ss_vars')).values[:nrows]
            self.W = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'j_pt')).values.flatten()[:nrows] ##### switch w to j_pt if using --make_weights
            self.flatY = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'y')).values.flatten()[:nrows]
        else:
            self.X = np.load('%s/%s_%s.npy'%(base, name, 'x'))[:nrows]
            self.SS = np.load('%s/%s_%s.npy'%(base, name, 'ss_vars'))[:nrows]
            self.W = np.load('%s/%s_%s.npy'%(base, name, 'w'))[:nrows]
            self.flatY = np.load('%s/%s_%s.npy'%(base, name, 'y'))[:nrows]

        #print self.SS.shape, self.SS[:100]
            
        if MULTICLASS:
            if args.pkl:
                self.Y = np_utils.to_categorical(
                            pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'y')),
                            max_Y
                        )
            else:
                 self.Y = np_utils.to_categorical(
                            np.load('%s/%s_%s.npy'%(base, name, 'y')),
                            max_Y
                        )
        else:
              if args.pkl:
                self.Y = np_utils.to_categorical(
                            (pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'y')).values[:nrows] > 0).astype(np.int),
                            2
                        )
              else:
                self.Y = np_utils.to_categorical(
                            (np.load('%s/%s_%s.npy'%(base, name, 'y'))[:nrows] > 0).astype(np.int),
                            2
                        )
 
        self.idx = np.random.permutation(self.Y.shape[0])
    @property
    def tidx(self):
        if VALSPLIT == 1 or VALSPLIT == 0:
            return self.idx
        else:
            return self.idx[int(VALSPLIT*len(self.idx)):]
    @property
    def vidx(self):
        if VALSPLIT == 1 or VALSPLIT == 0:
            return self.idx
        else:
            return self.idx[:int(VALSPLIT*len(self.idx))]
    def infer(self, model):
        if 'GRU' in model.name: self.X = np.reshape(self.X, (self.X.shape[0], 1, self.X.shape[1])) 
        if 'Dense' in model.name: self.X = np.reshape(self.X, (self.X.shape[0],self.X.shape[1]))
        self.Yhat[model.name] = model.predict(self.X)
    def standardize(self, mu, std):
        self.X = (self.X - mu) / std

def calc_ptweights(feat_train,Y_train):
    nevts = len(feat_train)
    nbins = 100
    ptbins = np.linspace(180.,3000.,num=nbins+1)
    sighist = np.zeros(nbins,dtype='f8')
    bkghist = np.zeros(nbins,dtype='f8')
    ptis = np.zeros(nevts)
    ptweights = np.ones(nevts,dtype='f8')
    #print "len of feat_train, ptbins: ", nevts, len(ptbins)
    print "feat_train and Y_train .shape: ", feat_train.shape, Y_train.shape
    #print feat_train[:10]
    #print "Y_train: ", Y_train[:5]
    for x in range(nevts):
        pti = 1
        while (pti<nbins):
            if (feat_train[x]>ptbins[pti-1] and feat_train[x]<ptbins[pti]): break
            pti = pti+1
        ptis[x] = pti
        if (pti<nbins):
            if (Y_train[x]==1):
                sighist[pti] = sighist[pti]+1.
            else:
                bkghist[pti] = bkghist[pti]+1.
    #print "sighist before norm: ", sighist
    #print "bkghist before norm: ", bkghist
    sighist = sighist/sum(sighist)
    bkghist = bkghist/sum(bkghist)
    #print "sighist after norm: ", sighist
    #print "bkghist after norm: ", bkghist
    ndivisions = 0
    for x in range(nevts):
        if (not (Y_train[x]==0)): continue
        pti = int(ptis[x]) - 1
        sig = sighist[pti]
        bkg = bkghist[pti]
        w = 1
        if sig != 0 and bkg != 0:
            w = sig/bkg             
            ndivisions += 1
        elif sig == 0:
            w = 0.1
        elif bkg == 0:
            w = 10
        #if ndivisions < 100: print "x, w, sig, bkg:", x, w, sig, bkg
        ptweights[x] = w
    #print "ndivisions: ", ndivisions
    #print "ptweights: ", ptweights[:100]
    return ptweights
                                        
class ClassModel(object):
    def __init__(self, n_inputs, h_hidden, n_targets, samples, model):
        self._hidden = 0
        self.name = model
        self.n_inputs = n_inputs
        self.n_targets = n_targets if MULTICLASS else 2
        self.n_hidden = n_hidden

        self.tX = np.vstack([s.X[:][s.tidx] for s in samples])
        self.tW = np.concatenate([s.W[s.tidx] for s in samples])
        self.vX = np.vstack([s.X[:][s.vidx] for s in samples])
        self.vW = np.concatenate([s.W[s.vidx] for s in samples])
        
        self.tflatY = np.concatenate([s.flatY[s.tidx] for s in samples])
        self.vflatY = np.concatenate([s.flatY[s.vidx] for s in samples])
        
        self.tY = np.vstack([s.Y[s.tidx] for s in samples])
        self.vY = np.vstack([s.Y[s.vidx] for s in samples])
        
        self.tSS = np.vstack([s.SS[s.tidx] for s in samples])
        self.vSS = np.vstack([s.SS[s.vidx] for s in samples])

        #print "tW before (i.e. fj_pt): ", self.tW.shape, self.tW[:10]
        ##### uncomment below if using --make_weights
        
        if args.make_weights:
            self.tW = calc_ptweights(self.tW, self.tflatY)
            self.vW = calc_ptweights(self.vW, self.vflatY)
            np.save("tW", self.tW)
            np.save("vW", self.vW)
        else:
            try:
                self.tW = np.load("tW.npy")
                self.vW = np.load("vW.npy")
            except:
                print "Error loading weights from numpy files"

        #print "\ntW after: ", self.tW.shape, self.tW[600000:600100]
                
        #separating the weights into signal/bkg and saving
        if 'Dense' in self.name and args.make_weights:
            signal_weights = []
            bkg_weights = []
        
            for x in range(len(self.tW)):
                if self.tflatY[x] == 1:
                    signal_weights.append(self.tW[x])
                else:
                    bkg_weights.append(self.tW[x])
                    
            for x in range(len(self.vW)):
                if self.vflatY[x] == 1:
                    signal_weights.append(self.vW[x])
                else:
                    bkg_weights.append(self.vW[x])

            np.save("dazsle_weights_signal.npy", signal_weights)
            np.save("dazsle_weights_bkg.npy", bkg_weights)
                
        #normalizing the weights
        for i in xrange(self.tY.shape[1]):
            tot = np.sum(self.tW[self.tY[:,i] == 1], dtype=np.int64) 
            #print "tot: ", tot
            self.tW[self.tY[:,i] == 1] *= 100.0/tot
            self.vW[self.vY[:,i] == 1] *= 100.0/tot
        
        #print "shapes of vX Y and W: ", self.vX.shape, self.vY.shape, self.vW.shape
        #print "self.tX Y and W", self.tX, "\n", self.tY, "\n", self.tW
        
        if 'GRU' in self.name:
            self.tX = np.reshape(self.tX, (self.tX.shape[0], 1, self.tX.shape[1]))
            self.vX = np.reshape(self.vX, (self.vX.shape[0], 1, self.vX.shape[1]))

            self.inputs = Input(shape=(1,self.tX.shape[2]), name='input')
            h = self.inputs
        
            NPARTS=20
            CLR=0.01
            LWR=0.1
     
            gru = GRU(n_inputs,activation='relu',recurrent_activation='hard_sigmoid',name='gru_base')(h)
            dense   = Dense(200, activation='relu')(gru)
            norm    = BatchNormalization(momentum=0.6, name='dense4_bnorm')(dense)
            dense   = Dense(100, activation='relu')(norm)
            norm    = BatchNormalization(momentum=0.6, name='dense5_bnorm')(dense)
            dense   = Dense(50, activation='relu')(norm)
            norm    = BatchNormalization(momentum=0.6, name='dense6_bnorm')(dense)
            dense   = Dense(20, activation='relu')(dense)
            dense   = Dense(10, activation='relu')(dense)
            outputs = Dense(self.n_targets, activation='sigmoid')(norm)
            self.model = Model(inputs=self.inputs, outputs=outputs)

            self.model.compile(loss='binary_crossentropy', optimizer=Adam(CLR), metrics=['accuracy'])
 
        if 'Dense' in self.name:
            self.inputs = Input(shape=(int(n_inputs),), name='input')
            h = self.inputs
            h = BatchNormalization(momentum=0.6)(h)
            if n_inputs > 50: n_inputs = int(n_inputs*0.1)
            for _ in xrange(n_hidden-1):
              h = Dense(int(n_inputs), activation='relu')(h)
              h = BatchNormalization()(h)
            h = Dense(int(n_inputs), activation='tanh')(h)
            h = BatchNormalization()(h)
            self.outputs = Dense(self.n_targets, activation='softmax', name='output')(h)
            self.model = Model(inputs=self.inputs, outputs=self.outputs)
            self.model.compile(optimizer=Adam(),
                               loss='binary_crossentropy')
 
        self.model.summary()


    def train(self, samples):
        #####
        history = self.model.fit(self.tX, self.tY, #sample_weight=self.tW, 
                                 batch_size=10000, epochs=10, shuffle=True,
                                 validation_data=(self.vX, self.vY))#, self.vW))

        with open('history.log','w') as flog:
            history = history.history
            flog.write(','.join(history.keys())+'\n')
            for l in zip(*history.values()):
                flog.write(','.join([str(x) for x in l])+'\n')

    def save_as_keras(self, path):
        _make_parent(path)
        self.model.save(path)
        print 'Saved to',path

    def save_as_tf(self,path):
        _make_parent(path)
        sess = K.get_session()
        print [l.op.name for l in self.model.inputs],'->',[l.op.name for l in self.model.outputs]
        graph = graph_util.convert_variables_to_constants(sess,
                                                          sess.graph.as_graph_def(),
                                                          [n.op.name for n in self.model.outputs])
        p0 = '/'.join(path.split('/')[:-1])
        p1 = path.split('/')[-1]
        graph_io.write_graph(graph, p0, p1, as_text=False)
        print 'Saved to',path

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def load_model(self, path):
        self.model = load_model(path)


def plot(binning, fn, samples, outpath, xlabel=None, ylabel=None):
    hists = {}

    for s in samples:
        h = utils.NH1(binning)
        #print "fn(s): ", fn(s)
        #####
        if type(fn) == int:
            h.fill_array(s.X[s.vidx,fn])#, weights=s.W[s.vidx])
        else:
            h.fill_array(fn(s))#, weights=s.W[s.vidx])
        h.scale()
        hists[s.name] = h
        
    p = utils.Plotter()
    for i,s in enumerate(samples):
        p.add_hist(hists[s.name], s.name, i)

    _make_parent(outpath)

    p.plot(xlabel=xlabel, ylabel=ylabel,
           output = outpath)
    p.plot(xlabel=xlabel, ylabel=ylabel,
           output = outpath + '_logy',
           logy=True)
    p.clear()
    return hists


def get_mu_std(samples, modeldir):
    X = np.array(np.vstack([s.X for s in samples]), np.float64)
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    np.save('standardize_mu.npy',mu)
    np.save('standardize_std.npy',std)

    for it,val in enumerate(np.nditer(mu)):
        if val == 0.: mu[it] = 1.
    for it,val in enumerate(np.nditer(std)):
        if val == 0.: std[it] = 1.

    np.save(modeldir+'standardize_mu.npy',mu)
    np.save(modeldir+'standardize_std.npy',std)
    
    return mu, std

def make_plots(samples):
        for s in samples:
            s.infer(model)

        samples.reverse()
        if REGRESSION:
            plot(np.linspace(60, 160, 20),
                 lambda s : s.Yhat[s.vidx][:,0],
                 samples, figsdir+'mass_regressed', xlabel='Regressed mass')
            plot(np.linspace(60, 160, 20),
                 lambda s : s.Y[s.vidx],
                 samples, figsdir+'mass_truth', xlabel='True mass')
        else:
            roccer_hists = {}
            roccer_hists_n = {}
            roccer_vars_n = {'N2':1}

            for i in xrange(len(samples) if MULTICLASS else 2):
                roccer_hists = plot(np.linspace(0, 1, 100), 
                     lambda s, i=i : s.Yhat[s.vidx,i],
                     samples, figsdir+'class_%i_%s'%(i,args.model), xlabel='Class %i %s'%(i,args.model))
  

                for idx,num in roccer_vars_n.iteritems():
                     roccer_hists_n[idx] = plot(np.linspace(0,1,50),
                     lambda s: s.N2[s.vidx,0], 
                     samples, figsdir+'class_%i_%s'%(i,idx), xlabel='Class %i %s'%(i,idx))


            r1 = utils.Roccer(y_range=range(0,1),axis=[0,1,0,1])
            r1.clear()
            print roccer_hists
            sig_hists = {args.model:roccer_hists[samples[0]],
                'N2':roccer_hists_n['N2'][samples[0]]}

            bkg_hists = {args.model:roccer_hists[samples[1]],
                'N2':roccer_hists_n['N2'][samples[1]]}

            r1.add_vars(sig_hists,           
                        bkg_hists,
                        {args.model:args.model,
                         'N2':'N2'}
            )
            r1.plot(figsdir+'class_%s_%sROC'%(str(args.version),args.model))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dense', action='store_true')
    parser.add_argument('--gru', action='store_true')
    parser.add_argument('--model',nargs='*',type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=2)
    parser.add_argument('--pkl', action='store_true')
    parser.add_argument('--make_weights', action='store_true')
    args = parser.parse_args()

    figsdir = 'plots/%s/'%(args.version)
    modeldir = 'models/evt/v%i/'%(args.version)

    _make_parent(modeldir)
    SIG = 'BGHToWW'
    BKG = 'BGHToZZ'

    models = ['Dense','GRU']

    samples = [SIG,BKG]

    samples = [Sample(s, basedir, len(samples)) for s in samples]
    n_inputs = samples[0].X.shape[1]
    print n_inputs
    print('# sig: ',samples[0].X.shape[0], '#bkg: ',samples[1].X.shape[0])

    print 'Standardizing...'
    mu, std = get_mu_std(samples,modeldir)
    #print "mu, std: ", mu, std
    [s.standardize(mu, std) for s in samples]

    n_hidden = 5
    if 'Dense' in models:
        modelDNN = ClassModel(n_inputs, n_hidden, len(samples),samples,'Dense')
        if args.train:
            print 'Training dense...'
            modelDNN.train(samples)
            modelDNN.save_as_keras(modeldir+'/weights_dense.h5')
            modelDNN.save_as_tf(modeldir+'/graph_dense.pb')
        else:
            print 'Loading dense...'
            modelDNN.load_model(modeldir+'weights_dense.h5')

        if args.plot:
            for s in samples:
              s.infer(modelDNN)
              #print "Yhat: \n", type(s.Yhat), s.Yhat.shape, '\n', s.Yhat
              np.save(str(s.name)+"_dnn_Yhat.npy", s.Yhat['Dense'])
      

    if 'GRU' in models:
        modelGRU = ClassModel(n_inputs, n_hidden, len(samples),samples,'GRU')
        if args.train:
            print 'Training gru...'
            modelGRU.train(samples)
            modelGRU.save_as_keras(modeldir+'/weights_gru.h5')
            modelGRU.save_as_tf(modeldir+'/graph_gru.pb')
        else:
            print 'Loading gru...'
            modelGRU.load_model(modeldir+'weights_gru.h5')
        if args.plot:
            for s in samples:
              s.infer(modelGRU)
              np.save(str(s.name)+"_gru_Yhat.npy", s.Yhat['GRU'])

    if args.plot:

        samples.reverse()
        roccer_hists = {}
        roccer_hists_SS = {}
        SS_vars = {'N2':1}

        sig_hists = {}
        bkg_hists = {} 


        for idx,num in SS_vars.iteritems():
                     roccer_hists_SS[idx] = plot(np.linspace(0,1,100),
                     lambda s: s.SS[s.vidx,0],
                     samples, figsdir+'%s'%(idx), xlabel='%s'%(idx))


        sig_hists['N2'] = roccer_hists_SS['N2'][SIG]    
        bkg_hists['N2'] = roccer_hists_SS['N2'][BKG]    

        for model in models:

            for i in xrange(len(samples) if MULTICLASS else 2):
                
                #for model in ['DNN','GRU']
                roccer_hists = plot(np.linspace(0, 1, 100), 
                       lambda s, i=i : s.Yhat[model][s.vidx,i],
                       samples, figsdir+'class_%i_%s'%(i,model), xlabel='Class %i %s'%(i,model))
  
                sig_hists[model] = roccer_hists[SIG]
                bkg_hists[model] = roccer_hists[BKG] 

        r1 = utils.Roccer(y_range=range(0,1),axis=[0,1,0,1])
        r1.clear()

        r1.add_vars(sig_hists,           
                    bkg_hists,
                    {'Dense':'Dense',
		     'GRU':'GRU',
                     'N2':'N2'}
        )
        r1.plot(figsdir+'class_%s_ROC'%(str(args.version)))
