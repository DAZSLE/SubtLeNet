#!/usr/bin/env python
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
#from subtlenet.backend.keras_objects import *
#from subtlenet.backend.losses import *
from keras.layers import Dense, BatchNormalization, Input, Dropout, Activation, Concatenate, GRU
from keras.utils import np_utils
from keras.optimizers import Adam, Nadam, SGD
import keras.backend as K
from tensorflow.python.framework import graph_util, graph_io
import os, sys
import numpy as np
import pandas as pd
from collections import namedtuple
import json

import subtlenet.utils as utils 
utils.set_processor('cpu')
VALSPLIT = 0.2 #0.7
MULTICLASS = False
REGRESSION = False
RESHAPE = True
np.random.seed(5)

basedir = '/home/rbisnath/pkl_files/flavor_cut/sv'
#'/uscms/home/rbisnath/nobackup/pkl_files/sv'

Nqcd = 1200000
Nsig = 1200000

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

# retrieving info from j_decayType so we can split response by flavor
decay_key = { #1:u/d, 2:c/s, 3:b, 4:tautau, 5:gluglu, 6:ZZ, 7:WW
    "QCD": 0,
    "ud": 1,
    "cs": 2,
    "b": 3,
    "tautau": 4,
    "gluglu": 5,
    "ZZ": 6,
    "WW": 7
}

# gets indicies (evt#) for a desired decay type
def get_flavor_inds(decays, decay_key):
    flavor_inds = {}
    for k, v in decay_key.iteritems():
        inds = np.where(decays == v)[0]
        if len(inds) != 0: flavor_inds[k] = inds
    return flavor_inds

# tools to reshape data so different kinds of features (cpf vs sv) are grouped
def get_cols(df, base_cols):
    cols = []
    all = list(df.columns)
    for b in base_cols:
        cols.append([c for c in all if b in c])
    return cols

def reshape_df(df, col_names, eles_per_event):
    arrays = []
    for k, v in col_names.iteritems():
        cols = get_cols(df, v)
        cols = [e for l in cols for e in l] #flattening list
        cols = list(dict.fromkeys(cols)) #removing duplicates
        #print "cols: ", cols
        arr = df[cols]
        #print arr.shape, '\n', arr.head()
        arr = arr.values

        n_evts = arr.shape[0]
        n_particles = eles_per_event[k]
        n_feats = len(v)

        arr = np.reshape(arr, (n_evts, n_particles, n_feats))
        arrays.append(arr)
    return arrays


class Sample(object):
    def __init__(self, name, base, max_Y):
        self.name = name
        print self.name

        nrows = Nqcd if 'QCD' in name else Nsig
        
        self.Yhat = {} 
        if args.pkl:
            if RESHAPE:
                self.X = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'x'))[:nrows]
                self.oldX = self.X.values
                self.X = reshape_df(self.X, column_names, elements_per_evt)
            else:
                self.X = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'x')).values[:nrows]
            self.SS = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'ss_vars')).values[:nrows]
            self.W = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'j_pt')).values.flatten()[:nrows] ##### switch 'w' to 'j_pt' if using --make_weights
            self.flatY = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'y')).values.flatten()[:nrows]
            self.decay_type = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'decay_type')).values.flatten()[:nrows]
        else:
            self.X = np.load('%s/%s_%s.npy'%(base, name, 'x'))[:nrows]
            self.SS = np.load('%s/%s_%s.npy'%(base, name, 'ss_vars'))[:nrows]
            self.W = np.load('%s/%s_%s.npy'%(base, name, 'w'))[:nrows]
            self.flatY = np.load('%s/%s_%s.npy'%(base, name, 'y'))[:nrows]
            self.decay_type = np.load('%s/%s_%s.npy'%(base, name, 'decay_type'))[:nrows]
            
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
        self.flavor_inds = get_flavor_inds(self.decay_type, decay_key)
        #print self.name, "self.flavor_inds: ", self.flavor_inds

        #print "s.decay_type: ", self.decay_type.shape, self.decay_type[:5]
        #print "s.Y: ", self.Y.shape, self.Y[:5]

        if args.save_ss:
            np.save(self.name+"_ss.npy", self.SS)

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
        if RESHAPE:
            if 'GRU' in model.name: X = self.X #np.reshape(self.X, (self.X.shape[0], 1, self.X.shape[1], self.X.shape[2])) 
            if 'Dense' in model.name: X = self.oldX #np.reshape(self.oldX, self.oldX.shape[0], self.oldX.shape[1])
        else:
            if 'GRU' in model.name: X = np.reshape(self.X, (self.X.shape[0], 1, self.X.shape[1])) 
            if 'Dense' in model.name: X = np.reshape(self.X, (self.X.shape[0],self.X.shape[1]))
        self.Yhat[model.name] = model.predict(X)
    def standardize(self, mu, std):
        if RESHAPE:
            pass
        self.X = (self.X - mu) / std
    def save_inference(self, model_name='Dense', path='', flavor_split=False):
        if flavor_split:
            inds = self.flavor_inds
            for k, v in inds.iteritems():
                np.save(path+str(self.name)+"_Y_"+k+".npy", self.Y[v])
                if 'Dense' in model_name:
                    np.save(path+str(self.name)+"_dnn_Yhat_"+k+".npy", self.Yhat['Dense'][v])
                elif 'GRU' in model_name:
                    np.save(path+str(self.name)+"_gru_Yhat_"+k+".npy", self.Yhat['GRU'][v])
        else:
            np.save(path+str(self.name)+"_Y_all.npy", self.Y)
            if 'Dense' in model_name:
                np.save(path+str(self.name)+"_dnn_Yhat_all.npy", self.Yhat['Dense'])
            elif 'GRU' in model_name:
                np.save(path+str(self.name)+"_gru_Yhat_all.npy", self.Yhat['GRU'])
        

def calc_ptweights(feat_train,Y_train):
    nevts = len(feat_train)
    nbins = 100
    ptbins = np.linspace(180.,3000.,num=nbins+1)
    sighist = np.zeros(nbins,dtype='f8')
    bkghist = np.zeros(nbins,dtype='f8')
    ptis = np.zeros(nevts)
    ptweights = np.ones(nevts,dtype='f8')
    print "in calc_ptweights, feat_train and Y_train .shape: ", feat_train.shape, Y_train.shape
    for x in range(nevts):
        pti = 0
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
    def __init__(self, n_inputs, h_hidden, n_targets, samples, model, n_categories=0):
        self._hidden = 0
        self.name = model
        self.n_inputs = n_inputs
        self.n_targets = n_targets if MULTICLASS else 2
        self.n_hidden = n_hidden

        if n_categories == 0 and RESHAPE: #DNN and s.X has multiple inputs
            self.tX = np.vstack([s.oldX[:][s.tidx] for s in samples])
            self.vX = np.vstack([s.oldX[:][s.vidx] for s in samples])
        elif n_categories == 0: #DNN or GRU with one input
            self.tX = np.vstack([s.X[:][s.tidx] for s in samples])
            self.vX = np.vstack([s.X[:][s.vidx] for s in samples])
        else: #multiple GRUs
            #print "samples[0].X len, shape of first ele", len(samples[0].X), samples[0].X[0].shape
            self.tX = [np.vstack([s.X[i][s.tidx] for s in samples]) for i in range(n_categories)]
            self.vX = [np.vstack([s.X[i][s.vidx] for s in samples]) for i in range(n_categories)]

        self.tW = np.concatenate([s.W[s.tidx] for s in samples])
        self.vW = np.concatenate([s.W[s.vidx] for s in samples])
        
        self.tflatY = np.concatenate([s.flatY[s.tidx] for s in samples])
        self.vflatY = np.concatenate([s.flatY[s.vidx] for s in samples])
        
        self.tY = np.vstack([s.Y[s.tidx] for s in samples])
        self.vY = np.vstack([s.Y[s.vidx] for s in samples])
        
        self.tSS = np.vstack([s.SS[s.tidx] for s in samples])
        self.vSS = np.vstack([s.SS[s.vidx] for s in samples])

        #print "tW before (i.e. fj_pt): ", self.tW.shape, self.tW[:10]
        ##### uncomment below if using --make_weights / comment out if using imported weights
        
        if args.make_weights and 'Dense' in self.name:
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

        if args.make_weights and 'Dense' in self.name:
            ordered_pts = np.concatenate([s.W for s in samples])
            ordered_ys = np.concatenate([s.flatY for s in samples])
            ordered_weights = calc_ptweights(ordered_pts, ordered_ys)
            #np.save("dazsle_weights_ordered.npy", ordered_weights)
            i = len(samples[0].X)
            sig_weights = ordered_weights[:i]
            bkg_weights = ordered_weights[i:]
            np.save("dazsle_weights_sig.npy", sig_weights)
            np.save("dazsle_weights_bkg.npy", bkg_weights)
            print "finished saving weights"
        
        
        #print "\ntW after: ", self.tW.shape, self.tW[:100]
 
        #normalizing the weights
        try:
            for i in xrange(self.tY.shape[1]):
                tot = np.sum(self.tW[self.tY[:,i] == 1], dtype=np.int64) 
                #print "tot: ", tot
                self.tW[self.tY[:,i] == 1] *= 100.0/tot
                self.vW[self.vY[:,i] == 1] *= 100.0/tot
        except:
            "Error when normalizing weights, check if nevts has changed between now and the last time this script was run with --make_weights"
        
        #print "shapes of vX Y and W: ", self.vX.shape, self.vY.shape, self.vW.shape
        #print "self.tX Y and W", self.tX, "\n", self.tY, "\n", self.tW
        
        if 'GRU' in self.name:
            if n_categories == 0:
                self.tX = np.reshape(self.tX, (self.tX.shape[0], 1, self.tX.shape[1]))
                self.vX = np.reshape(self.vX, (self.vX.shape[0], 1, self.vX.shape[1]))
                self.inputs = Input(shape=(1,self.tX.shape[2]), name='input')
            else:
                self.tX = [np.reshape(tX, (tX.shape[0], tX.shape[1], tX.shape[2])) for tX in self.tX]
                self.vX = [np.reshape(vX, (vX.shape[0], vX.shape[1], vX.shape[2])) for vX in self.vX]
                #for tX in self.tX: print "tX shape: ", tX.shape
                #self.inputs = [Input(shape=(1,tX.shape[2]), name='input') for tX in self.tX]
                self.inputs = [Input(shape=(self.tX[i].shape[1], self.tX[i].shape[2]), name='input_'+str(i)) for i in range(len(self.tX))]

            NPARTS=20
            CLR=0.01
            LWR=0.1
            
            if n_categories == 0:
                h = self.inputs
                gru = GRU(n_inputs,activation='relu',recurrent_activation='hard_sigmoid',name='gru_base')(h)
            else:
                hs = self.inputs
                grus = []
                for i in range(len(hs)):
                    h = hs[i]
                    gru = GRU(n_inputs,activation='relu',recurrent_activation='hard_sigmoid',name='gru_base_'+str(i))(h)
                    #print "gru: ", type(gru), gru.shape, gru
                    grus.append(gru)
                gru = Concatenate()(grus)


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
        history = self.model.fit(self.tX, self.tY, sample_weight=self.tW, 
                                 batch_size=10000, epochs=10, shuffle=True,
                                 validation_data=(self.vX, self.vY, self.vW))

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
    if RESHAPE:
        #X = np.array(np.vstack([np.concatenate([s.X[i].flatten() for i in range(len(s.X))]) for s in samples]), np.float64)
        X = np.array(np.concatenate([s.X[i].flatten() for s in samples for i in range(len(s.X))]))
    else:
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
    parser.add_argument('--save_ss', action='store_true')
    parser.add_argument('--features', type=str, nargs='?', action='store')
    args = parser.parse_args()

    if args.features:
        print "args.features == True"
        with open(args.features) as jsonfile:
            payload = json.load(jsonfile)
            column_names = payload['column_names']
            elements_per_evt = payload['elements_per_evt']
            
    if RESHAPE:
        n_categories = 3
    else:
        n_categories = 0

    figsdir = 'plots/%s/'%(args.version)
    modeldir = 'models/evt/v%i/'%(args.version)
    inferencedir = 'inference/'
    flavordir = inferencedir+'flavor_split/'

    _make_parent(modeldir)
    _make_parent(inferencedir)
    _make_parent(flavordir)
    SIG = 'BGHToWW'
    BKG = 'BGHToZZ'

    models = ['Dense','GRU']
    #models = ['GRU']

    samples = [SIG,BKG]

    samples = [Sample(s, basedir, len(samples)) for s in samples]
    if RESHAPE:
        n_inputs = 115#samples[0].X.shape[1]
        print n_inputs
        print('# sig: ',samples[0].X[0].shape[0], '#bkg: ',samples[1].X[0].shape[0])
    else:
        n_inputs = samples[0].X.shape[1]
        print n_inputs
        print('# sig: ',samples[0].X.shape[0], '#bkg: ',samples[1].X.shape[0])
    
    
    if not RESHAPE:
        print 'Standardizing...'
        mu, std = get_mu_std(samples,modeldir)
        #print "mu, std: ", mu, std
        [s.standardize(mu, std) for s in samples]

    n_hidden = 5
    if 'Dense' in models:
        modelDNN = ClassModel(n_inputs, n_hidden, len(samples),samples,'Dense', n_categories=0)
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
              #np.save(str(s.name)+"_dnn_Yhat.npy", s.Yhat['Dense'])
              #np.save(str(s.name)+"_Y.npy", s.Y)
              s.save_inference(model_name="Dense", path=inferencedir)
              s.save_inference(model_name="Dense", path=flavordir, flavor_split=True)
      

    if 'GRU' in models:
        modelGRU = ClassModel(n_inputs, n_hidden, len(samples),samples,'GRU', n_categories=n_categories)
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
              #np.save(str(s.name)+"_gru_Yhat.npy", s.Yhat['GRU'])
              s.save_inference(model_name="GRU", path=inferencedir)
              s.save_inference(model_name="GRU", path=flavordir, flavor_split=True)

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
