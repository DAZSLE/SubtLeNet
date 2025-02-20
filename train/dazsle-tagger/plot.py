import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# histograms

#samples = ["BGHToWW", "BGHToZZ"]
samples = ["QCD"]

inf_dir = 'inference/'
flavor_dir = inf_dir+'flavor_split/'

filenames = {
    "WW": {
        "N2": "BGHToWW_ss.npy",
        "GRU": inf_dir+"BGHToWW_gru_Yhat_all.npy",
        "DNN": inf_dir+"BGHToWW_dnn_Yhat_all.npy",
        #"weights": "dazsle_weights_sig.npy",
        "weights": "vW.npy",
        "Y": inf_dir+"BGHToWW_Y_all.npy",
        "dnn_base": flavor_dir+"BGHToWW_dnn_Yhat_",
        "gru_base": flavor_dir+"BGHToWW_gru_Yhat_",
        "Y_base": flavor_dir+"BGHToWW_Y_"
        },
    "ZZ": {
        "N2": "BGHToZZ_ss.npy",
        "GRU": inf_dir+"BGHToZZ_gru_Yhat_all.npy",
        "DNN": inf_dir+"BGHToZZ_dnn_Yhat_all.npy",
        #"weights": "dazsle_weights_bkg.npy",
        "weights": "vW.npy",
        "Y": inf_dir+"BGHToZZ_Y_all.npy",
        "dnn_base": flavor_dir+"BGHToZZ_dnn_Yhat_",
        "gru_base": flavor_dir+"BGHToZZ_gru_Yhat_",
        "Y_base": flavor_dir+"BGHToZZ_Y_"
        },
    "QCD": {
        "N2": "QCD_ss.npy",
        "GRU": inf_dir+"QCD_gru_Yhat_all.npy",
        "DNN": inf_dir+"QCD_dnn_Yhat_all.npy",
        #"weights": "dazsle_weights_bkg.npy",
        "weights": "vW.npy",
        "Y": inf_dir+"QCD_Y_all.npy",
        "dnn_base": flavor_dir+"QCD_dnn_Yhat_",
        "gru_base": flavor_dir+"QCD_gru_Yhat_",
        "Y_base": flavor_dir+"QCD_Y_"
        }
}

flavors = ['cs', 'ud', 'b']

flavor_labels = {
    "WW": {
        '': '',
        'cs': 'cs',
        'ud': 'light',
        'b': 'bb'
    },
    "ZZ": {
        '': '',
        'cs': 'cc/ss',
        'ud': 'light',
        'b': 'bb'
    },
    "QCD": {
        '': '',
        'cs': 'cs',
        'ud': 'ud',
        'b': 'b'
    },
    "roc": {
        '': '',
        'cs': 'W->cs, Z->cc/ss',
        'ud': 'W->light, Z->light',
        'b': 'Z->bb'
    }
}

def make_flavor_filenames():
    for k, v in filenames.iteritems():
        dnn_base = v['dnn_base']
        gru_base = v['gru_base']
        Y_base = v['Y_base']
        for f in flavors:
            v['DNN_'+f] = dnn_base+f+'.npy'
            v['GRU_'+f] = gru_base+f+'.npy'
            v['Y_'+f] = Y_base+f+'.npy'

make_flavor_filenames()

out = PdfPages("out.pdf")

def make_arrays(var):
    arrays = {}
    for k, v in filenames.iteritems():
        '''
        try:
            try:
                arrays[k] = np.load(v[var])[:, :1]
            except:
                arrays[k] = np.load(v[var])
        except:
            arrays[k] = None
        '''
        try:
            arr = np.load(v[var])
        except:
            #print "Couldn't load: ", v[var]
            continue
        #print "In make_arrays, ", var, arr.ndim
        if arr.ndim == 1:
            arrays[k] = arr
        elif arr.ndim == 2:
            arrays[k] = arr[:, :1]
        else:
            print "make arrays: got input w more than 2 dims"
        #print type(arrays[k]), arrays[k]

    return arrays

def make_hist(var, weight=False, title="", xlabel=""):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.title(title)
    plt.xlabel(xlabel)

    arrays = make_arrays(var)
    min_ = min([min(v) for v in arrays.itervalues()])
    max_ = max([max(v) for v in arrays.itervalues()])
    bins = np.linspace(min_, max_, 100)

    for k, v in arrays.iteritems():
        #print "working on hist for:", k
        if weight:
            weights = np.load(filenames[k]['weights'])
            n = min(len(weights), v.shape[0])
            v = v[:n]
            weights = weights[:n]
            #print "v shape min and max: ", v.shape, v.min(), v.max()
            #print "using weights: ", filenames[k]['weights'], len(weights)
            plt.hist(v, bins=bins, density=True, label=k, histtype='step', weights=weights)
        else:
            plt.hist(v, bins=bins, density=True, label=k, histtype='step')

    
    ''' # plot weighted vs unweighted
    for k, v in arrays.iteritems():
        plt.hist(v, bins=bins, density=True, label='weighted', histtype='step', weights=weights[k])
        plt.hist(v, bins=bins, density=True, label='unweighted', histtype='step')
    '''
    
    plt.legend(loc='upper right')
    
    PdfPages.savefig(out, dpi=100)
    return

def make_flavor_hists(var, flavors=[""], weight=False, title="", xlabel=""):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.title(title)
    plt.xlabel(xlabel)

    for f in flavors:
        orig_f = f
        if f != "":
            f = "_"+f
        
        arrays = make_arrays(var+f)

        min_ = min([min(v) for v in arrays.itervalues() if not v is None])
        max_ = max([max(v) for v in arrays.itervalues() if not v is None])
        bins = np.linspace(min_, max_, 100)

        for k, v in arrays.iteritems():
            if v is None: continue
            #print "working on:", k
            #print "v shape min and max: ", v.shape, v.min(), v.max()
            #v = v[np.logical_not(np.isnan(v))]
            label = k+"->"+flavor_labels[k][orig_f]
            if weight:
                weights = np.load(filenames[k]['weights'])
                #weights = weights[np.logical_not(np.isnan(weights))]
                n = min(len(weights), v.shape[0])
                v = v[:n]
                weights = weights[:n]
                #print "v.shape after trim", v.shape
                #print "using weights: ", filenames[k]['weights'], len(weights)
                plt.hist(v, bins=bins, density=True, label=label, histtype='step', weights=weights)
            else:
                plt.hist(v, bins=bins, density=True, label=label, histtype='step')
    
    plt.legend(loc='upper right')
    PdfPages.savefig(out, dpi=100)
    return

# roc curve
from sklearn.metrics import roc_curve

def make_roc(flavors=[""]):

    plt.figure(figsize=(6, 6), dpi=100)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    for f in flavors:
        orig_f = f
        if f != "":
            f = "_"+f
        try:
            y = np.concatenate([v for v in make_arrays("Y"+f).itervalues()])
            dnn_yhat = np.concatenate([v for v in make_arrays("DNN"+f).itervalues()])
            gru_yhat = np.concatenate([v for v in make_arrays("GRU"+f).itervalues()])
        except IOError:
            continue
        except Exception as e:
            print e

        fpr_dnn, tpr_dnn, _ = roc_curve(1 - y, dnn_yhat)
        print "gru y, yhat .shape: ", y.shape, gru_yhat.shape
        fpr_gru, tpr_gru, _ = roc_curve(1 - y, gru_yhat)

        label = flavor_labels['roc'][orig_f]
        plt.plot([0,1], [0,1], 'k--')
        plt.plot(fpr_dnn, tpr_dnn, label='DNN '+label)
        plt.plot(fpr_gru, tpr_gru, label='GRU '+label)

    plt.legend(loc='best')
    PdfPages.savefig(out, dpi=100)
    
    return


#make_hist("N2", weight=True, title="N2", xlabel="N2")

make_hist("DNN", weight=True, title="DNN", xlabel="Response")
make_hist("GRU", weight=True, title="GRU", xlabel="Response")

#make_flavor_hists("DNN", flavors=flavors, weight=True, title="DNN", xlabel="Response")
#make_flavor_hists("GRU", flavors=flavors, weight=True, title="GRU", xlabel="Response")

#make_roc()
#make_roc(flavors=['cs', 'ud'])

out.close()
