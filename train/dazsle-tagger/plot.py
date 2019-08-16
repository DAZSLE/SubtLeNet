import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# histograms

samples = ["BGHToWW", "BGHToZZ"]

inf_dir = 'inference/'
flavor_dir = inf_dir+'flavor_split/'

filenames = {
    "WW": {
        "N2": "BGHToWW_ss.npy",
        "GRU": inf_dir+"BGHToWW_gru_Yhat_all.npy",
        "DNN": inf_dir+"BGHToWW_dnn_Yhat_all.npy",
        "weights": "dazsle_weights_sig.npy",
        "Y": inf_dir+"BGHToWW_Y_all.npy",
        "dnn_base": flavor_dir+"BGHToWW_dnn_Yhat_",
        "gru_base": flavor_dir+"BGHToWW_gru_Yhat_",
        "Y_base": flavor_dir+"BGHToWW_Y_"
        },
    "ZZ": {
        "N2": "BGHToZZ_ss.npy",
        "GRU": inf_dir+"BGHToZZ_gru_Yhat_all.npy",
        "DNN": inf_dir+"BGHToZZ_dnn_Yhat_all.npy",
        "weights": "dazsle_weights_bkg.npy",
        "Y": inf_dir+"BGHToZZ_Y_all.npy",
        "dnn_base": flavor_dir+"BGHToZZ_dnn_Yhat_",
        "gru_base": flavor_dir+"BGHToZZ_gru_Yhat_",
        "Y_base": flavor_dir+"BGHToZZ_Y_"
        }
}

flavors = ['cs', 'ud', 'b']

flavor_labels = {
    '': '',
    'cs': 'cc/ss',
    'ud': 'uu/dd',
    'b': 'bb'
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
        try:
            try:
                arrays[k] = np.load(v[var])[:, :1]
            except:
                arrays[k] = np.load(v[var])
        except:
            arrays[k] = None
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
        #print "working on:", k
        #print "v shape min and max: ", v.shape, v.min(), v.max()
        if weight:
            weights = np.load(filenames[k]['weights'])
            #print "using weights: ", filenames[k]['weights'], len(weights)
            plt.hist(v, bins=bins, density=True, label=k, histtype='step', weights=weights[:v.shape[0]])
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
            if weight:
                weights = np.load(filenames[k]['weights'])
                #print "using weights: ", filenames[k]['weights'], len(weights)
                plt.hist(v, bins=bins, density=True, label=k+"->"+flavor_labels[orig_f], histtype='step', weights=weights[:v.shape[0]])
            else:
                plt.hist(v, bins=bins, density=True, label=k+"->"+flavor_labels[orig_f], histtype='step')
    
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

        fpr_dnn, tpr_dnn, _ = roc_curve(1 - y, dnn_yhat[:, :1])
        fpr_gru, tpr_gru, _ = roc_curve(1 - y, gru_yhat[:, :1])

        plt.plot([0,1], [0,1], 'k--')
        plt.plot(fpr_dnn, tpr_dnn, label='DNN '+flavor_labels[orig_f])
        plt.plot(fpr_gru, tpr_gru, label='GRU '+flavor_labels[orig_f])

    plt.legend(loc='best')
    PdfPages.savefig(out, dpi=100)
    
    return


make_hist("N2", weight=True, title="N2", xlabel="N2")

make_hist("DNN", weight=True, title="DNN", xlabel="Response")
make_hist("GRU", weight=True, title="GRU", xlabel="Response")

make_flavor_hists("DNN", flavors=flavors, weight=True, title="DNN", xlabel="Response")
make_flavor_hists("GRU", flavors=flavors, weight=True, title="GRU", xlabel="Response")

make_roc()
make_roc(flavors=flavors)
make_roc(flavors=['cs', 'ud'])

out.close()
