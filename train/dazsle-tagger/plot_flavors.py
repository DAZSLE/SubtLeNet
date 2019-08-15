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
            arrays[k] = np.load(v[var])[:, :1]
        except:
            arrays[k] = np.load(v[var])
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
        #print "v shape min and max: ", v.shape, '\n', v.min(), '\n', v.max()
        if weight:
            #print "using weights: ", weights[k], len(weights[k])
            plt.hist(v, bins=bins, density=True, label=k, histtype='step', weights=np.load(filenames[k]['weights']))
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
        if f != "":
            f = "_"+f
        y = np.concatenate([v for v in make_arrays("Y"+f).itervalues()])
        dnn_yhat = np.concatenate([v for v in make_arrays("DNN"+f).itervalues()])
        gru_yhat = np.concatenate([v for v in make_arrays("GRU"+f).itervalues()])

        fpr_dnn, tpr_dnn, _ = roc_curve(1 - y, dnn_yhat[:, :1])
        fpr_gru, tpr_gru, _ = roc_curve(1 - y, gru_yhat[:, :1])

        plt.plot([0,1], [0,1], 'k--')
        plt.plot(fpr_dnn, tpr_dnn, label='DNN'+f)
        plt.plot(fpr_gru, tpr_gru, label='GRU'+f)

    plt.legend(loc='best')
    PdfPages.savefig(out, dpi=100)
    
    return


make_hist("N2", weight=True, title="N2", xlabel="N2")
make_hist("DNN", weight=True, title="DNN", xlabel="Response")
make_hist("GRU", weight=True, title="GRU", xlabel="Response")
make_roc()

out.close()







'''
dazsle_weights = np.load(basedir+"dazsle_weights_ordered.npy")
i = len(dazsle_weights) - len(arrays["ZZ Unweighted"])

weights = {
    "WW": np.ones(len(arrays["WW"])),
    "ZZ Unweighted": np.ones(len(arrays["ZZ Unweighted"])),
    "ZZ Jeff": np.load(basedir+"jeff_weights_bkg.npy"),
    "ZZ DAZSLE": dazsle_weights[i:]
}'''
