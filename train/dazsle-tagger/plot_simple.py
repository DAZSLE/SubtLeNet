import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# histograms
'''
filenames = {
    #"WW": "BGHToWW_gru_Yhat.npy",
    #"ZZ": "BGHToZZ_gru_Yhat.npy"
    "WW": "WW_N2.npy",
    "ZZ": "ZZ_N2.npy"
    #"Dense": "DensetW.npy",
    #"GRU": "GRUtW.npy"
    #"WW": "ww_j_pt.npy",
    #"ZZ": "zz_j_pt.npy"
    #"Jeff": "jeff_weights_bkg.npy",
    #"DAZSLE": "dazsle_weights_bkg.npy"
    #"WW": "ww_j_pt.npy",
    #"ZZ Unweighted": "zz_j_pt.npy",
    #"ZZ Jeff": "zz_j_pt.npy",
    #"ZZ DAZSLE": "zz_j_pt.npy"
}
'''

inf_dir = "inference/"

samples = ["BGHToWW", "BGHToZZ"]

# files to use for N2, GRU, etc
N2 = {
    "WW": "BGHToWW_ss.npy",
    "ZZ": "BGHToZZ_ss.npy"
}

Y = {
    "WW": "BGHToWW_Y_all.npy",
    "ZZ": "BGHToZZ_Y_all.npy"
}

GRU = {
    "WW": "BGHToWW_gru_Yhat_all.npy",
    "ZZ": "BGHToZZ_gru_Yhat_all.npy"
}

DNN = {
    "WW": "BGHToWW_dnn_Yhat_all.npy",
    "ZZ": "BGHToZZ_dnn_Yhat_all.npy"
}

weights = {
    "WW": np.load("dazsle_weights_sig.npy"),
    "ZZ": np.load("dazsle_weights_bkg.npy")
}        

out = PdfPages("out.pdf")

def make_arrays(filenames):
    arrays = {}
    basedir = ""
    for k, v in filenames.iteritems():
        if 'Y' in v: basedir = inf_dir
        try:
            arrays[k] = np.load(basedir+v)[:, :1]
        except:
            arrays[k] = np.load(basedir+v)
        #print type(arrays[k]), arrays[k]

    return arrays

def make_hist(filenames, weight=False, title="", xlabel=""):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.title(title)
    plt.xlabel(xlabel)

    arrays = make_arrays(filenames)
    min_ = min([min(v) for v in arrays.itervalues()])
    max_ = max([max(v) for v in arrays.itervalues()])
    bins = np.linspace(min_, max_, 100)

    for k, v in arrays.iteritems():
        #print "working on:", k
        #print "v shape min and max: ", v.shape, '\n', v.min(), '\n', v.max()
        if weight:
            #print "using weights: ", weights[k], len(weights[k])
            plt.hist(v, bins=bins, density=True, label=k, histtype='step', weights=weights[k])
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

ys = [np.load(inf_dir+name+"_Y_all.npy") for name in samples]
y = np.concatenate(ys)
dnn_yhat = np.concatenate([v for v in make_arrays(DNN).itervalues()])
gru_yhat = np.concatenate([v for v in make_arrays(GRU).itervalues()])

def make_roc():

    plt.figure(figsize=(6, 6), dpi=100)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    fpr_dnn, tpr_dnn, _ = roc_curve(y.argmax(axis=1), dnn_yhat[:, :1])
    fpr_gru, tpr_gru, _ = roc_curve(y.argmax(axis=1), gru_yhat[:, :1])

    plt.plot([0,1], [0,1], 'k--')
    plt.plot(fpr_dnn, tpr_dnn, label='DNN')
    plt.plot(fpr_gru, tpr_gru, label='GRU')

    plt.legend(loc='best')
    PdfPages.savefig(out, dpi=100)
    
    return


make_hist(N2, weight=True, title="N2", xlabel="N2")
make_hist(DNN, weight=True, title="DNN", xlabel="Response")
make_hist(GRU, weight=True, title="GRU", xlabel="Response")
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
