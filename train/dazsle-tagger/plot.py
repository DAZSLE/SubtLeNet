import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# histograms

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

arrays = {}

for k, v in filenames.iteritems():
    arrays[k] = np.load(v).flatten() #for yhat: np.load(v)[:,:1].flatten()
    print type(arrays[k]), arrays[k]

'''    
#print len(arrays['WW'])
ww_weights = np.ones(len(arrays['WW']))
#ww_weights = np.load("ww_weights.npy")
zz_weights = np.load("zz_weights.npy")
#print "dtype of ww_weights element: ", type(ww_weights[0])

weights = {
    "WW": ww_weights,
    "ZZ": zz_weights
}
'''

weights = {
    "WW": np.load("dazsle_weights_sig.npy"),
    "ZZ": np.load("dazsle_weights_bkg.npy")
}

'''
dazsle_weights = np.load("dazsle_weights_ordered.npy")
i = len(dazsle_weights) - len(arrays["ZZ Unweighted"])

weights = {
    "WW": np.ones(len(arrays["WW"])),
    "ZZ Unweighted": np.ones(len(arrays["ZZ Unweighted"])),
    "ZZ Jeff": np.load("jeff_weights_bkg.npy"),
    "ZZ DAZSLE": dazsle_weights[i:]
}'''

out = PdfPages("N2_dazsle_weights.pdf")

def make_hist():

    plt.figure(figsize=(6, 6), dpi=100)
    plt.title("N2")
    plt.xlabel("N2")

    min_ = min([min(v) for v in arrays.itervalues()])
    max_ = max([max(v) for v in arrays.itervalues()])
    bins = np.linspace(min_, max_, 100)

    
    for k, v in arrays.iteritems():
        #print "v shape min and max: ", v.shape, '\n', v.min(), '\n', v.max(), '\n'
        print "working on:", k
        #plt.hist(v, bins=bins, density=True, label=k, histtype='step')
        print "using weights: ", weights[k], len(weights[k]), len(v)
        plt.hist(v, bins=bins, density=True, label=k, histtype='step', weights=weights[k])
    
    ''' # weighted vs unweighted
    for k, v in arrays.iteritems():
        plt.hist(v, bins=bins, density=True, label='weighted', histtype='step', weights=weights[k])
        plt.hist(v, bins=bins, density=True, label='unweighted', histtype='step')
    '''
    
    plt.legend(loc='upper right')
    
    PdfPages.savefig(out, dpi=100)
    return

# roc curve
from sklearn.metrics import roc_curve

y = np.load("combined_Y.npy")
dnn_yhat = np.load("combined_dnn_Yhat.npy")
#print "dnn_yhat:", dnn_yhat[:10]
gru_yhat = np.load("combined_gru_Yhat.npy")

out = PdfPages("roc.pdf")

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


# calls

#make_hist()
make_roc()

out.close()
