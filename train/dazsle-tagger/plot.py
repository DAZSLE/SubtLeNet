import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

filenames = {
    #"WW": "BGHToWW_dnn_Yhat.npy",
    #"ZZ": "BGHToZZ_dnn_Yhat.npy"
    #"Dense": "DensetW.npy",
    #"GRU": "GRUtW.npy"
    #"WW": "ww_j_pt.npy",
    #"ZZ": "zz_j_pt.npy"
    #"Jeff": "jeff_weights_bkg.npy",
    #"DAZSLE": "dazsle_weights_bkg.npy"
    "WW": "ww_j_pt.npy",
    "ZZ Unweighted": "zz_j_pt.npy",
    "ZZ Jeff": "zz_j_pt.npy",
    "ZZ DAZSLE": "zz_j_pt.npy"
}

arrays = {}

for k, v in filenames.iteritems():
    arrays[k] = np.load(v).flatten()
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
    "WW": np.ones(len(arrays["WW"])),
    "ZZ Unweighted": np.ones(len(arrays["ZZ Unweighted"])),
    "ZZ Jeff": np.load("jeff_weights_bkg.npy"),
    "ZZ DAZSLE": np.load("dazsle_weights_bkg.npy")
}

out = PdfPages("j_pt_comparison.pdf")

def make_hist():

    plt.figure(figsize=(6, 6), dpi=100)
    plt.title("j_pt comparison")
    plt.xlabel("j_pt")

    min_ = min([min(v) for v in arrays.itervalues()])
    max_ = max([max(v) for v in arrays.itervalues()])
    bins = np.linspace(min_, max_, 100)

    
    for k, v in arrays.iteritems():
        #print "v shape min and max: ", v.shape, '\n', v.min(), '\n', v.max(), '\n'
        print "working on:", k
        #plt.hist(v, bins=bins, density=True, label=k, histtype='step')
        print "using weights: ", weights[k]
        plt.hist(v, bins=bins, density=True, label=k, histtype='step', weights=weights[k])
    
    ''' # weighted vs unweighted
    for k, v in arrays.iteritems():
        plt.hist(v, bins=bins, density=True, label='weighted', histtype='step', weights=weights[k])
        plt.hist(v, bins=bins, density=True, label='unweighted', histtype='step')
    '''
    
    plt.legend(loc='upper right')
    
    PdfPages.savefig(out, dpi=100)

make_hist()
    
out.close()
