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
    "WW": "ww_j_pt.npy",
    "ZZ": "zz_j_pt.npy"
}

arrays = {}

for k, v in filenames.iteritems():
    arrays[k] = np.load(v).flatten()
    print type(arrays[k]), arrays[k]
    #dfs[k] = pd.DataFrame(data=np.load(v))

#tW = np.load("tW.npy")
print len(arrays['WW'])
ww_weights = np.ones(len(arrays['WW']))
#ww_weights = np.load("ww_weights.npy")
zz_weights = np.load("zz_weights.npy")

#print "dtype of ww_weights element: ", type(ww_weights[0])

weights = {
    "WW": ww_weights,
    "ZZ": zz_weights
}

out = PdfPages("j_pt_weighted.pdf")

def make_hist():

    plt.figure(figsize=(6, 6), dpi=100)
    #plt.xlabel("response")
    #plt.title("response for WW vs ZZ")
    #plt.xlabel("weights")
    #plt.title("weights for DNN and GRU")
    plt.title("weighted j_pt")

    
    for k, v in arrays.iteritems():
        #print "v shape min and max: ", v.shape, '\n', v.min(), '\n', v.max(), '\n'
        print "working on:", k
        bins = np.linspace(min(v), max(v), 100)
        plt.hist(v, bins=bins, density=True, label=k, histtype='step', weights=weights[k])
    '''
    for k, v in arrays.iteritems():
        bins = np.linspace(min(v), max(v), 100)
        plt.hist(v, bins=bins, density=False, label='weighted', histtype='step', weights=weights[k])
        plt.hist(v, bins=bins, density=False, label='unweighted', histtype='step')
    '''
    plt.legend(loc='upper right')
    
    PdfPages.savefig(out, dpi=100)

make_hist()
    
out.close()
