import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

data = {}

data["j_msd"] = {
    "WW": "WW_j_msd.npy",
    "ZZ": "ZZ_j_msd.npy",
    "QCD": "QCD_j_msd.npy"
}

data["gru_yhat"] = {
    "WW": "inference/BGHToWW_gru_Yhat_all.npy",
    "ZZ": "inference/BGHToZZ_gru_Yhat_all.npy",
    "QCD": "inference/QCD_gru_Yhat_all.npy"
}

data["vidx"] = {
    "WW": "inference/BGHToWW_vidx.npy",
    "ZZ": "inference/BGHToZZ_vidx.npy",
    "QCD": "inference/QCD_vidx.npy"
}

axis_labels = {
    "j_msd": "Jet Mass",
    "gru_yhat": "GRU Response"
}

out = PdfPages("out.pdf")

def make_plot(xvar, yvar, sample, title=""):
    if title == "": title = axis_labels[xvar]+" vs "+axis_labels[yvar]+" for "+sample
    
    plt.figure(figsize=(6, 6), dpi=100)
    plt.xlabel(axis_labels[xvar])
    plt.ylabel(axis_labels[yvar])
    plt.title(title)
    
    x = np.load(data[xvar][sample])
    y = np.load(data[yvar][sample])
    #print "x dims and shape: ", x.ndim, x.shape
    #print "y dims and shape: ", y.ndim, y.shape

    if x.ndim != 1: x = x[:, 0]
    if y.ndim != 1: y = y[:, 0]

    vidx = np.load(data['vidx'][sample])
    x = x[vidx]

    print sample
    print "x dims and shape: ", x.ndim, x.shape
    print "y dims and shape: ", y.ndim, y.shape

    kwargs = {'ls': 'None'}
    
    plt.plot(x, y, 'bo', **kwargs)

    #plt.legend(loc='upper right')

    PdfPages.savefig(out, dpi=100)
    return

def make_heatmap(xvar, yvar, sample, title="", log=False):
    if title == "": title = axis_labels[xvar]+" vs "+axis_labels[yvar]+" for "+sample
    
    plt.figure(figsize=(6, 6), dpi=100)
    plt.xlabel(axis_labels[xvar])
    plt.ylabel(axis_labels[yvar])
    plt.title(title)

    fig, ax = plt.subplots(1, 1)
    
    x = np.load(data[xvar][sample])
    y = np.load(data[yvar][sample])
    #print "x dims and shape: ", x.ndim, x.shape
    #print "y dims and shape: ", y.ndim, y.shape

    if x.ndim != 1: x = x[:, 0]
    if y.ndim != 1: y = y[:, 0]

    vidx = np.load(data['vidx'][sample])
    x = x[vidx]

    print sample
    print "x dims and shape: ", x.ndim, x.shape
    print "y dims and shape: ", y.ndim, y.shape

    if log:
        z = np.exp(-(x**2 + y**2))
        #plt.hist2d(x, y, bins=100, norm=matplotlib.colors.LogNorm(vmin=z.min(), vmax=z.max()))
        pcm = ax.pcolor(x, y, z, norm=matplotlib.colors.LogNorm(vmin=z.min(), vmax=z.max()), cmap='plasma')
        fig.colorbar(pcm, ax=ax, extend='max')
    else:
        plt.hist2d(x, y, bins=100, normed=False, cmap='plasma')

        cb = plt.colorbar()

    cb.set_label('Number of entries')

    PdfPages.savefig(out, dpi=100)
    return

#make_plot("j_msd", "gru_yhat", "WW")
#make_plot("j_msd", "gru_yhat", "ZZ")
#make_plot("j_msd", "gru_yhat", "QCD")

make_heatmap("j_msd", "gru_yhat", "WW", log=True)
make_heatmap("j_msd", "gru_yhat", "ZZ", log=True)
make_heatmap("j_msd", "gru_yhat", "QCD", log=True)

out.close()
