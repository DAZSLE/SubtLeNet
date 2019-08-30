import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np

f = open("GRU_history.log", "r")
lines = f.readlines()

#acc,loss,val_acc,val_loss

data = []
for l in lines[1:]:
    data.append(l[:-1].split(","))

data = pd.DataFrame(data=data, columns=["accuracy", "loss", "val_acc", "val_loss"])
#print(data.head())
aliases = {
    "accuracy": "training accuracy",
    "loss": "training loss",
    "val_acc": "testing accuracy",
    "val_loss": "testing loss"
}

out = PdfPages("plots/0/acc_loss.pdf")

def make_plot(data, columns, title="", xlabel=""):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.title(title)
    plt.xlabel(xlabel)

    for col in list(columns):
        x = list(np.linspace(1, len(data), len(data)))
        y = list(data[col])
        y = [float(n) for n in y]
        #print(x, y)
        plt.plot(x, y, label=aliases[col])

    plt.legend(loc='upper right')
    
    PdfPages.savefig(out, dpi=100)
    return

make_plot(data, ["accuracy", "val_acc"], title="Accuracy", xlabel="Epoch")
make_plot(data, ["loss", "val_loss"], title="Loss", xlabel="Epoch")

out.close()
