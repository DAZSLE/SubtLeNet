import pandas as pd

base = '/uscms/home/rbisnath/nobackup/pkl_files/cpf_4vect/'

letters = ['x', 'y', 'w', 'ss_vars', 'decayType']

sigs = [pd.read_pickle(base+"BGHToWW_"+l+".pkl") for l in letters]
bkgs = [pd.read_pickle(base+"BGHToZZ_"+l+".pkl") for l in letters]

print "file order is x, y, w, ss_vars, decayType"
print "df.shape for signal files: "
for df in sigs: print df.shape
print "df.shape for background files: "
for df in bkgs: print df.shape

print "number of nan values in signal files: "
for df in sigs: print df.isnull().sum().sum()
print "number of nan values in background files: "
for df in bkgs: print df.isnull().sum().sum()
