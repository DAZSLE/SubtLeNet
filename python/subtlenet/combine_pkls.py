from os import listdir
from os.path import isfile, join

base = '/uscms/home/rbisnath/nobackup/pkl_files/particle_level/'

first_dir = base+'first_half/'
second_dir = base+'second_half/'

#first = [f for f in listdir(first_dir) if isfile(join(first_dir, f))]
#second = [f for f in listdir(second_dir) if isfile(join(second_dir,f))]

first = ['BGHToWW_x.pkl', 'BGHToZZ_x.pkl']
second = ['BGHToWW_x.pkl', 'BGHToZZ_x.pkl']

import pandas as pd

first_dfs = [pd.read_pickle(first_dir+f) for f in first]
second_dfs = [pd.read_pickle(second_dir+f) for f in second]

combined = {}#[pd.concat(f, s) for f in first_dfs second_dfs]
for i in range(len(first_dfs)):
    combined[i] = pd.concat([first_dfs[i], second_dfs[i]], axis=1)

for i in range(len(combined)):
    df = combined[i]
    name = first[i]
    print df.head()
    df.to_pickle(name)

