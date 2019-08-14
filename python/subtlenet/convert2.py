import argparse
import os
import json
import uproot
import numpy as np
import pandas as pd
import re

# grabbing command line args
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, help='dir where output files will be stored')
parser.add_argument('--name', type=str, help='name of sample to process')
parser.add_argument('--json', type=str, help='json file controlling which files/samples/features/cuts are used')
parser.add_argument('--background', action='store_true', help='use background cut instead of signal cut, also sets y to 0')
parser.add_argument('--verbosity', type=int, nargs='?', default=1, help='0-no printing, 1-print df.head() of output files (default), 2-print info about x at different stages and 1, 3-print the list of variables available in the input .root file')
parser.add_argument('--dry', action='store_true', help='if enabled, runs the whole program but doesn\'t save to .pkl files')

args = parser.parse_args()

VERBOSITY = args.verbosity

if (("Background" in args.name) or ("QCD" in args.name)) and not args.background:
    print "\n warning: must include '--background' to apply background cut (defaults to signal cut)\n"

# making the output direrctory in case it doesn't already exist
try:
    os.makedirs(args.out)
except:
    pass

# parsing the json file
with open(args.json) as jsonfile:
    payload = json.load(jsonfile)
    basedir = payload['base']
    features = payload['features']
    weight = payload['weight']
    cut_vars = payload['cut_vars']
    #cut = payload['cut']
    if args.background:
        cut = payload['background_cut']
    else:
        cut = payload['signal_cut']
    substructure_vars = payload['substructure_vars']
    default = payload['default']
    per_part = bool(payload['per_part'])
    if VERBOSITY != 0: print "per_part: ", per_part
    nparticles = payload['nparticles']

    for sample in payload['samples']:
        if sample['name'] == args.name:
            samples = sample['samples']
    
    in_cut_vars_only = list(set(cut_vars) - set(features))


# reading from the root files
try:
    filenames = [basedir+'/'+sample+'.root' for sample in samples]
except NameError:
    print "NameError, it's likely that --name was given a bad arg; make sure the name passed is paired with a .root file in the .json file"
files = [uproot.open(f) for f in filenames]
trees = [f['Events'] for f in files]
keys = trees[0].keys()
nevents = 0
#nevents_after_cut = 0
#uncomment the line below to see what keys will be accepted for features/defualts in the json
if VERBOSITY == 3:
    print '\n Keys: \n', keys, type(keys)

#if no features are provided, want to grab features w same number of entries as the default
if features == []:
    try:
        default
    except ValueError:
        print 'no default provided'
    as_dict = trees[0].arrays(keys)
    default_shape = as_dict[default].shape
    #print '\n\ndefault_shape: ', default_shape
    for k in keys:
        shape = as_dict[k].shape
        #print shape
        if shape == default_shape:
            #print 'appending^'
            features.append(k)

# takes a dict produced by tree.arrays and reformats it
# the input dict has var names for keys, and 2d (evt and particle) arrays for values
# the output dict has keys like 'fj_cpf_pt[0]' and 1d arrays (evt) for values
def reformat_dict(d):
    ks = []
    vs = []

    for k, v in d.iteritems():
        if VERBOSITY == 2: print "in reformat_dict, k, v.shape: ", k, v.shape
        if v.ndim == 1:
            ks.append(k)
            vs.append(v)
        elif v.ndim == 2:
            for i in range(v.shape[1]):
                if i == nparticles: break
                ks.append(k+"[{}]".format(i))
                vs.append([v[j][i] for j in range(v.shape[0])])
        else:
            raise ValueError("Too many dimensions")

    new_dict = dict(zip(ks, vs))
    return new_dict

good_indicies = [] #stores result* of applying cuts to x so we can drop the same indicies from the other dataframes; *list of remaining indicies after cut

def get_branches_as_df(branches, mode):
    dicts = [tree.arrays(branches=branches) for tree in trees] 
    if per_part:
        dicts = [reformat_dict(d) for d in dicts]
    dfs = [pd.DataFrame.from_dict(d, dtype=np.float16) for d in dicts]
  
    df = pd.concat(dfs)

    #df = pd.DataFrame(dicts)

    #print mode, '\n', df.head()
    if mode=='features' and cut: 
        #the first call to this function must have mode=features for everything to work
        if VERBOSITY != 0: print "cut:", cut
        global nevents
        nevents = df.shape[0]
        if VERBOSITY == 2: print "df.shape before cut: ", df.shape
        df = df[eval(cut)].dropna()
        if VERBOSITY == 2: print "df.shape after cut: ", df.shape
        df = df.reset_index()
        global good_indicies 
        good_indicies = list(df['index'])
        df = df.drop('index', axis=1)
    return df

# unpacking the branches into x, y, weight and substructure vars       
X = get_branches_as_df(features + in_cut_vars_only, 'features')
if VERBOSITY == 2:
    print "X before: ", X.shape, '\n', X.head()
    print list(X.columns)

def is_extra(s):
    m = re.search(r"\d+", s)
    if m:
        return int(m.group(0)) >= nparticles
    return True
'''
if per_part:
    extra_columns = list(filter(is_extra, list(X.columns)))
    weird_columns = list(filter(lambda s: "fj_pt" in s, list(X.columns)))
    if VERBOSITY == 2:
        print "\nextra: ", extra_columns
        print "weird: ", weird_columns
    X = X.drop(in_cut_vars_only + weird_columns + extra_columns, axis=1)
else:
    X = X.drop(in_cut_vars_only, axis=1)
'''
X = X.drop(in_cut_vars_only, axis=1)

if VERBOSITY == 2:
    print '\nX.columns:\n', list(X.columns)
    print "dtypes: ", X.dtypes
W = get_branches_as_df([weight], 'weight')
y = 0 if args.background else 1
print "nevents: ", nevents
Y = pd.DataFrame(data=y*np.ones(shape=(nevents, 1)))
substructure = get_branches_as_df(substructure_vars, 'substructure')
decayType = get_branches_as_df(['fj_decayType'], 'decayType')

def calc_ptweights(feat_train,Y_train):
    nbins=20
    ptbins = np.linspace(400.,1000.,num=nbins+1)
    sighist = np.zeros(nbins,dtype='f8')
    bkghist = np.zeros(nbins,dtype='f8')
    ptweights = np.ones(nevents_after_cut,dtype='f8')
    print "len of feat_train, ptbins: ", len(feat_train), len(ptbins)
    for x in range(nevents_after_cut):
        pti = 1
        while (pti<nbins):
            if (feat_train[x]>ptbins[pti-1] and feat_train[x]<ptbins[pti]): break
            pti = pti+1
        if (pti<nbins):
            if (Y_train[x]==1):
                sighist[pti] = sighist[pti]+1.
            else:
                bkghist[pti] = bkghist[pti]+1.
    #sighist = sighist/sum(sighist)
    #bkghist = bkghist/sum(bkghist)
    for x in range(nevents_after_cut):
        if (not Y_train[x]==0): continue
        pti = 1
        while (pti<nbins):
            if (feat_train[x]>ptbins[pti-1] and feat_train[x]<ptbins[pti]): break
            pti = pti+1
        if (pti<nbins and bkghist[pti]>0.): ptweights[x] = sighist[pti]/bkghist[pti]
    return ptweights

# writing to .pkl files
def save(df, label):
    fout = args.out+'/'+args.name+'_'+label+'.pkl'
    if label != 'x':
        df = df.iloc[good_indicies]
    df = df.reset_index(drop=True)
    df.to_pickle(fout)
    if VERBOSITY == 1 or VERBOSITY == 2:
        print '\n'+label+'\n', df.shape, '\n'
        print df.head()
        print "n nan: ", df.isnull().sum().sum()

if not args.dry:
    save(X, 'x')
    save(Y, 'y')
    #save(W, 'w')
    save(W, 'j_pt')
    save(substructure, 'ss_vars')
    save(decayType, 'decayType')

