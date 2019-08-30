import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import argparse
import json
import os

#grabbing command line args
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, help='filename of output pdf (the .pdf extension is optional)')
parser.add_argument('--json', type=str, help='json file controlling the input files used')
parser.add_argument('--trim', type=float, nargs='?', action='store', help='slice this proportion of items from both ends of the Data')
parser.add_argument('--dpi', type=int, nargs='?', action='store', help='dpi for output pdf')
args = parser.parse_args()

#separating the 'out' argument into the file path and the file name
split = args.out.split('/')
out_dir = '/'+'/'.join(split[:-1])
out_name = split[-1]

#adding the .pdf in case it's not there
if out_name.split('.')[-1] != 'pdf':
    out_name = out_name + '.pdf'

#making the output directory in case it doesn't already exist
parent_dir = os.getcwd()
try:
    os.makedirs(parent_dir+out_dir)
except:
    pass

out = PdfPages(out_name)  #(out_dir + '/' + out_name)

dpi = 100
if args.dpi:
    dpi = args.dpi

#parsing the json file
with open(args.json) as jsonfile:
    payload = json.load(jsonfile)
    base_dir = payload['base_dir']
    filenames = payload['filenames']
    displaynames = payload['displaynames']
    per_part = bool(payload['per_part'])
    print "particle level: ", per_part
    combine_particles = bool(payload['combine_particles'])
    if per_part:
        cut_vars = payload['cut_vars']
        cuts = payload['particle_cuts']
    else:
        jet_cut = payload['jet_cut']

#reading data from the .pkl files and applying selections
dfs = {}

for k, v in filenames.iteritems():
    #print "dfs[k] = pd.read_pickle(base_dir+v+.pkl)", k, v
    dfs[k] = pd.read_pickle(base_dir+v+"_x.pkl")

var_names = list(dfs[list(dfs)[0]])

def apply_particle_cuts():
    if cuts and cut_vars:
        if len(cuts) != len(cut_vars): raise ValueError("Different number of cuts and cut_vars")
        #going from 'fj_cpf_pfType' to ['fj_cpf_pfType[0]', ...]
        all_cut_vars = [[spec_var for spec_var in var_names if gen_var in spec_var] for gen_var in cut_vars]
        #all_cut_vars = [var+"[0]" for var in cut_vars]
        #print "all_cut_vars: ", all_cut_vars
        for k, df in dfs.iteritems():
            #print k, "\ndf.shape before cuts: ", df.shape
            for i in range(len(cuts)):
                for var in all_cut_vars[i]:
                    #print cuts[i].format(var)
                    df = df[eval(cuts[i].format(var))]
            #print "df.shape after cuts: ", df.shape
            dfs[k] = df.reset_index(drop=True)

def combine_particle_columns():
    nparticles = len([var for var in var_names if var_names[0][:-3] in var])
    gen_var_names = [var_names[i][:-3] for i in range(0, len(var_names), nparticles)]
    #print gen_var_names
    for k, df in dfs.iteritems():
        #print "\ncombine_particle_columns: df.head before changes\n", df.head(), df.shape, '\n'
        condensed_data = []
        for var in gen_var_names:
            columns = [v for v in var_names if var in v]
            #print "combine_particle_columns: current var, columns", var, columns
            combined = pd.concat([df[col] for col in columns], axis=0)
            condensed_data.append(combined)
        #print "condensed_data[0]: ", type(condensed_data[0]), condensed_data[0].head()
        #print len(gen_var_names), len(condensed_data)
        #print gen_var_names
        condensed_data = [s.reset_index(drop=True) for s in condensed_data]
        df = pd.concat(condensed_data, axis=1, keys=gen_var_names)
        #df = pd.DataFrame(data=dict(zip(gen_var_names, condensed_data)))
        #print "combine_particle_columns: df.head after changes\n", df.head(), df.shape, '\n'
        dfs[k] = df

if per_part:
    apply_particle_cuts()
    if combine_particles:
        combine_particle_columns()
else:
    if jet_cut:
        for k, df in dfs.iteritems():
            pass#dfs[k] = df[eval(cut)]

var_names = list(dfs[list(dfs)[0]])

# functions to make individual plots

def trim(data, p):
    #print "data.shape before trim: ", data.shape
    data = pd.Series(data=stats.trimboth(data, p))
    #print "data.shape after trim: ", data.shape
    return data

def make_hist(var):
    #Plots a histogram comparing var across all dataframes
    plt.figure()#(figsize=(4, 4), dpi=dpi)
    plt.xlabel(var)
    plt.title(var)
    
    for k, v in dfs.iteritems():
        data = v[var]

        # special rules
        #if 'jet_charge' in var:
        #    data = data.apply(lambda x: abs(x))

        if args.trim:
            data = trim(data, args.trim)
        min_ = data.min()
        max_ = data.max()
        bins = np.linspace(min_, max_, 100)
        data.plot.hist(bins, label=displaynames[k], histtype='step', density=True)

    plt.legend(loc='upper right')
    
    PdfPages.savefig(out, dpi=dpi)
    return

def make_plot(xvar, yvar):
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    plt.title(xvar+" vs. "+yvar)

    for df in dfs.itervalues():
        if args.maxz:
            df = df[np.abs(stats.zscore(df[xvar])) < max_zscore]

        x = df[xvar]
        y = df[yvar]

        kwargs = {'ls': 'None'}
        
        plt.plot(x, y, 'bo', **kwargs)

    plt.legend(loc='upper right')

    PdfPages.savefig(out, dpi=dpi)
    return

# functions to make plots for a given list of vars
def make_hists(vars_to_plot):
    problems = {}
    for var in vars_to_plot:
        try:
            make_hist(var)
        except Exception as e:
            problems[var] = str(e)
    return problems

# all vars
#print "var_names: ", var_names
#make_hists(var_names)

# jet charge
#make_hist('jet_charge_k0_l0')
j_c_vars = [v for v in var_names if 'jet_charge' in v]
make_hists(j_c_vars)

# kinematics
#kinematics = ['fj_cpf_pt', 'fj_cpf_eta', 'fj_cpf_phi', 'fj_cpf_dz', 'fj_cpf_pup', 'fj_cpf_q']
#make_hists(kinematics)

# misc
#make_hist("N2_times_pt_weight")
#make_hist("pt_weight")

#make_hists([v for v in var_names if "cpf" in v])
#make_hist("fj_pt")

out.close()
