import numpy as np
import pandas as pd
import argparse
import json
import ROOT
import sys

# https://stackoverflow.com/questions/3160699/python-progress-bar
progress_bar_width = 40
# setup toolbar
#sys.stdout.write("[%s]" % (" " * progress_bar_width))
#sys.stdout.flush()
#sys.stdout.write("\b" * (progress_bar_width+1)) # return to start of line, after '['

parser = argparse.ArgumentParser()
#might want to add an --out arg in case we want to leave the original .pkl
#this script should be able to use the same .json files as plot.py
parser.add_argument('--json', type=str, help='json file pointing to the .pkl files to be edited')
args = parser.parse_args()

with open(args.json) as jsonfile:
    payload = json.load(jsonfile)
    base_dir = payload['base_dir']
    filenames = payload['filenames']

dfs = {}
weights = {}
ss_vars = {}

for k, v in filenames.iteritems():
    dfs[k] = pd.read_pickle(base_dir+v+'_x.pkl')
    weights[k] = pd.read_pickle(base_dir+v+'_w.pkl')
    ss_vars[k] = pd.read_pickle(base_dir+v+'_ss_vars.pkl')

#dPhi_metjet
def make_dPhi_metjet():
    for k, v in dfs.iteritems():
        v['dPhi_metjet'] = v['pfmetphi'] - v['fj_phi']
    return

#jetplusmet_mass
def calc_jpm_mass(j_pt, j_eta, j_phi, j_mass, met, met_phi):
    jet = ROOT.TLorentzVector(j_pt, j_eta, j_phi, j_mass)
    met = ROOT.TLorentzVector(met, j_eta, met_phi, 0)
    jpm = jet+met
    return jpm.M()

def make_jetplusmet_mass(j_pt='fj_pt', j_eta='fj_eta', j_phi='fj_phi', j_mass='fj_mass', met='pfmet', met_phi='pfmetphi'):
    for k, v in dfs.iteritems():
        v['jetplusmet_mass'] = np.vectorize(calc_jpm_mass)(v[j_pt], v[j_eta], v[j_phi], v[j_mass], v[met], v[met_phi])
        print 'fj_mass:\n', v[j_mass][:5]
        print 'jetplusmet_mass:\n', v['jetplusmet_mass'][:5]

#jetplusmet_pt
def calc_jpm_pt(j_pt, j_eta, j_phi, j_mass, met, met_phi):
    jet = ROOT.TLorentzVector(j_pt, j_eta, j_phi, j_mass)
    met = ROOT.TLorentzVector(met, j_eta, met_phi, 0)
    jpm = jet+met
    return jpm.Pt()

def make_jetplusmet_pt(j_pt='fj_pt', j_eta='fj_eta', j_phi='fj_phi', j_mass='fj_mass', met='pfmet', met_phi='pfmetphi'):
    for k, v in dfs.iteritems():
        v['jetplusmet_pt'] = np.vectorize(calc_jpm_pt)(v[j_pt], v[j_eta], v[j_phi], v[j_mass], v[met], v[met_phi])

def make_jpm_vars():
    make_jetplusmet_mass()
    make_jetplusmet_pt()

#jet_charge
def calc_jet_charge(kappa, lambda_, p_qs, p_pts): #p_qs = [$fj_cpf_q[0], ...]
    total = 0
    #print "p_qs:", p_qs
    if len(p_qs) != len(p_pts):
        raise ValueError("calc_jet_charge: mismatched lists")

    for i in range(len(p_qs)):
        total += p_qs[i] * (p_pts[i] ** kappa)

    return total

def make_jet_charge(kappa=0, lambda_=0, p_q='fj_cpf_q', p_pt='fj_cpf_pt'):
    col_name = 'jet_charge_k{}_l{}'.format(kappa, lambda_)
    print "making: ", col_name
    for k, v in dfs.iteritems():
        p_qvars = [col for col in v.columns if p_q in col]
        p_ptvars = [col for col in v.columns if p_pt in col]

        qs = v[p_qvars].values
        pts = v[p_ptvars].values

        one_percent = v.shape[0] / 100
        percent_per_tick = 100 / progress_bar_width
        milestone = one_percent * percent_per_tick
        
        for i in range(v.shape[0]):
            v.at[i, col_name] = calc_jet_charge(kappa, lambda_, qs[i], pts[i])
            if i % milestone == 0:
                sys.stdout.write("-")
                sys.stdout.flush()
        print "\n"
        '''
        p_qs = v[p_qvars]
        p_pts = v[p_ptvars]
        
        
        for i in range(v.shape[0]):
            v.at[i, col_name] = calc_jet_charge(kappa, lambda_, p_qs.iloc[i], p_pts.iloc[i])
            
        '''
#sys.stdout.write("]\n") # this ends the progress bar

# N2*weight
def make_N2_weighted():
    for k, v in dfs.iteritems():
        w = weights[k]
        N2 = ss_vars[k]
        #print w.head(), "\n", N2.head()
        v['N2_times_pt_weight'] = np.multiply(w, N2)
        print k, '\n', v.head()

# weight
def make_pt_weight():
    for k, v in dfs.iteritems():
        v['pt_weight'] = weights[k]

def save():
    for k, v in dfs.iteritems():
        print k, '\n', v.head()
        v.to_pickle(base_dir+filenames[k]+'_x.pkl')
    return

#make_dPhi_metjet()
#make_jpm_vars()
#make_jet_charge(kappa=0.5, lambda_=0)

kappas = [0, 0.1, 0.2, 0.5, 0.7, 1]
for k in kappas:
    make_jet_charge(kappa=k, lambda_=0)

#make_N2_weighted()
#make_pt_weight()
    
save()





