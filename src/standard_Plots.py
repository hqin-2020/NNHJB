import json
import os
import numpy as np
import tensorflow as tf
from standard_para import *
from standard_training import *
from standard_plotting import *
import argparse
from stationaryDensityModules import *

parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--model",type=str,default='ah_0135')
args = parser.parse_args()

workdir = os.path.dirname(os.getcwd())
model = args.model
srcdir = workdir + '/src/'
datadir = workdir + '/data/' + model + '/'
outputdir = workdir + '/output/' + model + '/'
docdir = workdir + '/doc/' + model + '/'

json_location =  datadir + 'parameters.json'

with open(json_location) as json_file:
  paramsFromFile= json.load(json_file)
params = setModelParametersFromFile(paramsFromFile)

mfrSuite_resdir = outputdir + 'mfrSuite_res'

load_saved_results = True
if load_saved_results:
  ## Load trained NNs back from memory
  logXiH_NN = tf.saved_model.load(outputdir   + 'logXiH_NN')
  logXiE_NN = tf.saved_model.load(outputdir   + 'logXiE_NN')
  kappa_NN = tf.saved_model.load(outputdir    + 'kappa_NN')

### Load all MRF values
## Load MRF approximations and state variables
logXiE                   = np.genfromtxt(mfrSuite_resdir + '/xi_e_final.dat').transpose()
logXiH                   = np.genfromtxt(mfrSuite_resdir + '/xi_h_final.dat').transpose()
kappa                    = np.genfromtxt(mfrSuite_resdir + '/kappa_final.dat').transpose()
W                        = np.genfromtxt(mfrSuite_resdir + '/W.dat')
Z                        = np.genfromtxt(mfrSuite_resdir + '/Z.dat')
V                        = np.genfromtxt(mfrSuite_resdir + '/V.dat')
X                        = np.array([W,Z,V]).transpose()

q_MFR = np.genfromtxt(mfrSuite_resdir + '/q_final.dat').transpose().reshape(-1,1)
r_MFR = np.genfromtxt(mfrSuite_resdir + '/r_final.dat').transpose().reshape(-1,1)
Pi_h_MFR = np.genfromtxt(mfrSuite_resdir + '/PiH_final.dat').reshape(3,-1).transpose()
Pi_e_MFR = np.genfromtxt(mfrSuite_resdir + '/PiE_final.dat').reshape(3,-1).transpose()
sigmaR_MFR = np.genfromtxt(mfrSuite_resdir + '/sigmaR_final.dat').reshape(3,-1).transpose()
dent_MFR   = np.genfromtxt(mfrSuite_resdir + '/dent.txt').reshape(-1,1)

# Form X and order_states dictionary
X_var = tf.Variable(X, dtype=tf.float64)

logXiE_NNs = logXiE_NN(X_var).numpy().squeeze(axis=1)
logXiH_NNs = logXiH_NN(X_var).numpy().squeeze(axis=1)
kappa_NNs  = kappa_NN(X_var).numpy().squeeze(axis=1)

variables = calc_var(logXiH_NN,logXiE_NN, kappa_NN, W.reshape(-1,1),Z.reshape(-1,1),V.reshape(-1,1), params)
q_NN = variables['Q']
r_NN = variables['r']
Pi_h_NN = variables['Pi']
Pi_e_NN = variables['Pi_e']
sigmaR_NN = variables['sigmaR']
muX_NN = variables['muX']
sigmaX_NN = variables['sigmaX']
q_NN, r_NN, Pi_h_NN, Pi_e_NN, sigmaR_NN, muX_NN, sigmaX_NN = q_NN.numpy(), r_NN.numpy(), Pi_h_NN.numpy(), Pi_e_NN.numpy(), sigmaR_NN.numpy(), np.matrix(muX_NN), [np.matrix(el) for el in sigmaX_NN]
dent_NN, FKmat_NN, stateGrid = computeDent(np.matrix(X), {'muX': muX_NN, 'sigmaX': sigmaX_NN})

mfr_Results   = [logXiE, logXiH, kappa]
nn_Results    = [logXiE_NNs, logXiH_NNs, kappa_NNs]  
fix_points    = [14, 14]
function_name = ['Experts value function', 'Households value function', 'Kappa policy function']
var_name      = ['xi_e', 'xi_h', 'kappa']
plot_content  = 'Value Function, Policy Function: ' + model
generateSurfacePlots(mfr_Results, nn_Results, fix_points, X, function_name, var_name, plot_content, height=800, width=1300, path = docdir)
# generateScatterPlots(mfr_Results, nn_Results, function_name, height=20, width=7)

mfr_Results   = [q_MFR, r_MFR]
nn_Results    = [q_NN, r_NN]  
fix_points    = [14, 14]
function_name = ['Capital Price', 'Short Term Interest Rate']
var_name      = ['q', 'r']
plot_content  = 'Capital Price, Short Term Interest Rate: ' + model
generateSurfacePlots(mfr_Results, nn_Results, fix_points, X, function_name, var_name, plot_content, height=800, width=1000, path = docdir)

mfr_Results   = [Pi_h_MFR[:,i] for i in range(3)]
nn_Results    = [Pi_h_NN[:,i] for i in range(3)]
fix_points    = [14, 14]
function_name = ['First Shock', 'Second Shock', 'Third Shock']
var_name      = ['Pi_h' for i in range(3)]
plot_content  = 'Households Risk Price: ' + model
generateSurfacePlots(mfr_Results, nn_Results, fix_points, X, function_name, var_name, plot_content, height=800, width=1300, path = docdir)

mfr_Results   = [Pi_e_MFR[:,i] for i in range(3)]
nn_Results    = [Pi_e_NN[:,i] for i in range(3)]
fix_points    = [14, 14]
function_name = ['First Shock', 'Second Shock', 'Third Shock']
var_name      = ['Pi_e' for i in range(3)]
plot_content  = 'Experts Risk Price: ' + model
generateSurfacePlots(mfr_Results, nn_Results, fix_points, X, function_name, var_name, plot_content, height=800, width=1300, path = docdir)

mfr_Results   = [sigmaR_MFR[:,i] for i in range(3)]
nn_Results    = [sigmaR_NN[:,i] for i in range(3)]
fix_points    = [14, 14]
function_name = ['First Shock', 'Second Shock', 'Third Shock']
var_name      = ['sigma_R' for i in range(3)]
plot_content  = 'Local Capital Return Volatility: ' + model
generateSurfacePlots(mfr_Results, nn_Results, fix_points, X, function_name, var_name, plot_content, height=800, width=1300, path = docdir)

mfr_Results   = [dent_MFR]
nn_Results    = [dent_NN]
fix_points    = [14,14]
function_name = ['Stationary density']
var_name      = ['Density']
plot_content  = 'Stationary densities: ' + model
generateSurfacePlots(mfr_Results, nn_Results, fix_points, X, function_name, var_name, plot_content, height=800, width=1300, path = docdir)