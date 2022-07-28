import json
import tensorflow as tf 
import pandas as pd
import time 
import os
import pickle
from standard_para import *
from standard_training import *
import argparse

parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--nV",type=int,default=30)
parser.add_argument("--nVtilde",type=int,default=0)
parser.add_argument("--V_bar",type=float,default=1.0)
parser.add_argument("--Vtilde_bar",type=float,default=0.0)
parser.add_argument("--sigma_V_norm",type=float,default=0.132)
parser.add_argument("--sigma_Vtilde_norm",type=float,default=0.0)

parser.add_argument("--a_e",type=float,default=0.14)
parser.add_argument("--a_h",type=float,default=0.135)
parser.add_argument("--psi_e",type=float,default=1.0)
parser.add_argument("--psi_h",type=float,default=1.0)
parser.add_argument("--gamma_e",type=float,default=1.0)
parser.add_argument("--gamma_h",type=float,default=1.0)
parser.add_argument("--chiUnderline",type=float,default=1.0)

parser.add_argument("--logXiE_NN_layers",type=int,default=4)
parser.add_argument("--logXiH_NN_layers",type=int,default=4)
parser.add_argument("--kappa_NN_layers",type=int,default=4)
parser.add_argument("--weight",type=float,default=10.0)
parser.add_argument("--boundary",type=int,default=5)
parser.add_argument("--points_size",type=int,default=2)
args = parser.parse_args()

nV                = args.nV
nVtilde           = args.nVtilde
V_bar             = args.V_bar
Vtilde_bar        = args.Vtilde_bar
sigma_V_norm      = args.sigma_V_norm
sigma_Vtilde_norm = args.sigma_Vtilde_norm
domain_list       = [nV, nVtilde, V_bar, Vtilde_bar, sigma_V_norm, sigma_Vtilde_norm]
if sigma_Vtilde_norm == 0:
  domain_folder = 'WZV'
elif sigma_V_norm == 0:
  domain_folder = 'WZVtilde'

a_e               = args.a_e
a_h               = args.a_h
psi_e             = args.psi_e
psi_h             = args.psi_h
gamma_e           = args.gamma_e
gamma_h           = args.gamma_h
chiUnderline      = args.chiUnderline
parameter_list    = [chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h]
chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h = [str("{:0.3f}".format(param)).replace('.', '', 1)  for param in parameter_list]
model_folder = 'chiUnderline_' + chiUnderline + '_a_e_' + a_e + '_a_h_' + a_h  + '_gamma_e_' + gamma_e + '_gamma_h_' + gamma_h + '_psi_e_' + psi_e + '_psi_h_' + psi_h

logXiE_NN_layers = args.logXiE_NN_layers
logXiH_NN_layers = args.logXiH_NN_layers
kappa_NN_layers = args.kappa_NN_layers
weight = args.weight
boundary = args.boundary
points_size = args.points_size
layer_folder = 'logXiE_NN_layers_' + str(logXiE_NN_layers) +'_logXiH_NN_layers_' + str(logXiH_NN_layers) +'_kappa_NN_layers_'+ str(kappa_NN_layers) + '_weight_' + str(int(weight)) + '_points_size_' + str(points_size) + '_boundary_' + str(boundary)

workdir = os.path.dirname(os.getcwd())
srcdir = workdir + '/src/'
datadir = workdir + '/data/' + domain_folder + '/' + model_folder + '/'
outputdir = workdir + '/output/' + domain_folder + '/' + model_folder + '/' + layer_folder + '/'
docdir = workdir + '/doc/' + domain_folder + '/' + model_folder + '/'+ layer_folder + '/'
os.makedirs(datadir,exist_ok=True)
os.makedirs(docdir,exist_ok=True)
os.makedirs(outputdir,exist_ok=True)

setModelParameters(parameter_list, domain_list)
with open(datadir + 'parameters_NN.json') as json_file:
    paramsFromFile = json.load(json_file)
params = setModelParametersFromFile(paramsFromFile)

batchSize = 2048 * points_size
dimension = 4
units = 16
iter_num = 10
activation = 'tanh'
kernel_initializer = 'glorot_normal'

NN_info = {'logXiE_NN_layers': logXiE_NN_layers, 'logXiH_NN_layers': logXiH_NN_layers, 'kappa_NN_layers': kappa_NN_layers, 'weight': weight, 'boundary': boundary,\
          'points_size': points_size, 'dimension': dimension, 'units': units, 'activation': activation, 'kernel_initializer': kernel_initializer, 'iter_num': iter_num}
with open(outputdir + "/NN_info.json", "w") as f:
  json.dump(NN_info,f)

## Use float64 by default
tf.keras.backend.set_floatx("float64")

logXiE_NN = tf.keras.Sequential(
    [tf.keras.Input(shape=[dimension,]),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(1,  activation= None,  kernel_initializer='glorot_normal')])

logXiH_NN = tf.keras.Sequential(
    [tf.keras.Input(shape=[dimension,]),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(1,  activation= None , kernel_initializer='glorot_normal')])

kappa_NN = tf.keras.Sequential(
    [tf.keras.Input(shape=[dimension,]),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(1,  activation='sigmoid', kernel_initializer='glorot_normal')])

start = time.time()
targets = tf.zeros(shape=(batchSize,1), dtype=tf.float64)
for iter in range(iter_num):
  W = tf.random.uniform(shape = (batchSize,1), minval = params['wMin'], maxval = params['wMax'], dtype=tf.float64)
  Z = tf.random.uniform(shape = (batchSize,1), minval = params['zMin'], maxval = params['zMax'], dtype=tf.float64)
  V = tf.random.uniform(shape = (batchSize,1), minval = params['vMin'], maxval = params['vMax'], dtype=tf.float64)
  Vtilde = tf.random.uniform(shape = (batchSize,1), minval = params['VtildeMin'], maxval = params['VtildeMax'], dtype=tf.float64)
  print('Iteration', iter)
  training_step_BFGS(logXiH_NN, logXiE_NN, kappa_NN, W, Z, V, Vtilde, params, targets)
end = time.time()
print('Elapsed time for training {:.4f} sec'.format(end - start))

## Save trained neural network approximations and respective model parameters
tf.saved_model.save(logXiH_NN, outputdir   + 'logXiH_NN')
tf.saved_model.save(logXiE_NN, outputdir   + 'logXiE_NN')
tf.saved_model.save(kappa_NN,  outputdir   + 'kappa_NN')

with open(outputdir + 'params.pickle', 'wb') as handle:
  pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)



