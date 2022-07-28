import json
import tensorflow as tf 
import time 
import os
import pickle
from standard_para import *
from standard_training import *
import argparse

parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--a_e",type=float,default=0.14)
parser.add_argument("--a_h",type=float,default=0.135)
parser.add_argument("--psi_e",type=float,default=1.0)
parser.add_argument("--psi_h",type=float,default=1.0)
parser.add_argument("--gamma_e",type=float,default=1.0)
parser.add_argument("--gamma_h",type=float,default=1.0)
parser.add_argument("--chiUnderline",type=float,default=1.0)
args = parser.parse_args()

a_e             = args.a_e
a_h             = args.a_h
psi_e           = args.psi_e
psi_h           = args.psi_h
gamma_e         = args.gamma_e
gamma_h         = args.gamma_h
chiUnderline    = args.chiUnderline
parameter_list = [chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h]
folder_name = ''
psi_e = str("{:0.3f}".format(psi_e)).replace('.', '', 1) 
psi_h = str("{:0.3f}".format(psi_h)).replace('.', '', 1) 
gamma_e = str("{:0.3f}".format(gamma_e)).replace('.', '', 1) 
gamma_h = str("{:0.3f}".format(gamma_h)).replace('.', '', 1) 
a_e = str("{:0.3f}".format(a_e)).replace('.', '', 1) 
a_h = str("{:0.3f}".format(a_h)).replace('.', '', 1) 
chiUnderline = str("{:0.3f}".format(chiUnderline)).replace('.', '', 1) 
folder_name = folder_name + 'chiUnderline_' + chiUnderline + '_a_e_' + a_e + '_a_h_' + a_h  + '_gamma_e_' + gamma_e + '_gamma_h_' + gamma_h + '_psi_e_' + psi_e + '_psi_h_' + psi_h

workdir = os.path.dirname(os.getcwd())
srcdir = workdir + '/src/'
datadir = workdir + '/data/' + folder_name + '/'
outputdir = workdir + '/output/' + folder_name + '/'
docdir = workdir + '/doc/' + folder_name + '/'

try:
  os.mkdir(docdir)
except:
  pass
try:
  os.mkdir(outputdir)
except:
  pass

setModelParameters(parameter_list)
json_location =  datadir + 'parameters_NN.json'
with open(json_location) as json_file:
    paramsFromFile= json.load(json_file)
params = setModelParametersFromFile(paramsFromFile)

points_size = 5
batchSize = 2048*points_size
dimension = 3
units = 16
activation = 'tanh'
kernel_initializer = 'glorot_normal'

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
for iter in range(10):
  W = tf.random.uniform(shape = (batchSize,1), minval = params['wMin'], maxval = params['wMax'], dtype=tf.float64)
  Z = tf.random.uniform(shape = (batchSize,1), minval = params['zMin'], maxval = params['zMax'], dtype=tf.float64)
  V = tf.random.uniform(shape = (batchSize,1), minval = params['vMin'], maxval = params['vMax'], dtype=tf.float64)
  
  print('Iteration', iter)
  training_step_BFGS(logXiH_NN, logXiE_NN, kappa_NN, W, Z, V, params, targets)
end = time.time()
print('Elapsed time for training {:.4f} sec'.format(end - start))

## Save trained neural network approximations and respective model parameters
save_results = True
if save_results:

  VF_h_Name = 'logXiH_NN'
  VF_e_Name = 'logXiE_NN'
  policy_Name = 'kappa_NN'

  tf.saved_model.save(logXiH_NN, outputdir   + VF_h_Name)
  tf.saved_model.save(logXiE_NN, outputdir   + VF_e_Name)
  tf.saved_model.save(kappa_NN, outputdir + policy_Name)

  with open(outputdir + 'params.pickle', 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

