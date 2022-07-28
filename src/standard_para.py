import tensorflow as tf 
import mfr.modelSoln as m
import json
import os

def setModelParameters(parameter_list, domain_list):

  chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h = parameter_list 
  nV, nVtilde, V_bar, Vtilde_bar, sigma_V_norm, sigma_Vtilde_norm = domain_list 

  params = m.paramsDefault.copy()

  ## Dimensionality params
  params['nDims']             = 3
  params['nShocks']           = 3

  ## Grid parameters 
  params['numSds']            = 5
  params['uselogW']           = 0

  params['nWealth']           = 100
  params['nZ']                = 30
  params['nV']                = nV
  params['nVtilde']           = nVtilde


  ## Economic params
  params['nu_newborn']        = 0.1
  params['lambda_d']          = 0.02
  params['lambda_Z']          = 0.252
  params['lambda_V']          = 0.156
  params['lambda_Vtilde']     = 1.38
  params['delta_e']           = 0.05
  params['delta_h']           = 0.05
  params['a_e']               = a_e
  params['a_h']               = a_h
  params['rho_e']             = psi_e
  params['rho_h']             = psi_h
  params['phi']               = 3.0
  params['gamma_e']           = gamma_e
  params['gamma_h']           = gamma_h
  params['equityIss']         = 2
  params['chiUnderline']      = chiUnderline
  params['alpha_K']           = 0.05

  ## Alogirthm behavior and results savings params
  params['method']            = 2
  params['dt']                = 0.1
  params['dtInner']           = 0.1

  params['tol']               = 1e-5
  params['innerTol']          = 1e-5

  params['verbatim']          = -1
  params['maxIters']          = 4000
  params['maxItersInner']     = 2000000
  params['iparm_2']           = 28
  params['iparm_3']           = 0
  params['iparm_28']          = 0
  params['iparm_31']          = 0
  params['overwrite']         = 'Yes'
  params['exportFreq']        = 10000
  params['CGscale']           = 1.0
  params['hhCap']             = 1
  params['preLoad']           = 'None'

  ## Domain params
  params['Vtilde_bar']        = Vtilde_bar
  params['Z_bar']             = 0.0
  params['V_bar']             = V_bar
  params['sigma_K_norm']      = 0.04
  params['sigma_Z_norm']      = 0.0141
  params['sigma_V_norm']      = sigma_V_norm
  params['sigma_Vtilde_norm'] = sigma_Vtilde_norm
  params['wMin']              = 0.01
  params['wMax']              = 0.99

  ## Shock correlation params
  params['cov11']             = 1.0
  params['cov12']             = 0.0
  params['cov13']             = 0.0
  params['cov14']             = 0.0
  params['cov21']             = 0.0
  params['cov22']             = 1.0
  params['cov23']             = 0.0
  params['cov24']             = 0.0
  params['cov31']             = 0.0
  params['cov32']             = 0.0
  params['cov33']             = 1.0
  params['cov34']             = 0.0
  params['cov41']             = 0.0
  params['cov42']             = 0.0
  params['cov43']             = 0.0
  params['cov44']             = 0.0

  if params['sigma_Vtilde_norm'] == 0:
    domain_folder = 'WZV'
  elif params['sigma_V_norm'] == 0:
    domain_folder = 'WZVtilde'

  chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h = [str("{:0.3f}".format(param)).replace('.', '', 1)  for param in parameter_list]
  model_folder = 'chiUnderline_' + chiUnderline + '_a_e_' + a_e + '_a_h_' + a_h  + '_gamma_e_' + gamma_e + '_gamma_h_' + gamma_h + '_psi_e_' + psi_e + '_psi_h_' + psi_h

  params['folderName']        = model_folder

  workdir = os.path.dirname(os.getcwd())
  datadir = workdir + '/data/' + domain_folder + '/' + model_folder + '/'
  with open(datadir + '/parameters_NN.json', 'w') as f:
      json.dump(params,f)

def setModelParametersFromFile(paramsFromFile):

  params = {}
  ####### Model parameters #######
  params['nu_newborn']             = tf.constant(paramsFromFile['nu_newborn'],         dtype=tf.float64);
  params['lambda_d']               = tf.constant(paramsFromFile['lambda_d'],           dtype=tf.float64);
  params['lambda_Z']               = tf.constant(paramsFromFile['lambda_Z'],           dtype=tf.float64);
  params['lambda_V']               = tf.constant(paramsFromFile['lambda_V'],           dtype=tf.float64);
  params['lambda_Vtilde']          = tf.constant(paramsFromFile['lambda_Vtilde'],      dtype=tf.float64);
  params['Vtilde_bar']             = tf.constant(paramsFromFile['Vtilde_bar'],         dtype=tf.float64);
  params['Z_bar']                  = tf.constant(paramsFromFile['Z_bar'],              dtype=tf.float64);
  params['V_bar']                  = tf.constant(paramsFromFile['V_bar'],              dtype=tf.float64);
  params['a_e']                    = tf.constant(paramsFromFile['a_e'],                dtype=tf.float64);
  params['a_h']                    = tf.constant(paramsFromFile['a_h'],                dtype=tf.float64);  ###Any negative number means -infty
  params['phi']                    = tf.constant(paramsFromFile['phi'],                dtype=tf.float64);
  params['gamma_e']                = tf.constant(paramsFromFile['gamma_e'],            dtype=tf.float64);
  params['gamma_h']                = tf.constant(paramsFromFile['gamma_h'],            dtype=tf.float64);
  params['psi_e']                  = tf.constant(paramsFromFile['rho_e'],              dtype=tf.float64);  ### Mismatch btw paper and MFR notation
  params['psi_h']                  = tf.constant(paramsFromFile['rho_h'],              dtype=tf.float64);  ### Mismatch btw paper and MFR notation
  params['rho_e']                  = tf.constant(paramsFromFile['delta_e'],            dtype=tf.float64); 
  params['rho_h']                  = tf.constant(paramsFromFile['delta_h'],            dtype=tf.float64);  
  params['sigma_K_norm']           = tf.constant(paramsFromFile['sigma_K_norm'],       dtype=tf.float64);
  params['sigma_Z_norm']           = tf.constant(paramsFromFile['sigma_Z_norm'],       dtype=tf.float64);
  params['sigma_V_norm']           = tf.constant(paramsFromFile['sigma_V_norm'],       dtype=tf.float64);
  params['sigma_Vtilde_norm']      = tf.constant(paramsFromFile['sigma_Vtilde_norm'],  dtype=tf.float64);
  params['equityIss']              = tf.constant(paramsFromFile['equityIss'],          dtype=tf.float64);
  params['chiUnderline']           = tf.constant(paramsFromFile['chiUnderline'],       dtype=tf.float64);
  params['delta']                  = tf.constant(paramsFromFile['alpha_K'],            dtype=tf.float64);


  params['cov11']                  = tf.constant(paramsFromFile['cov11'],              dtype=tf.float64);
  params['cov12']                  = tf.constant(paramsFromFile['cov12'],              dtype=tf.float64);
  params['cov13']                  = tf.constant(paramsFromFile['cov13'],              dtype=tf.float64);
  params['cov14']                  = tf.constant(paramsFromFile['cov14'],              dtype=tf.float64);

  params['cov21']                  = tf.constant(paramsFromFile['cov21'],              dtype=tf.float64);
  params['cov22']                  = tf.constant(paramsFromFile['cov22'],              dtype=tf.float64);
  params['cov23']                  = tf.constant(paramsFromFile['cov23'],              dtype=tf.float64);
  params['cov24']                  = tf.constant(paramsFromFile['cov24'],              dtype=tf.float64);

  params['cov31']                  = tf.constant(paramsFromFile['cov31'],              dtype=tf.float64);
  params['cov32']                  = tf.constant(paramsFromFile['cov32'],              dtype=tf.float64);
  params['cov33']                  = tf.constant(paramsFromFile['cov33'],              dtype=tf.float64);
  params['cov34']                  = tf.constant(paramsFromFile['cov34'],              dtype=tf.float64);

  params['cov41']                  = tf.constant(paramsFromFile['cov41'],              dtype=tf.float64);
  params['cov42']                  = tf.constant(paramsFromFile['cov42'],              dtype=tf.float64);
  params['cov43']                  = tf.constant(paramsFromFile['cov43'],              dtype=tf.float64);
  params['cov44']                  = tf.constant(paramsFromFile['cov44'],              dtype=tf.float64);

  params['numSds']                 = tf.constant(paramsFromFile['numSds'],             dtype=tf.float64);

  ########### Derived parameters
  ## Covariance matrices 
  params['sigmaK']                 = tf.concat([params['cov11'] * params['sigma_K_norm'], 
                                                params['cov12'] * params['sigma_K_norm'],
                                                params['cov13'] * params['sigma_K_norm'],
                                                params['cov14'] * params['sigma_K_norm']], 0)

  params['sigmaZ']                 = tf.concat([params['cov21'] * params['sigma_Z_norm'], 
                                                params['cov22'] * params['sigma_Z_norm'],
                                                params['cov23'] * params['sigma_Z_norm'],
                                                params['cov24'] * params['sigma_Z_norm']], 0)

  params['sigmaV']                 = tf.concat([params['cov31'] * params['sigma_V_norm'], 
                                                params['cov32'] * params['sigma_V_norm'],
                                                params['cov33'] * params['sigma_V_norm'],
                                                params['cov34'] * params['sigma_V_norm']], 0)
  
  params['sigmaVtilde']            = tf.concat([params['cov31'] * params['sigma_Vtilde_norm'], 
                                                params['cov32'] * params['sigma_Vtilde_norm'],
                                                params['cov33'] * params['sigma_Vtilde_norm'],
                                                params['cov34'] * params['sigma_Vtilde_norm']], 0)
  
  ## Min and max of state variables
  ## min/max for W
  params['wMin'] = tf.constant(0.01, dtype=tf.float64)
  params['wMax'] = tf.constant(1 - params['wMin'], dtype=tf.float64)
  
  ## min/max for Z
  zVar  = tf.pow(params['V_bar'] * params['sigma_Z_norm'], 2) / (2 * params['lambda_Z'])
  params['zMin'] = params['Z_bar'] - params['numSds'] * tf.sqrt(zVar)
  params['zMax'] = params['Z_bar'] + params['numSds'] * tf.sqrt(zVar)

  ## min/max for V
  if params['sigma_V_norm'] == 0:
    params['vMin'] = params['V_bar']
    params['vMax'] = params['V_bar']
  else:
    shape = 2 * params['lambda_V'] * params['V_bar']  /  (tf.pow(params['sigma_V_norm'],2));
    rate = 2 * params['lambda_V'] / (tf.pow(params['sigma_V_norm'],2));
    params['vMin'] = tf.constant(0.00001, dtype=tf.float64)
    params['vMax'] = params['V_bar'] + params['numSds'] * tf.sqrt( shape / tf.pow(rate, 2));
  
  ## min/max for Vtilde
  if params['sigma_Vtilde_norm'] == 0:
    params['VtildeMin'] = params['Vtilde_bar']
    params['VtildeMax'] = params['Vtilde_bar']
  else:
    vtildeVar  = tf.pow(params['Vtilde_bar'] * params['sigma_Vtilde_norm'], 2) / (2 * params['lambda_Vtilde'])
    params['VtildeMin'] = tf.constant(0.00001, dtype=tf.float64)
    params['VtildeMax'] = params['Vtilde_bar'] + params['numSds'] * tf.sqrt(vtildeVar)

  return params