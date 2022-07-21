import tensorflow as tf 

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
  params['delta']                  = tf.constant(paramsFromFile['alpha_K'],              dtype=tf.float64);


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
  
  ## Min and max of state variables
  ## min/max for V
  shape = 2 * params['lambda_V'] * params['V_bar']  /   (tf.pow(params['sigma_V_norm'],2));
  rate = 2 * params['lambda_V'] / (tf.pow(params['sigma_V_norm'],2));
  params['vMin'] = tf.constant(0.00001, dtype=tf.float64)

  
  params['vMax'] = params['V_bar'] + params['numSds'] * tf.sqrt( shape / tf.pow(rate, 2));

  ## min/max for Z
  zVar  = tf.pow(params['V_bar'] * params['sigma_Z_norm'], 2) / (2 * params['lambda_Z'])
  params['zMin'] = params['Z_bar'] - params['numSds'] * tf.sqrt(zVar)
  params['zMax'] = params['Z_bar'] + params['numSds'] * tf.sqrt(zVar)

  ## min/max for W
  params['wMin'] = tf.constant(0.01, dtype=tf.float64)
  params['wMax'] = tf.constant(1 - params['wMin'], dtype=tf.float64)

  return params