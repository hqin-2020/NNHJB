import numpy as np
import tensorflow as tf 
import time 
from scipy import optimize

@tf.function 
def calc_var(valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params):

    X = tf.concat([W,Z,V,Vtilde], axis=1)

    ## Parse information
    nShocks      = params['sigmaK'].shape[0]
    nStates      = X.shape[1]
    batchSize    = X.shape[0]

    ## Value functions and kappa
    logXiE       = valueFunctionLogE(X);    logXiH       = valueFunctionLogH(X)
    xiE          = tf.exp(logXiE);          xiH          = tf.exp(logXiH)
    if params['a_h'] > 0:
      kappa        = tf.reshape(constraintsFunctionKappa(X), [batchSize, 1] )
    else:
      kappa        = tf.ones([batchSize,1], dtype=tf.float64)

    ## Compute Q
    num_q          = (1 - kappa) * params['a_h'] + kappa * params['a_e'] + 1 / params['phi']
    den_q          = (1 - W) * tf.pow(params['rho_h'], 1 / params['psi_h']) \
                    * tf.pow(xiH, 1 - 1 / params['psi_h']) + W * tf.pow(params['rho_e'], 1 / params['psi_e']) \
                    * tf.pow(xiE, 1 - 1 / params['psi_e']) + 1 / params['phi']
    Q              = num_q / den_q                                                                                                          ##### eq. (44)
    logQ           = tf.math.log(Q)

    ### Derivatives
    dW_logQ     = tf.gradients(logQ, W)[0];         dZ_logQ     = tf.gradients(logQ, Z)[0];         dV_logQ     = tf.gradients(logQ, V)[0];       dVtilde_logQ    = tf.gradients(logQ, Vtilde)[0]
    dW2_logQ    = tf.gradients(dW_logQ, W)[0];      dZ2_logQ    = tf.gradients(dZ_logQ, Z)[0];      dV2_logQ    = tf.gradients(dV_logQ, V)[0];    dVtilde2_logQ   = tf.gradients(dVtilde_logQ, Vtilde)[0]
    
    dW_Q        = tf.gradients(Q, W)[0];            dZ_Q        = tf.gradients(Q, Z)[0];            dV_Q        = tf.gradients(Q, V)[0];          dVtilde_Q       = tf.gradients(Q, Vtilde)[0]
    dW2_Q       = tf.gradients(dW_Q, W)[0];         dZ2_Q       = tf.gradients(dZ_Q, Z)[0];         dV2_Q       = tf.gradients(dV_Q, V)[0];       dVtilde2_Q      = tf.gradients(dVtilde_Q, Vtilde)[0]
    dWdZ_Q      = tf.gradients(dW_Q, Z)[0];         dWdV_Q      = tf.gradients(dW_Q, V)[0];         dZdV_Q      = tf.gradients(dZ_Q, V)[0]

    dW_logXiE   = tf.gradients(logXiE, W)[0];       dZ_logXiE   = tf.gradients(logXiE, Z)[0];       dV_logXiE   = tf.gradients(logXiE, V)[0];     dVtilde_logXiE  = tf.gradients(logXiE, Vtilde)[0]
    dW2_logXiE  = tf.gradients(dW_logXiE, W)[0];    dZ2_logXiE  = tf.gradients(dZ_logXiE, Z)[0];    dV2_logXiE  = tf.gradients(dV_logXiE, V)[0];  dVtilde2_logXiE = tf.gradients(dVtilde_logXiE, Vtilde)[0]
    dWdZ_logXiE = tf.gradients(dW_logXiE, Z)[0];    dWdV_logXiE = tf.gradients(dW_logXiE, V)[0];    dZdV_logXiE = tf.gradients(dZ_logXiE, V)[0]

    dW_logXiH   = tf.gradients(logXiH, W)[0];       dZ_logXiH   = tf.gradients(logXiH, Z)[0];       dV_logXiH   = tf.gradients(logXiH, V)[0];     dVtilde_logXiH  = tf.gradients(logXiH, Vtilde)[0]
    dW2_logXiH  = tf.gradients(dW_logXiH, W)[0];    dZ2_logXiH  = tf.gradients(dZ_logXiH, Z)[0];    dV2_logXiH  = tf.gradients(dV_logXiH, V)[0];  dVtilde2_logXiH = tf.gradients(dVtilde_logXiH, Vtilde)[0]
    dWdZ_logXiH = tf.gradients(dW_logXiH, Z)[0];    dWdV_logXiH = tf.gradients(dW_logXiH, V)[0];    dZdV_logXiH = tf.gradients(dZ_logXiH, V)[0]

    dX_logQ     = tf.concat([dW_logQ, dZ_logQ, dV_logQ, dVtilde_logQ], axis=1)
    dX2_logQ    = tf.concat([dW2_logQ, dZ2_logQ, dV2_logQ, dVtilde2_logQ], axis=1)
    dX_Q        = tf.concat([dW_Q, dZ_Q, dV_Q, dVtilde_Q], axis=1)
    dX2_Q_diag  = tf.concat([dW2_Q, dZ2_Q, dV2_Q, dVtilde2_Q], axis=1)
    dX_logXiE   = tf.concat([dW_logXiE, dZ_logXiE, dV_logXiE, dVtilde_logXiE], axis=1)
    dX2_logXiE  = tf.concat([dW2_logXiE, dZ2_logXiE, dV2_logXiE, dVtilde2_logXiE], axis=1)
    dX_logXiH   = tf.concat([dW_logXiH, dZ_logXiH, dV_logXiH, dVtilde_logXiH], axis=1)
    dX2_logXiH  = tf.concat([dW2_logXiH, dZ2_logXiH, dV2_logXiH, dVtilde2_logXiH], axis=1)

    ## Compute drifts and volatilities. For now, assume no idio vol (\varsigma = 0 everywhere)
    sigmaK      = params['sigmaK'] * tf.sqrt(V)
    sigmaZ      = params['sigmaZ'] * tf.sqrt(V)
    sigmaV      = params['sigmaV'] * tf.sqrt(V)
    sigmaVtilde = params['sigmaVtilde'] * tf.sqrt(Vtilde)

    muK         = Z + logQ / params['phi'] - params['delta']  - 0.5*tf.reduce_sum(sigmaK*sigmaK, axis=1, keepdims=True)                         ###### eq. (46)
    muZ         = params['lambda_Z'] * (params['Z_bar'] - Z)                                                                                    ###### eq. (53)
    muV         = params['lambda_V'] * (params['V_bar'] - V)                                                                                    ###### eq. (53)
    muVtilde    = params['lambda_Vtilde'] * (params['Vtilde_bar'] - Vtilde)                                                                     ###### eq. (53)

    ## Compute chi
    sigmaXtilde         = [sigmaZ, sigmaV, sigmaVtilde]                                                                                         ###### eq. (69)
    Dx                  = sigmaK + (sigmaZ*dZ_logQ + sigmaV*dV_logQ + sigmaVtilde*dVtilde_logQ)                                                 ###### eq. (70)
    DxNormSq            = tf.reduce_sum(Dx * Dx, axis = 1, keepdims=True)                                                                
    DzetaOmega          = W*(1-W)*DxNormSq * ( (params['gamma_h'] - 1.0) * dW_logXiH - (params['gamma_e'] - 1.0) * dW_logXiE )                  ###### eq. (71)
    DzetaX              = tf.zeros(DzetaOmega.shape, dtype=tf.float64)
    for s in range(nShocks):
      for n in range(1,nStates):
        DzetaX          = DzetaX + Dx[:,s:s+1] * ( sigmaXtilde[n-1][:,s:s+1] * ( ( params['gamma_h'] - 1.0) * dX_logXiH[:,n:n+1] - 
                                                (params['gamma_e'] - 1.0) * dX_logXiE[:,n:n+1] ) )
    DzetaX              = DzetaX * W* (1 - W)                                                                                                   ###### eq. (72)
    chiN                = DzetaX - W* (1 - W) * (params['gamma_e'] - params['gamma_h']) * DxNormSq
    chiD                = ( ( 1 - W) * params['gamma_e'] + W * params['gamma_h'] ) * DxNormSq + dW_logQ * DzetaX - DzetaOmega
    chi                 = chiN / chiD + W                                                                                                 
    chi                 = tf.math.maximum(chi, params['chiUnderline'])                                                                          ###### eq. (68)
    
    ## Compute deltaE and deltaH
    sigmaQ              = ( (chi * kappa - W) * sigmaK * dW_logQ + sigmaZ * dZ_logQ + sigmaV * dV_logQ + sigmaVtilde * dVtilde_logQ)/(1.0 -  (chi * kappa - W ) * dW_logQ)    ###### eq. (57)
    sigmaR              = sigmaK  + sigmaQ                                                                                                      ###### eq. (58) simplified
    sigmaW              = (chi * kappa - W) * sigmaR                                                                                            ###### eq. (52)
    sigmaRNormSq        = tf.reduce_sum(sigmaR * sigmaR, axis = 1, keepdims=True) 
    sigmaRsigmaXDerivs  = tf.zeros(sigmaRNormSq.shape, dtype=tf.float64)
    for s in range(nShocks):
      sigmaRsigmaXDerivs = sigmaRsigmaXDerivs \
                        + sigmaR[:,s:s+1] * (((params['gamma_h'] - 1) * dX_logXiH[:,0:1] - (params['gamma_e'] - 1) * dX_logXiE[:,0:1] ) * sigmaW[:,s:s+1] \
                                            + ((params['gamma_h'] - 1) * dX_logXiH[:,1:2] - (params['gamma_e'] - 1) * dX_logXiE[:,1:2] ) * sigmaZ[:,s:s+1] \
                                            + ((params['gamma_h'] - 1) * dX_logXiH[:,2:3] - (params['gamma_e'] - 1) * dX_logXiE[:,2:3] ) * sigmaV[:,s:s+1] 
                                            + ((params['gamma_h'] - 1) * dX_logXiH[:,3:4] - (params['gamma_e'] - 1) * dX_logXiE[:,3:4] ) * sigmaVtilde[:,s:s+1] )
                                                                                                                                                ###### last term in eq. (63)
    deltaE              = params['gamma_e'] * chi * kappa / W * (sigmaRNormSq + Vtilde) - \
                          params['gamma_h'] * (1 - chi * kappa) / (1 - W) * sigmaRNormSq - sigmaRsigmaXDerivs                                   ###### eq. (63)
    deltaH              = params['chiUnderline'] * deltaE - (params['a_e'] - params['a_h']) / tf.exp(logQ)                                      ###### eq. (64)
    
    ## Compute r and Pi
    Pi      = params['gamma_h'] * ( (1.0 - chi * kappa) / (1.0 - W)  ) * sigmaR + \
              (params['gamma_h'] - 1.0) * (sigmaW * dW_logXiH + sigmaZ * dZ_logXiH + sigmaV * dV_logXiH + sigmaVtilde * dVtilde_logXiH)         ###### eq. (62)
    Pi_e    =  (params['gamma_e'] * chi * kappa / W ) * sigmaR + \
                (params['gamma_e'] - 1.0) * (sigmaW * dW_logXiE + sigmaZ * dZ_logXiE + sigmaV * dV_logXiE + sigmaVtilde * dVtilde_logXiE)
    betaE   = chi * kappa / W
    betaH   = (1 - kappa) / (1 - W)    
    muW     = (W * (1.0 - W)) * ( tf.pow(params['rho_h'], 1.0 / params['psi_h'] ) * tf.pow(xiH, 1 - 1.0 / params['psi_h'] ) 
                                   - tf.pow(params['rho_e'], 1.0 / params['psi_e'] ) * tf.pow(xiE, 1 - 1.0 / params['psi_e'] ) + betaE * deltaE - betaH * deltaH )  \
                                   + tf.reduce_sum(sigmaR * (Pi - sigmaR),axis=1, keepdims=True) * (chi * kappa - W) + params['lambda_d'] * (params['nu_newborn'] - W) 
                                                                                                                                                ###### eq. (51)
    muX     = tf.concat([muW, muZ, muV, muVtilde], axis=1)                                                                                      ###### eq. (53)
    sigmaX  = [sigmaW, sigmaZ, sigmaV, sigmaVtilde]                                                                                             ###### eq. (54)
    muQ     = 1 / Q * tf.reduce_sum(muX*dX_Q, axis=1, keepdims=True) + \
                1 / (2*Q) * ( tf.reduce_sum(sigmaW*sigmaW, axis=1, keepdims=True)*dW2_Q   +
                              tf.reduce_sum(sigmaZ*sigmaZ, axis=1, keepdims=True)*dZ2_Q   + 
                              tf.reduce_sum(sigmaV*sigmaV, axis=1, keepdims=True)*dV2_Q   +
                              tf.reduce_sum(sigmaVtilde*sigmaVtilde, axis=1, keepdims=True)*dVtilde2_Q) #+
                                # 2*tf.reduce_sum(sigmaW*sigmaV, axis=1, keepdims=True)*dWdV_Q + 
                                # 2*tf.reduce_sum(sigmaW*sigmaZ, axis=1, keepdims=True)*dWdZ_Q +  
                                # 2*tf.reduce_sum(sigmaZ*sigmaV, axis=1, keepdims=True)*dZdV_Q   )                                              ###### eq. (56)
    r       = muQ + muK + tf.reduce_sum(sigmaK * sigmaQ,axis=1, keepdims=True) - tf.reduce_sum(sigmaR * Pi,axis=1, keepdims=True) \
              - (1 - W ) * (betaH * deltaH - tf.pow(params['rho_h'], 1 / params['psi_h']) * tf.pow(xiH, 1- 1 / params['psi_h'] ) ) \
                - W * (betaE * deltaE - tf.pow(params['rho_e'], 1 / params['psi_e']) * tf.pow(xiE, 1- 1 / params['psi_e'] ) )                   ###### eq. (61)
    
    I       = logQ / params['phi']                                                                                                              ###### Def. of iota
    muRe    = (params['a_e'] - 1.0 / params['phi'] * (tf.exp(params['phi'] * I ) - 1) ) / Q + I - params['delta'] + \
              Z + muQ + tf.reduce_sum(sigmaQ * sigmaK, axis=1, keepdims=True) ###### Not used anywhere

    muRh    = (params['a_h'] - 1.0 / params['phi'] * (tf.exp(params['phi'] * I ) - 1) ) / Q + I - params['delta'] + \
              Z + muQ + tf.reduce_sum(sigmaQ * sigmaK, axis=1, keepdims=True) ###### Not used anywhere

    variables = {'logXiE'    : logXiE,      'logXiH'    : logXiH,\
                 'xiE'       : xiE,         'xiH'       : xiH,\
                 'dX_logXiE' : dX_logXiE,   'dX_logXiH' : dX_logXiH,\
                 'dX2_logXiE': dX2_logXiE,  'dX2_logXiH': dX2_logXiH,\
                 'kappa'     : kappa,       'chi'       : chi,\
                 'sigmaW'    : sigmaW,      'sigmaZ'    : sigmaZ,\
                 'sigmaV'    : sigmaV,      'sigmaVtilde' : sigmaVtilde,\
                 'Pi'        : Pi,          'Pi_e'      : Pi_e,\
                 'r'         : r,           'Q'         : Q,\
                 'betaE'     : betaE,       'betaH'     : betaH,\
                 'sigmaR'    : sigmaR,      
                 'sigmaRNormSq' : sigmaRNormSq, 'sigmaRsigmaXDerivs' : sigmaRsigmaXDerivs,\
                 'deltaE'    : deltaE,      'deltaH'    : deltaH,\
                 'muX'       : muX,         'sigmaX'    : sigmaX}

    return variables

@tf.function 
def HJB_loss_E(valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params):

    X = tf.concat([W,Z,V,Vtilde], axis=1)
    ## Parse information
    nShocks      = params['sigmaK'].shape[0]
    nStates      = X.shape[1]
    batchSize    = X.shape[0]
    
    variables = calc_var(valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params)
    xiE = variables['xiE']
    logXiE = variables['logXiE']
    dX_logXiE = variables['dX_logXiE']
    dX2_logXiE = variables['dX2_logXiE']
    deltaE = variables['deltaE']
    muX = variables['muX']
    Pi = variables['Pi']
    r = variables['r']
    sigmaW = variables['sigmaW']
    sigmaZ = variables['sigmaZ']
    sigmaV = variables['sigmaV']
    sigmaVtilde = variables['sigmaVtilde']
    sigmaX = variables['sigmaX']
    sigmaR = variables['sigmaR']
    sigmaRNormSq = variables['sigmaRNormSq']

    #### Constant term and xiE
    Fe             = tf.zeros([batchSize,1], dtype=tf.float64)

    if params['psi_e'] == 1:
      Fe           = Fe + (-logXiE + tf.math.log(params['rho_e'])) * params['rho_e'] - params['rho_e']
    else:
      Fe           = Fe + params['psi_e'] / (1 - params['psi_e']) * tf.pow(params['rho_e'], 1 / params['psi_e'] ) \
      * tf.pow(xiE, 1 - 1 / params['psi_e']) - params['rho_e'] / (1 - params['psi_e'])

    Fe             = Fe + r + tf.square(deltaE + tf.reduce_sum(sigmaR * Pi,axis=1, keepdims=True)) / (2 * params['gamma_e'] * (sigmaRNormSq + Vtilde)) ###### eq. (39)

    for s in range(nShocks):
      for s_sub in range(nShocks):
        Fe = Fe + ( sigmaX[0][:,s:s+1] * dX_logXiE[:,0:1] + sigmaX[1][:,s:s+1] * dX_logXiE[:,1:2] + sigmaX[2][:,s:s+1] * dX_logXiE[:,2:3] ) \
            * ( sigmaR[:,s:s+1] * sigmaR[:,s_sub:s_sub+1] * (1.0 - params['gamma_e']) / ( sigmaRNormSq  + Vtilde) + (params['gamma_e']) * (s == s_sub)) \
            * ( sigmaW[:,s_sub:s_sub+1] * dX_logXiE[:,0:1] + sigmaZ[:,s_sub:s_sub+1] * dX_logXiE[:,1:2] + sigmaV[:,s_sub:s_sub+1] * dX_logXiE[:,2:3] + sigmaVtilde[:,s_sub:s_sub+1] * dX_logXiE[:,3:4]) * \
            (1.0 - params['gamma_e']) / params['gamma_e'] * 0.5        ###### eq. (39)

    #### First and second partials
    firstPartialsE   = tf.zeros([batchSize, 1], dtype=tf.float64)
    secondPartialsE  = tf.zeros([batchSize, 1], dtype=tf.float64)

    for n in range(nStates): ###### eq. (39)
      firstPartialsE    = firstPartialsE + (muX[:,n:n+1]+ (1 - params['gamma_e'] ) / params['gamma_e'] * \
      tf.reduce_sum(sigmaX[n] * sigmaR, axis=1, keepdims=True) * (deltaE + tf.reduce_sum(Pi * sigmaR,axis=1, keepdims=True)) / (sigmaRNormSq + Vtilde) ) * dX_logXiE[:,n:n+1]

      secondPartialsE   = secondPartialsE + 0.5 * tf.reduce_sum(sigmaX[n] * sigmaX[n], axis=1, keepdims=True) * dX2_logXiE[:,n:n+1]

    HJB_E  = Fe + firstPartialsE + secondPartialsE

    return HJB_E

@tf.function 
def HJB_loss_H(valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params):

    X = tf.concat([W,Z,V,Vtilde], axis=1)
    ## Parse information
    nShocks      = params['sigmaK'].shape[0]
    nStates      = X.shape[1]
    batchSize    = X.shape[0]

    variables = calc_var(valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params)
    xiH = variables['xiH']
    logXiH = variables['logXiH']
    dX_logXiH = variables['dX_logXiH']
    dX2_logXiH = variables['dX2_logXiH']
    muX = variables['muX']
    Pi = variables['Pi']
    r = variables['r']
    betaH = variables['betaH']
    sigmaW = variables['sigmaW']
    sigmaZ = variables['sigmaZ']
    sigmaV = variables['sigmaV']
    sigmaVtilde = variables['sigmaVtilde']
    sigmaX = variables['sigmaX']

    ### Constant term and xiH
    Fh             = tf.zeros([batchSize,1], dtype=tf.float64)

    if params['psi_h'] == 1:
      Fh           = Fh + (-logXiH + tf.math.log(params['rho_h'])) * params['rho_h'] - params['rho_h']
    else:
      Fh           = Fh + params['psi_h'] / (1 - params['psi_h']) * tf.pow(params['rho_h'], 1 / params['psi_h'] ) \
      * tf.pow(xiH, 1 - 1 / params['psi_h']) - params['rho_h'] / (1 - params['psi_h'])

    Fh             = Fh + r + (tf.reduce_sum(Pi*Pi, axis=1, keepdims=True) + tf.square(params['gamma_h'] * betaH * tf.sqrt(Vtilde)) )/ (2 * params['gamma_h']) ###### eq. (38)

    for s in range(nShocks): ###### eq. (38)
        Fh           = Fh + 0.5 * (1.0 - params['gamma_h']) / params['gamma_h'] * \
        tf.square(sigmaW[:,s:s+1] * dX_logXiH[:,0:1] + sigmaZ[:,s:s+1] * dX_logXiH[:,1:2] + sigmaV[:,s:s+1] * dX_logXiH[:,2:3] + sigmaVtilde[:,s:s+1] * dX_logXiH[:,3:4])

    #### First and second partials
    firstPartialsH   = tf.zeros([batchSize, 1], dtype=tf.float64)
    secondPartialsH  = tf.zeros([batchSize, 1], dtype=tf.float64)
    for n in range(nStates):
      firstPartialsH  = firstPartialsH + (muX[:,n:n+1] + (1 - params['gamma_h']) / params['gamma_h'] \
                                      * tf.reduce_sum(sigmaX[n] * Pi, axis=1, keepdims=True) ) * dX_logXiH[:,n:n+1]
      secondPartialsH = secondPartialsH + 0.5 * tf.reduce_sum(sigmaX[n] * sigmaX[n],axis=1, keepdims=True) * dX2_logXiH[:,n:n+1]


    HJB_H  = Fh + firstPartialsH + secondPartialsH

    return HJB_H 

@tf.function 
def loss_kappa(valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params):

    X = tf.concat([W,Z,V,Vtilde], axis=1)
    ## Parse information
    batchSize   = X.shape[0]
    W           = X[:batchSize,0:1]

    variables = calc_var(valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params)
    Q = variables['Q']
    kappa = variables['kappa']
    sigmaRNormSq = variables['sigmaRNormSq']
    sigmaRsigmaXDerivs = variables['sigmaRsigmaXDerivs']

    rightTerm          = W * params['gamma_h'] * (1 - params['chiUnderline'] * kappa) * sigmaRNormSq + W * params['gamma_h'] * ((1 - kappa) / params['chiUnderline']) * Vtilde - (1 - W) \
                          * params['gamma_e'] * params['chiUnderline'] * kappa * ( sigmaRNormSq + Vtilde) + W * (1 - W) * \
                            (params['a_e'] - params['a_h']) / (params['chiUnderline'] * Q) + W * (1 - W) * sigmaRsigmaXDerivs  ###### eq. (66)

    kappa_min          = tf.math.minimum( 1 - kappa, rightTerm ) ###### eq. (66)
    return kappa_min

def function_factory(model, loss, valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params, loss_type, targets, weight = 10, boundary = 0.00001):

    ## Obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n
    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # Create a function that will compute the value and gradient. This can be the function that the factory returns
    @tf.function
    def val_and_grad(params_1d):
        with tf.GradientTape() as tape:
          ## Update the parameters in the model
            assign_new_model_parameters(params_1d)
            ## Calculate the loss 
            if model == constraintsFunctionKappa:
              loss_value = loss_type(loss(valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params)+\
                                    weight*loss(valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, 10**(-boundary)*tf.ones(W.shape, dtype=tf.float64), Z, V, params), targets)
            else:
              loss_value = loss_type(loss(valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params), targets)
        ## Calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)
        del tape

        ## Print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        ## Store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    def f(params_1d):
      return [vv.numpy().astype(np.float64)  for vv in val_and_grad(params_1d)]

    ## Store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f
  

## Training step BFGS
def training_step_BFGS(valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params, targets, weight = 0, boundary = 0.00001):

  ## Train kappa
  loss_fun = tf.keras.losses.MeanSquaredError()
  func_K = function_factory(constraintsFunctionKappa, loss_kappa, valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params, loss_fun, targets, weight, boundary)
  init_params_K = tf.dynamic_stitch(func_K.idx, constraintsFunctionKappa.trainable_variables)

  start = time.time()
  results = optimize.minimize(func_K, x0 = init_params_K.numpy(), 
                              method = 'L-BFGS-B', 
                              jac    = True,
                              options = {'maxiter': 500,
                                          'maxfun': 50000,
                                          'gtol': 1.0 * np.finfo(float).eps,
                                          'maxcor': 100,
                                          'maxls': 100,
                                          'ftol' : 1.0 * np.finfo(float).eps})
                                          # 'ftol' : 1.0 * np.finfo(float).eps})
  end = time.time()
  print('Elapsed time for kappa {:.4f} sec'.format(end - start))
  # after training, the final optimized parameters are still in results.position
  # so we have to manually put them back to the model
  func_K.assign_new_model_parameters(results.x)

  ## Train experts NN
  loss_fun = tf.keras.losses.MeanSquaredError()
  func_E = function_factory(valueFunctionLogE, HJB_loss_E, valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params, loss_fun, targets)
  init_params_E = tf.dynamic_stitch(func_E.idx, valueFunctionLogE.trainable_variables)

  start = time.time()
  results = optimize.minimize(func_E, x0 = init_params_E.numpy(), 
                              method = 'L-BFGS-B', 
                              jac    = True,
                              options = {'maxiter': 500,
                                          'maxfun': 50000,
                                          'gtol': 1.0 * 1.0 * np.finfo(float).eps,
                                          'maxcor': 100,
                                          'maxls': 100,
                                          'ftol' : 1.0 * 1.0 * np.finfo(float).eps})
                                          # 'ftol' : 1.0 * np.finfo(float).eps})
  end = time.time()
  print('Elapsed time for experts {:.4f} sec'.format(end - start))
  # after training, the final optimized parameters are still in results.position
  # so we have to manually put them back to the model
  func_E.assign_new_model_parameters(results.x)

  ## Train households NN
  loss_fun = tf.keras.losses.MeanSquaredError()
  func_H = function_factory(valueFunctionLogH, HJB_loss_H, valueFunctionLogH, valueFunctionLogE, constraintsFunctionKappa, W, Z, V, Vtilde, params, loss_fun, targets)
  init_params_H = tf.dynamic_stitch(func_H.idx, valueFunctionLogH.trainable_variables)

  start = time.time()
  results = optimize.minimize(func_H, x0 = init_params_H.numpy(), 
                              method = 'L-BFGS-B', 
                              jac    = True,
                              options = {'maxiter': 500,
                                          'maxfun': 50000,
                                          'gtol': 1.0 * 1.0 * np.finfo(float).eps,
                                          'maxcor': 100,
                                          'maxls': 100,
                                          'ftol' : 1.0 * 1.0 * np.finfo(float).eps})
                                          # 'ftol' : 1.0 * np.finfo(float).eps})
  end = time.time()
  print('Elapsed time for households {:.4f} sec'.format(end - start))
  # after training, the final optimized parameters are still in results.position
  # so we have to manually put them back to the model
  func_H.assign_new_model_parameters(results.x)  