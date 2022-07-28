import argparse
import os 
srcdir = os.getcwd()

parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--logXiE_NN_layers",type=int,default=4)
parser.add_argument("--logXiH_NN_layers",type=int,default=4)
parser.add_argument("--kappa_NN_layers",type=int,default=4)
args = parser.parse_args()

with open(srcdir + '/BFGS.py', 'r') as f:
    lines = f.readlines()

logXiE_NN_layers = args.logXiE_NN_layers
logXiH_NN_layers = args.logXiH_NN_layers
kappa_NN_layers = args.kappa_NN_layers
layer = "      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),\n"
i = lines.index('      ####### logXiE_NN structure #######\n') 
lines[i:i+1] = [layer for i in range(logXiE_NN_layers)]
i = lines.index('      ####### logXiH_NN structure #######\n') 
lines[i:i+1] = [layer for i in range(logXiH_NN_layers)]
i = lines.index('      ####### kappa_NN structure #######\n') 
lines[i:i+1] = [layer for i in range(kappa_NN_layers)]

with open(srcdir + '/standard_BFGS.py', 'w') as f:
    f.writelines(lines)