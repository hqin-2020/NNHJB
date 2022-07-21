import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os

def generateSurfacePlots(mfr_Results, nn_Results, fixed_points, X, function_name, var_name, plot_content = 'Value Function, Policy Function' ,float_formatter = "{0:.4f}", height=800, width=1200, path = os.path.dirname(os.getcwd()) + '/doc/'):

    W, Z, V            = X[:,0], X[:,1], X[:,2]
    plot_results       = [mfr_Results, nn_Results]
    plot_results_name  = ['MFR', "NN"]
    plot_color_style   = ['Viridis', 'Plasma']
    fixed_var          = [Z, V]
    fixed_inv_var      = [V, Z]
    fixed_var_name     = ['Z','V']
    fixed_inv_var_name = ['V','Z']
    n_points           = np.unique(W).shape[0]
    plot_row_dims      = len(fixed_points)
    plot_col_dims      = len(mfr_Results)

    fixed_idx = [fixed_var[i] == np.unique(fixed_var[i])[fixed_points[i]]         for i in range(plot_row_dims)]
    fixed_val = [float_formatter.format(np.unique(fixed_var[i])[fixed_points[i]]) for i in range(plot_row_dims)]
    
    fixed_subplot_titles = []
    subplot_types = []
    for row in range(plot_row_dims):
      subplot_type = []
      for col in range(plot_col_dims):
        fixed_twoNorm = float_formatter.format(np.linalg.norm(nn_Results[col][fixed_idx[row]] - mfr_Results[col][fixed_idx[row]]))
        fixed_subplot_titles.append(function_name[col] + '. '+ fixed_var_name[row] +' fixed at '+ str(fixed_val[row]) +'. <br> ||diff.||_2 = ' + str(fixed_twoNorm))
        subplot_type.append({'type': 'surface'})
      subplot_types.append(subplot_type)

    fig = make_subplots(
        rows=plot_row_dims, cols=plot_col_dims, horizontal_spacing=.1, vertical_spacing=.1,
        subplot_titles=(fixed_subplot_titles), specs=subplot_types)
    
    for row in range(plot_row_dims):
      for col in range(plot_col_dims):
        showlegend = True if ((col == 0) & (row == 0)) else False
        fig.update_scenes(dict(xaxis_title='W', yaxis_title=fixed_inv_var_name[row], zaxis_title=var_name[col]), row = row+1, col = col+1)
        for i in range(len(plot_results)):
          fig.add_trace(go.Surface(
            x=W[fixed_idx[row]].reshape([n_points, 30], order='F'),
            y=fixed_inv_var[row][fixed_idx[row]].reshape([n_points, 30], order='F'),
            z=plot_results[i][col][fixed_idx[row]].reshape([n_points, 30], order='F'),
            colorscale=plot_color_style[i], showscale=False, name= plot_results_name[i], showlegend=showlegend), row = row+1, col = col+1)

    
    fig.update_layout(title= 'MRF vs NN - '+ plot_content +' surface plots', title_x = 0.5, title_y = 0.98, height=height, width=width)
    fig.write_html(path + "/" + plot_content + ".html")
    fig.show()

def generateScatterPlots(mfr_Results, nn_Results, function_name, height=20, width=7):
  plot_row_dims     = len(mfr_Results)
  n_points          = 90000
  fig, axes = plt.subplots(plot_row_dims,1, figsize = (height,width))
  for i, ax in enumerate(axes.flatten()):
    ax.scatter(np.arange(0,n_points), mfr_Results[i], label = 'MFR', alpha = 0.3, s=0.1)
    ax.scatter(np.arange(0,n_points), nn_Results[i], label = 'NN', alpha = 0.3, s=0.1)
    ax.set_title(function_name[i])
    ax.legend()
  fig.tight_layout()
  plt.show()