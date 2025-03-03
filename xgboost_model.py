import astropy.io.fits as pf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from math import *

def plot_style():
   
    a={'legend.fontsize': 12,
       'axes.labelsize': 15,
       'axes.titlesize':15,
       'xtick.labelsize':12,
       'ytick.labelsize':12,
       'xtick.major.size':8,
       'xtick.minor.size':4,
       'ytick.major.size':8,
       'ytick.minor.size':4,
       'figure.facecolor':'w',
       'lines.linewidth' : 1.5,
       'xtick.major.width':1.5,
       'ytick.major.width':1.5,
       'xtick.minor.width':1,
       'ytick.minor.width':1,
       'axes.linewidth':1,
       'xtick.direction':'in',
       'ytick.direction':'in',
       'ytick.labelleft':True,
       'text.usetex' : False,
       'font.family': 'sans-serif'}
  
    plt.rcParams.update(a)

# Load data
data_path = '/work1/kaixiang/data_COSMOS2020/'
data = pf.open(data_path + 'COSMOS2020_FARMER_R1_v2.2_p3.fits')
catalog = data[1].data

# Select data
gg = (catalog['MODEL_FLAG'] == 0) & (catalog['FLAG_COMBINED'] == 0) &\
(catalog['lp_zPDF'] > 0) & (np.abs(catalog['lp_zPDF']-catalog['lp_zPDF_l68']) <= 0.5) &\
(np.abs(catalog['lp_zPDF_u68']-catalog['lp_zPDF']) <= 0.5) & (catalog['lp_type'] == 0) &\
(~np.isnan(catalog['lp_MNUV'])) & (~np.isnan(catalog['lp_MR'])) & (~np.isnan(catalog['lp_MJ'])) &\
(catalog['lp_NbFilt'] > 8)

gg_data = catalog[gg]

### Select input data
bands = ['CFHT_u', 'HSC_g', 'HSC_r', 'HSC_i', 'HSC_z', 'UVISTA_Y', 'UVISTA_J', 'UVISTA_H', 'UVISTA_Ks']
band_index = ['u', 'g', 'r', 'i', 'z', 'Y', 'J', 'H', 'Ks']
for i in bands:
    if i=='CFHT_u':
        gg1 = (~np.isnan(gg_data[i+'_MAG'])) & (~np.isnan(gg_data[i+'_MAGERR'])) & (gg_data[i+'_VALID']) &\
        (gg_data[i+'_FLUX']/gg_data[i+'_FLUXERR'] >=5)
    else:
        gg1 = gg1 & \
        (~np.isnan(gg_data[i+'_MAG'])) & (~np.isnan(gg_data[i+'_MAGERR'])) & (gg_data[i+'_VALID']) &\
        (gg_data[i+'_FLUX']/gg_data[i+'_FLUXERR'] >=5)

gg_data = gg_data[gg1]

### Calculate U-V and V-J colors to select post-starburst galaxies
rJ_color = gg_data['lp_MR'] - gg_data['lp_MJ']
NUVr_color = gg_data['lp_MNUV'] - gg_data['lp_MR']

### Make labels for post-starburst galaxies
PSB = (NUVr_color > 2.5) & (NUVr_color > 3*rJ_color + 1) & (rJ_color < 0.5)

### Make the input array
phot = []
for i,band in enumerate(bands):
    if i==0:
        phot = gg_data[bands[i]+'_MAG']
    else:
        phot = np.c_[phot, gg_data[band+'_MAG']]

phot_err = []
ebands = ['e'+i for i in band_index]
for i,band in enumerate(bands):
    if i==0:
        phot_err = gg_data[bands[i]+'_MAGERR']
    else:
        phot_err = np.c_[phot_err, gg_data[band+'_MAGERR']]

color = []
color_index = []
for i in range(len(bands)-1):
    for j in range(i+1,len(bands)):
        color_index.append(band_index[i]+'-'+band_index[j])
        if j==1:
            color = phot[:,j]-phot[:,i]
        else:
            color = np.c_[color, phot[:,j]-phot[:,i]]

### Input
#array = np.c_[phot,phot_err,color]
#input_index = np.concatenate((band_index,ebands,color_index), axis=None)
#array = np.c_[phot,color]
#input_index = np.concatenate((band_index,color_index), axis=None)
#array = np.c_[phot,phot_err]
#input_index = np.concatenate((band_index,ebands), axis=None)
array = color
input_index = color_index
#array = phot
#input_index = band_index

PSB_label = PSB

# Data normalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
# Build training dataset & test dataset
x_train, x_test, y_train, y_test = tts(array, PSB_label, test_size = 0.4)
x_train, x_val, y_train, y_val = tts(x_train, y_train, test_size = 0.4)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.fit_transform(x_val)
x_test = scaler.transform(x_test)
whole_data = scaler.transform(array)

import xgboost as xgb
params = {
    'max_depth': 6,
    'eta': 0.01,
    'objective': 'multi:softprob',
    'num_class': 2,
    'tree_method': 'hist',
    'eval_metric': 'mlogloss',
    'subsample': 0.8,
    'eval_metric':'auc',
    'device': 'cuda',
}
'''
params = {
    'booster': 'gbtree',  # Use gradient boosting trees
    'colsample_bynode': 0.8,
    'max_depth': 5,
    'num_parallel_tree': 100,  # Number of trees in the ensemble
    'eta': 0.01,
    'objective': 'multi:softprob',
    'num_class': 2,
    'tree_method': 'hist',
    'eval_metric': 'mlogloss',
    'subsample': 0.8,
    'max_delta_step': 1,
    'eval_metric':'auc',
    'device': 'cuda',
}
'''

# Convert data to DMatrix format
x_train_df = pd.DataFrame(x_train, columns=input_index)
x_val_df = pd.DataFrame(x_val, columns=input_index)
x_test_df = pd.DataFrame(x_test, columns=input_index)
whole_df = pd.DataFrame(whole_data, columns=input_index)

dtrain = xgb.DMatrix(x_train_df, label=y_train,feature_names=input_index)
dval = xgb.DMatrix(x_val_df, label=y_val,feature_names=input_index)
dtest = xgb.DMatrix(x_test_df, label=y_test,feature_names=input_index)
dwhole = xgb.DMatrix(whole_df, label=PSB_label,feature_names=input_index)

# Train the XGBoost model
num_round = 5000
evals = [(dtrain, 'train'), (dval, 'validation')]
bst = xgb.train(params, dtrain, num_round, evals=evals, early_stopping_rounds=50, num_boost_round=1)

# Make predictions on the test set
y_pred_prob = bst.predict(dtest)

# Thresholding
threshold = 0.6
y_pred = (y_pred_prob[:, 1] > threshold)

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, f1_score, recall_score, accuracy_score
plot_style()
fig, ax = plt.subplots(figsize=(8,6))
CM = confusion_matrix(y_test, y_pred)
sns.heatmap(CM, center = True,cbar=True,annot=True,ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
path = '/work1/kaixiang/fig/xgboost/'
plt.savefig(path + 'output.png')

# Plot color-color diagram
plot_style()
fig, ax = plt.subplots(figsize=(8,6))
plt.xlabel(r'$r-J$')
plt.ylabel(r'$NUV-r$')

# Make predictions on the test set
y_pred_whole_prob = bst.predict(dwhole)
predition = (y_pred_whole_prob[:, 1] > threshold)

ax.scatter(rJ_color, NUVr_color, c='0.4', s=1, alpha = 0.5)
ax.scatter(rJ_color[predition], NUVr_color[predition], c='limegreen', s=2, label='Classified PSB')
leftx, rightx = np.min(rJ_color), np.max(rJ_color)
lefty, righty = np.min(NUVr_color), np.max(NUVr_color)
plt.xlim(leftx, rightx)
plt.ylim(lefty, righty)
c_cut = 2.5
plt.hlines(y = c_cut, xmin = leftx, xmax = (c_cut-1)/3, color='red', ls='--')
plt.plot([(c_cut-1)/3, (righty-1)/3], [c_cut, righty], c='red', ls='--')
plt.vlines(x = 0.5, ymin = c_cut, ymax = righty, color='red', ls='--')
plt.legend(frameon=False, numpoints=1)
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
plt.savefig(path+'color-color_diagram_ml.png')

# Feature Importance
plot_style()
fig, ax = plt.subplots(figsize=(8, 15))
xgb.plot_importance(bst, title=None, xlabel='F score', ylabel=None,\
                    importance_type='weight', ax=ax, capsize=5)
fig.tight_layout()
plt.savefig(path + 'Importance.png')

# Redshift Distribution
z_bin = np.arange(0,3.01,0.1)
z_bin_med = np.arange(0.05,3.01,0.1)
purity = []
completeness = []
for i in range(len(z_bin)-1):
    z_range = (gg_data['lp_zPDF'] > z_bin[i]) & (gg_data['lp_zPDF'] > z_bin[i+1])
    data_z = whole_data[z_range]
    data_z_label = PSB_label[z_range]
    data_z_df = pd.DataFrame(data_z, columns=input_index)
    ddata_z = xgb.DMatrix(data_z_df, label=data_z_label)
    
    # Make predictions on the test set
    y_pred_data_z_prob = bst.predict(ddata_z)
    predition_z = (y_pred_data_z_prob[:, 1] > threshold)
    purity.append(precision_score(data_z_label, predition_z))
    completeness.append(recall_score(data_z_label, predition_z))
plot_style()
fig, ax = plt.subplots(figsize=(8,6))
plt.plot(z_bin_med, purity, c='C0', label='Precision')
plt.plot(z_bin_med, completeness, c='C1', label='Recall')
plt.legend(frameon=False, numpoints=1)
plt.xlabel('z')
plt.ylabel('Score')
plt.savefig(path + 'z_dependence.png')

# Record the scores
model_score = np.zeros(1, dtype={'names':('Acc', 'P', 'R', 'F1', 'AccErr', 'PErr', 'RErr', 'F1Err'),\
                                 'formats':('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')})

Acc = []
P = []
R = []
F1 = []

results = {}

n_bootstrap = 500  # You can adjust this number

# Perform bootstrapping
for _ in range(n_bootstrap):
    x_train, x_test, y_train, y_test = tts(array, PSB_label, test_size = 0.4)
    x_train, x_val, y_train, y_val = tts(x_train, y_train, test_size = 0.4)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.fit_transform(x_val)
    x_test = scaler.transform(x_test)
    
    # Create a bootstrap sample
    indices = np.random.choice(len(x_test), len(x_test), replace=True)
    x_boot = x_test[indices]
    y_boot = y_test[indices]
    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    dtest = xgb.DMatrix(x_boot, label=y_boot)
    
    # Train the XGBoost model
    num_round = 5000
    evals = [(dtrain, 'train'), (dval, 'validation')]
    bst = xgb.train(params, dtrain, num_round, evals=evals, early_stopping_rounds=50, num_boost_round=1)
    
    # Make predictions on the test set
    y_pred_prob = bst.predict(dtest)

    # Thresholding
    y_pred = (y_pred_prob[:, 1] > threshold)

    Acc.append(accuracy_score(y_test, y_pred))
    P.append(precision_score(y_test, y_pred))
    R.append(recall_score(y_test, y_pred))
    F1.append(f1_score(y_test, y_pred))

# Record the scores as pickle file
model_score['Acc'] = [np.mean(Acc)]
model_score['P'] = [np.mean(P)]
model_score['R'] = [np.mean(R)]
model_score['F1'] = [np.mean(F1)]
model_score['AccErr'] = [np.std(Acc)]
model_score['PErr'] = [np.std(P)]
model_score['RErr'] = [np.std(R)]
model_score['F1Err'] = [np.std(F1)]
results['xgboost'] = model_score

import pickle
with open(path+'model_score.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)