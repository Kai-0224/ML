import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import astropy.io.fits as pf
from math import *
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Load data
data_path = '/work1/kaixiang/data_COSMOS2020/'
research_path = '/work1/kaixiang/research/'
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

color = []
color_index = []
for i in range(len(bands)-1):
    for j in range(i+1,len(bands)):
        color_index.append(bands[i]+'-'+bands[j])
        if j==1:
            color = phot[:,j]-phot[:,i]
        else:
            color = np.c_[color, phot[:,j]-phot[:,i]]

### Input: mag + color
array = np.c_[phot, color]
input_index = np.concatenate((bands, color_index), axis=None)
PSB_label = PSB

# Data normalization
from sklearn.preprocessing import StandardScaler
# Build training dataset & test dataset
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(array, PSB_label, test_size = 0.4 ,random_state=42)
whole_data = array

# Convert data to PyTorch tensors
train_data = torch.tensor(x_train[~y_train], dtype=torch.float32)
train_labels = torch.tensor(y_train[~y_train], dtype=torch.float32)
test_data = torch.tensor(x_test, dtype=torch.float32)
whole_data = torch.tensor(whole_data, dtype=torch.float32)

# Create a train_dataloader
batch_size = 500
train_dataset = TensorDataset(train_data, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the autoencoder model with convolutional layers
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),  # Added BatchNorm
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(8),  # Added BatchNorm
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(4),  # Added BatchNorm
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(8),  # Added BatchNorm
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),  # Added BatchNorm
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid activation to constrain output between 0 and 1
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for 1D convolution
        x = self.encoder(x)
        x = self.decoder(x).squeeze(1)  # Remove channel dimension
        return x

'''
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid activation to constrain output between 0 and 1
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for 1D convolution
        x = self.encoder(x)
        x = self.decoder(x).squeeze(1)  # Remove channel dimension
        return x
'''
# Instantiate the model, loss function, and optimizer
model = ConvAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the autoencoder on normal data
num_epochs = 50

for epoch in range(num_epochs):
    for batch_data, batch_labels in train_dataloader:
        # Filter out anomaly data
        normal_batch_data = batch_data

        # Forward pass
        outputs = model(normal_batch_data)
        loss = criterion(outputs, normal_batch_data)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# Detect the threshold of anomalies based on reconstruction error
model.eval()
with torch.no_grad():
    reconstructed_data = model(train_data)
    mse_loss = torch.mean((reconstructed_data - train_data)**2, dim=1)
    threshold = torch.mean(mse_loss) + 2.5 * torch.std(mse_loss)  # Adjust the threshold based on your dataset

# Evaluate the model on test data (which includes both normal and anomaly data)
model.eval()
with torch.no_grad():
    reconstructed_data = model(test_data)
    mse_loss = torch.mean((reconstructed_data - test_data)**2, dim=1)

# Identify anomalies based on the threshold
anomalies = (mse_loss > threshold).numpy().astype(bool)

# Evaluate performance metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, f1_score, recall_score, accuracy_score

CM = confusion_matrix(y_test, anomalies)
ax= plt.subplot()
sns.heatmap(CM, center = True,cbar=True,annot=True,ax=ax)
ax.set_xlabel('Predicted Label',fontsize=10)
ax.set_ylabel('True Label',fontsize=10)
path = '/work1/kaixiang/fig/anomaly/'
plt.savefig(path + 'output.png')

# Record the scores
model_score = np.zeros(1, dtype={'names':('Acc', 'P', 'R', 'F1', 'ROC AUC'),'formats':('f8', 'f8', 'f8', 'f8', 'f8')})

Acc = []
P = []
R = []
F1 = []
AUROC = []

results = {}

Acc.append(accuracy_score(y_test, anomalies))
P.append(precision_score(y_test, anomalies))
R.append(recall_score(y_test, anomalies))
F1.append(f1_score(y_test, anomalies))
AUROC.append(roc_auc_score(y_test,anomalies))

# Plot color-color diagram
plt.figure(figsize=(10,8))
plt.xlabel('Rest-frame r-J')
plt.ylabel('Rest-frame NUV-r')

# Apply the autoencoder
model.eval()
with torch.no_grad():
    whole_reconstructed_data = model(whole_data)
    whole_mse_loss = torch.mean((whole_reconstructed_data - whole_data)**2, dim=1)

# Identify anomalies in the test set
predition = (whole_mse_loss > threshold).numpy().astype(bool)

plt.scatter(rJ_color, NUVr_color, c='0.4', s=2, alpha = 0.5)
plt.scatter(rJ_color[predition], NUVr_color[predition], c='limegreen', s=8, label='Classified PSB')
leftx, rightx = np.min(rJ_color), np.max(rJ_color)
lefty, righty = np.min(NUVr_color), np.max(NUVr_color)
plt.xlim(leftx, rightx)
plt.ylim(lefty, righty)
c_cut = 3.1
plt.hlines(y = c_cut, xmin = leftx, xmax = (c_cut-1)/3, color='red', ls='--')
plt.plot([(c_cut-1)/3, (righty-1)/3], [c_cut, righty], c='red', ls='--')
plt.vlines(x = 0.5, ymin = c_cut, ymax = righty, color='red', ls='--')
plt.legend()
plt.savefig(path+'color-color_diagram.png')

# Record the scores as pickle file
model_score['Acc'] = Acc
model_score['P'] = P
model_score['R'] = R
model_score['F1'] = F1
model_score['ROC AUC'] = AUROC
results['autoencoder'] = model_score

import pickle
with open(path+'model_score.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)