import astropy.io.fits as pf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from math import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim import Adam

def plot_style():
   
    a={'legend.fontsize': 15,
       'axes.labelsize': 20,
       'axes.titlesize':20,
       'xtick.labelsize':15,
       'ytick.labelsize':15,
       'xtick.major.size':5,
       'xtick.minor.size':2.5,
       'ytick.major.size':5,
       'ytick.minor.size':2.5,
       'figure.facecolor':'w',
       'lines.linewidth' : 1.5,
       'xtick.major.width':2,
       'ytick.major.width':2,
       'xtick.minor.width':1.5,
       'ytick.minor.width':1.5,
       'axes.linewidth':1.5,
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
from sklearn.model_selection import train_test_split as tts
# Build training dataset & test dataset
x_train, x_test, y_train, y_test = tts(array, PSB_label, test_size = 0.4)
x_train = x_train.reshape(-1, 1, len(input_index))
x_test = x_test.reshape(-1, 1, len(input_index))
whole_dataset = array.reshape(-1, 1, len(input_index))

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BinaryClassifierWithCNN(nn.Module):
    def __init__(self, input_length, fc1_units=128):
        super(BinaryClassifierWithCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=1, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(32)

        self.flatten = nn.Flatten()

        # Compute output size after convolution
        fc_input_size = 32 * input_length  # 32 channels * sequence length

        self.fc1 = nn.Linear(fc_input_size, fc1_hidden_units)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(fc1_hidden_units, 1)  # Output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = F.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Binary classification output
        return x.squeeze(1)

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, fc1_hidden_units=128):
        super(BinaryClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, kernal, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(kernal * input_size)
        self.conv2 = nn.Conv1d(kernal, kernal, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(kernal * input_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(kernal * input_size, fc1_hidden_units)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(fc1_hidden_units, 1)  # Output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = F.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)  

# Assuming you have y_train and y_test as your labels
train_labels = torch.tensor(y_train, dtype=torch.float32)
test_labels = torch.tensor(y_test, dtype=torch.float32)
whole_labels = torch.tensor(PSB_label, dtype=torch.float32)

train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float32), train_labels)
test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float32), test_labels)
whole_data = TensorDataset(torch.tensor(whole_dataset, dtype=torch.float32), whole_labels)

batch_size = 500
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Create an instance of the model
#model = BinaryClassifierWithCNN(len(input_index))
model = BinaryClassifier(len(input_index))

# Assuming model is an instance of your BinaryClassifier
model = model.to(device)  # Move the entire model to the GPU

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 1000  # You can adjust this
for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        # Assuming inputs is your input data
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs to the GPU

        # Now perform the forward pass
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Adjust threshold
threshold = 0.8

model.eval()
predicted_labels = []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        # Assuming inputs is your input data
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs to the GPU

        # Now perform the forward pass
        outputs = model(inputs)

        predicted = (outputs > threshold).float()
        predicted_labels.extend(predicted.cpu().numpy())

# Convert the list to a numpy array
predicted_labels = np.array(predicted_labels)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, precision_score, f1_score, recall_score, accuracy_score
import seaborn as sns
plot_style()
CM = confusion_matrix(y_test, predicted_labels)
ax= plt.subplot()
sns.heatmap(CM, center = True,cbar=True,annot=True,ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
path = '/work1/kaixiang/fig/sequential_model/'
plt.savefig(path + 'output.png')

class PyTorchWrapper(nn.Module):
    def __init__(self, model):
        super(PyTorchWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))

from sklearn.inspection import permutation_importance
import numpy as np

def compute_permutation_importance(model, dataloader, criterion, device):
    model.eval()
    true_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            true_labels.extend(labels.cpu().numpy().astype(int))

    true_labels = np.array(true_labels)

    def score_function(model, x_test, y_test):
        model.eval()
        with torch.no_grad():
            outputs = model(x_test)
            predicted_labels = (outputs > threshold).float().cpu().numpy()

        precision = precision_score(y_test, predicted_labels)
        return precision

    result = permutation_importance(
        PyTorchWrapper(model), x_test, y_test, scoring=score_function,
        n_repeats=100, random_state=44, n_jobs=-1
    )

    return result


# Assuming model, criterion, and device are defined
result = compute_permutation_importance(model, test_loader, criterion, device)

# Now you can plot the importances with error bars
plot_style()
fig, ax = plt.subplots(figsize=(15,8))
forest_importances = pd.Series(result.importances_mean, index=input_index)
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_ylabel("Mean precision decrease")
fig.tight_layout()
plt.savefig(path+'Permutation_Feature_Importance.png')

# Plot color-color diagram
plot_style()
plt.figure(figsize=(10,8))
plt.xlabel('Rest-frame r-J')
plt.ylabel('Rest-frame NUV-r')

whole_dataloader = DataLoader(whole_data, batch_size=batch_size, shuffle=False)

predicted_labels_whole = []
with torch.no_grad():
    for inputs, labels in whole_dataloader:
        # Assuming inputs is your input data
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs to the GPU

        # Now perform the forward pass
        outputs = model(inputs)

        predicted = (outputs > threshold).float()
        predicted_labels_whole.extend(predicted.cpu().numpy())

predition = predicted_labels_whole

plt.scatter(rJ_color, NUVr_color, c='0.4', s=2, alpha = 0.5)
plt.scatter(rJ_color[predition], NUVr_color[predition], c='limegreen', s=2, label='Classified PSB')
leftx, rightx = np.min(rJ_color), np.max(rJ_color)
lefty, righty = np.min(NUVr_color), np.max(NUVr_color)
plt.xlim(leftx, rightx)
plt.ylim(lefty, righty)
c_cut = 2.5
plt.hlines(y = c_cut, xmin = leftx, xmax = (c_cut-1)/3, color='red', ls='--')
plt.plot([(c_cut-1)/3, (righty-1)/3], [c_cut, righty], c='red', ls='--')
plt.vlines(x = 0.5, ymin = c_cut, ymax = righty, color='red', ls='--')
plt.legend()
plt.savefig(path+'color-color_diagram.png')

# Record the scores
model_score = np.zeros(1, dtype={'names':('Acc', 'P', 'R', 'F1', 'AccErr', 'PErr', 'RErr', 'F1Err'),\
                                 'formats':('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')})

Acc = []
P = []
R = []
F1 = []

results = {}

from sklearn.utils import resample
# Set the number of bootstrap samples
n_bootstrap = 500  # You can adjust this number

# Perform bootstrapping
for i in range(n_bootstrap):
    x_train, x_test, y_train, y_test = tts(array, PSB_label, test_size = 0.4)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).reshape(-1, 1, len(input_index))
    x_test = scaler.transform(x_test).reshape(-1, 1, len(input_index))
    # Generate a bootstrap sample
    x_train_boot, y_train_boot = resample(x_train, y_train)
    x_train_boot = x_train_boot.reshape(-1, 1, len(input_index))
    # Assuming you have y_train and y_test as your labels
    train_labels = torch.tensor(y_train_boot, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.float32)

    train_data = TensorDataset(torch.tensor(x_train_boot, dtype=torch.float32), train_labels)
    test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float32), test_labels)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            # Assuming inputs is your input data
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs to the GPU

            # Now perform the forward pass
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            # Assuming inputs is your input data
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs to the GPU

            # Now perform the forward pass
            outputs = model(inputs)
            predicted = (outputs > threshold).float()
            predicted_labels.extend(predicted.cpu().numpy())

    # Convert the list to a numpy array
    predicted_labels = np.array(predicted_labels)

    Acc.append(accuracy_score(y_test, predicted_labels))
    P.append(precision_score(y_test, predicted_labels))
    R.append(recall_score(y_test, predicted_labels))
    F1.append(f1_score(y_test, predicted_labels))

# Record the scores as pickle file
model_score['Acc'] = [np.mean(Acc)]
model_score['P'] = [np.mean(P)]
model_score['R'] = [np.mean(R)]
model_score['F1'] = [np.mean(F1)]
model_score['AccErr'] = [np.std(Acc)]
model_score['PErr'] = [np.std(P)]
model_score['RErr'] = [np.std(R)]
model_score['F1Err'] = [np.std(F1)]
results['sequential_model'] = model_score

import pickle
path = '/work1/kaixiang/fig/sequential_model/'
with open(path+'model_score_sequential_model.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)