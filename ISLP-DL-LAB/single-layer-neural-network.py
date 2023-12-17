import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from sklearn.linear_model import (LinearRegression, LogisticRegression, Lasso)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from sklearn.model_selection import (train_test_split, GridSearchCV)

import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset

from torchmetrics import (MeanAbsoluteError, R2Score)
from torchinfo import summary
from torchvision.io import read_image
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from pytorch_lightning .utilities.seed import seed_everything

seed_everything(0, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)

from torchvision.datasets import MNIST, CIFAR100
from torchvision.models import (resnet50, ResNet50_Weights)
from torchvision.transforms import (Resize, Normalize, CenterCrop, ToTensor)

from ISLP.torch import (SimpleDataModule, SimpleModule, ErrorTracker, rec_num_workers)

from ISLP.torch.imdb import (load_lookup, load_tensor, load_sparse, load_sequential)
from glob import glob
import json

Hitters = load_data('Hitters').dropna()
n = Hitters.shape [0]

model = MS(Hitters.columns.drop('Salary'), intercept=False)
X = model.fit_transform(Hitters).to_numpy ()
Y = Hitters['Salary'].to_numpy()

# Splitting the data

(X_train,
    X_test,
    Y_train,
    Y_test) = train_test_split(X, 
                               Y, 
                               test_size=1/3, 
                               random_state=1)

# Fitting a linear model
print("--- Fitting a linear model ---")
hit_lm = LinearRegression ().fit(X_train, Y_train)
Yhat_test = hit_lm.predict(X_test)
linear_model_error = np.abs(Yhat_test - Y_test).mean()
print("Linear model error:", linear_model_error)

# Fitting the Lasso
print("--- Fitting a Lasso model ---")
scaler = StandardScaler(with_mean=True, with_std=True)
lasso = Lasso(warm_start=True, max_iter=30000)
standard_lasso = Pipeline(steps =[('scaler', scaler), ('lasso', lasso)])

X_s = scaler.fit_transform(X_train)
n = X_s.shape[0]
lam_max = np.fabs(X_s.T.dot(Y_train - Y_train.mean ())).max() / n
param_grid = {'alpha': np.exp(np.linspace(0,np.log(0.01), 100)) * lam_max}

# k-fold validation
cv = KFold(10, shuffle=True, random_state=1)
grid = GridSearchCV(lasso, param_grid, cv=cv, scoring='neg_mean_absolute_error')
grid.fit(X_train, Y_train);

trained_lasso = grid.best_estimator_
Yhat_test = trained_lasso.predict(X_test)
lasso_model_error = np.fabs(Yhat_test - Y_test).mean()
print("Lasso model error:", lasso_model_error)

# Fitting a neural network
print("--- Fitting a Neural Network ---")

# This is the definition of the model structure, defined as a class
class HittersModel(nn.Module):
    def __init__(self, input_size):
        super(HittersModel, self).__init__ ()
        self.flatten = nn.Flatten ()
        self.sequential = nn.Sequential(
        nn.Linear(input_size, 50), nn.ReLU(), nn.Dropout(0.4), nn.Linear(50, 1))

    def forward(self, x):
        x = self.flatten(x)
        return torch.flatten(self.sequential(x))
    

# Creating a HittersModel object    
hit_model = HittersModel(X.shape[1])

dl_model_summary = summary(hit_model, input_size=X_train.shape, col_names =['input_size', 'output_size', 'num_params'])
print(dl_model_summary)


# Converting the test and training data into 32-bit float
X_train_t = torch.tensor(X_train.astype(np.float32))
Y_train_t = torch.tensor(Y_train.astype(np.float32))
hit_train = TensorDataset(X_train_t, Y_train_t)

X_test_t = torch.tensor(X_test.astype(np.float32))
Y_test_t = torch.tensor(Y_test.astype(np.float32))
hit_test = TensorDataset(X_test_t, Y_test_t)

max_num_workers = rec_num_workers()

hit_dm = SimpleDataModule(hit_train, hit_test, batch_size=32, num_workers=min(4, max_num_workers), validation=hit_test)

hit_module = SimpleModule.regression(hit_model, metrics ={'mae':MeanAbsoluteError()})

# Logging the results in a csv file
hit_logger = CSVLogger('logs', name='hitters')

hit_trainer = Trainer(deterministic=True, max_epochs =50, log_every_n_steps=5, logger=hit_logger, callbacks=[ ErrorTracker()])
hit_trainer.fit(hit_module, datamodule=hit_dm)

neural_network_results = hit_trainer.test(hit_module, datamodule=hit_dm)
print("Neural Network error:", neural_network_results)

# plotting results
hit_results = pd.read_csv(hit_logger.experiment.metrics_file_path)

def summary_plot(results, ax, col='loss', valid_legend='Validation', training_legend='Training', ylabel='Loss', fontsize=20):
    for(column, color, label) in zip([f'train_{col}_epoch', f'valid_{col}'], ['black', 'red'], [training_legend, valid_legend]):
        results.plot(x='epoch', y=column, label=label, marker='o', color=color, ax=ax)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)

    return ax

fig, ax = subplots (1, 1, figsize=(6, 6)) 
ax = summary_plot(hit_results, ax, col='mae', ylabel='MAE', valid_legend='Validation (= Test)')
ax.set_ylim([0, 400])
ax.set_xticks(np.linspace(0, 50, 11).astype(int));

hit_model.eval()
preds = hit_module(X_test_t)
neural_network_error = torch.abs(Y_test_t - preds).mean()
print("Neural Network error:", neural_network_error)

plt.show()

# clean up
del(Hitters, 
    hit_model,
    hit_dm,
    hit_logger, 
    hit_test, 
    hit_train, 
    X, 
    Y, 
    X_test, 
    X_train, 
    Y_test, 
    Y_train, 
    X_test_t, 
    Y_test_t, 
    hit_trainer, 
    hit_module)
