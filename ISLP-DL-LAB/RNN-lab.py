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


max_num_workers=8
(imdb_seq_train, imdb_seq_test) = load_sequential(root='data/IMDB')
padded_sample = np.asarray(imdb_seq_train.tensors[0][0])
sample_review = padded_sample[padded_sample > 0][:12]

print(sample_review[:12])

# loading in the documents
imdb_seq_dm = SimpleDataModule (imdb_seq_train, 
                                imdb_seq_test, 
                                validation=2000, 
                                batch_size=300, 
                                num_workers=min(6, max_num_workers)
                                )


# defining the Neural Network  model
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel , self).__init__()
        self.embedding = nn.Embedding(input_size, 32)
        self.lstm = nn.LSTM(input_size=32, 
                            hidden_size=32, 
                            batch_first=True)
        self.dense = nn.Linear(32, 1) 

    def forward(self, x): 
        val, (h_n, c_n)=self.lstm(self.embedding(x))
        return torch.flatten(self.dense(val[:,-1]))
    
((X_train, Y_train), 
 (X_valid, Y_valid), 
 (X_test, Y_test)) = load_sparse(validation=2000, 
                                 random_state=0, 
                                 root='data/IMDB')
            
lstm_model = LSTMModel(X_test.shape[-1])
summary(lstm_model, input_data=imdb_seq_train.tensors[0][:10], col_names=['input_size', 'output_size', 'num_params'])

# logging the results to a csv
lstm_module = SimpleModule. binary_classification(lstm_model)
lstm_logger = CSVLogger('logs', name='IMDB_LSTM')

lstm_trainer = Trainer(deterministic=True, 
                       max_epochs=20, 
                       logger=lstm_logger,
                       callbacks=[ ErrorTracker ()])

lstm_trainer.fit(lstm_module, datamodule=imdb_seq_dm)
print("RNN results:", lstm_trainer.test(lstm_module , datamodule=imdb_seq_dm))

def summary_plot(results, ax, col='loss', valid_legend='Validation', training_legend='Training', ylabel='Loss', fontsize=20):
    for(column, color, label) in zip([f'train_{col}_epoch', f'valid_{col}'], ['black', 'red'], [training_legend, valid_legend]):
        results.plot(x='epoch', y=column, label=label, marker='o', color=color, ax=ax)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)

    return ax

# print learning curves
lstm_results = pd.read_csv(lstm_logger.experiment.metrics_file_path)
fig, ax = subplots(1, 1, figsize =(6, 6)) 
summary_plot(lstm_results, ax, col='accuracy', ylabel='Accuracy')
ax.set_xticks(np.linspace (0, 20, 5). astype(int))
ax.set_ylabel('Accuracy')
ax.set_ylim([0.5, 1])

plt.show()

# clean up
del(lstm_model, 
    lstm_trainer,
    lstm_logger,
    imdb_seq_dm, 
    imdb_seq_train, 
    imdb_seq_test)