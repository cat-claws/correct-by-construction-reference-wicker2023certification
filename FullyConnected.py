import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import FairCertModule
from typing import Union
import numpy as np 

class FullyConnected(pl.LightningModule):
    
    def __init__(self, hidden_dim: int = 128, hidden_lay: int = 1, 
                 learning_rate: float = 0.001,  
                 dataset="Folk", in_dim=20, num_cls=2, class_weights=torch.Tensor([1.,1.]), mode="NONE", glob_advs = 0,
                 epsilon: float = 0.00, alpha: float = 0.50):
        super().__init__()
        self._current_index = 0
        self.save_hyperparameters()
        HIDDEN_LAY = hidden_lay
        self.activations = [torch.relu]
        self.dataset = dataset
        self.class_weights = class_weights
    
        self.in_dim = in_dim
        self.num_cls = num_cls
        self.gamma = 0.03
        self.glob_advs = glob_advs
        # ------------------------------------------------------------------------------------
        # Setting up model passes 
        # ------------------------------------------------------------------------------------
        
        self.lays=nn.ModuleList()
        self.layers = []
        self.l1 = torch.nn.Linear(self.in_dim, self.hparams.hidden_dim)
        self.lays.append(self.l1); self.activations.append(torch.relu)
        self.layers.append("Linear")
        
        for i in range(HIDDEN_LAY - 1):
            self.lays.append(torch.nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim))
            self.activations.append(torch.relu)
            self.layers.append("Linear")
          
        self.lf = torch.nn.Linear(self.hparams.hidden_dim, self.num_cls)
        self.lays.append(self.lf)
        self.layers.append("Linear")
        
        # ------------------------------------------------------------------------------------
        
        self.ALPHA = alpha              # Regularization Parameter (Weights the Reg. Term)
        self.EPSILON = epsilon          # Input Peturbation Budget at Training Time

        self.LEARN_RATE = learning_rate # Learning Rate Hyperparameter
        self.MAX_EPOCHS = 10            # Maximum Epochs to Train the Model for

        self.EPSILON_LINEAR = True      # Put Epsilon on a Linear Schedule?
        self.extra_epochs = 0
        
        if(self.EPSILON_LINEAR):
            self.eps = 0.0
        else:
            self.eps = self.EPSILON
        self.mode = mode.upper()
        self.inputfooling = False
       
    def set_params(self, **kwargs):
        self.ALPHA =  kwargs.get('alpha', 0.00)
        self.EPSILON =  kwargs.get('epsilon', 0.00)
        self.LEARN_RATE =  kwargs.get('learn_rate', 0.001)
        self.MAX_EPOCHS =  int(kwargs.get('max_epochs', 15))
        self.EPSILON_LINEAR = bool(kwargs.get('epsilon_linear', True))
        self.GAMMA_LINEAR = bool(kwargs.get('gamma_linear', True))
        if(self.EPSILON_LINEAR):
            self.eps = 0.0
        else:
            self.eps = self.EPSILON
           
    def set_fair_interval(self, interval):
        self.fair_interval = torch.Tensor(interval)
        
    def forward(self, x, index=0):
        """
        Multiple head forward pass uses the index parameter
        """
        x = x.view(x.size(0), -1)
        for i in range(len(self.lays)-1):
            x = torch.relu(self.lays[i](x))
        x = self.lf(x)
        return x

    def classify(self, x):
        outputs = self.forward(x)
        return F.softmax(outputs, dim=1), torch.max(outputs, 1)[1]
    
    def loss(self, x, y, weight=-1):
        return F.cross_entropy(x, y, weight=self.class_weights)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if(self.glob_advs != 0):
            x_glob_worst = FairCertModule.global_pgd_ibp(self, self.fair_interval, self.eps, self.glob_advs)
            y_glob_worst = torch.randint(0, 1, (1,self.glob_advs)).long()
            regval = FairCertModule.fairness_regularizer(self, x_glob_worst, y_glob_worst, 
                                                         self.fair_interval, self.eps,
                                                         nclasses=self.num_cls)
            regval /= self.glob_advs
        else:
            regval = 0.0
        if self.mode == "FAIR-IBP":
            regval += FairCertModule.fairness_regularizer(self, x, y, self.fair_interval, self.eps,
                                                         nclasses=self.num_cls)
            #print("2: Full regval:  ", regval)
            #print("In IBP, delta: ", regval)
            loss = ((1-self.ALPHA)*self.loss(y_hat, y)) + (self.ALPHA * regval)
        elif self.mode == "FAIR-PGD":
            #print("in pgd")
            regval += FairCertModule.fairness_regularizer_PGD(self, x, y, self.fair_interval, self.eps,
                                                         nclasses=self.num_cls)    
            loss = ((1-self.ALPHA)*self.loss(y_hat, y)) + (self.ALPHA * regval)
        elif self.mode == "FAIR-DRO":
            #print("in pgd")
            x_dro_worst = FairCertModule.global_pgd_ibp_s(self, self.fair_interval, self.eps, self.gamma, x)   
            y_dro_worst = torch.randint(0, 1, (1,len(x_dro_worst))).long()
            regval = FairCertModule.fairness_regularizer(self, x_dro_worst, y_dro_worst, 
                                                         self.fair_interval, self.eps,
                                                         nclasses=self.num_cls)
            #print(regval)
            loss = ((1-self.ALPHA)*self.loss(y_hat, y)) + (self.ALPHA * regval)
        else:
            loss = self.loss(y_hat, y, weight=self.class_weights)
        return loss
  
    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    def validation_epoch_end(self, outputs) -> None:
        self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True)
        if(self.EPSILON_LINEAR):
            self.eps += self.EPSILON/self.MAX_EPOCHS
            self.eps = min(self.eps, self.EPSILON)
            print("Updated eps: ", self.eps)
    def test_epoch_end(self, outputs) -> None:
        self.log("test_acc", torch.stack(outputs).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LEARN_RATE, weight_decay=1e-4)
        #return torch.optim.SGD(self.parameters(), lr=self.LEARN_RATE, weight_decay=1e-5)
        
    def predict_proba(self, data: Union[torch.FloatTensor, np.array]) -> np.array:
        """
        Computes probabilistic output for c classes
        :param data: torch tabular input
        :return: np.array
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = data.float()

        return self.forward(input).detach().numpy()
    
    def predict(self, data):
        """
        :param data: torch or list
        :return: np.array with prediction
        """
        
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = torch.squeeze(data).float()
        
        outputs = self.forward(input)
        s = F.softmax(outputs, dim=1)
        return s.detach().numpy()
    
    def __iter__(self):
        return self
    
    def __next__(self): 
        return self.lays.__next__()
    