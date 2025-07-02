"""
A file which simply returns the normalized training and test set for the desired dataset
"""

import os
import sys
import copy
sys.path.append('..')

import torch
import random
import numpy as np
import pandas as pd
import FairCertModule
from FairCertModule import ACS_categories
import folktables
from folktables import ACSDataSource, ACSIncome, ACSEmployment
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
SEED = 0
random.seed(SEED)
np.random.seed(SEED)


def get_dataset(dataset, year='2015', state='CA', scaler=None, retscaler=False, groups=False):
    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    #data_source2 = ACSDataSource(survey_year='2021', horizon='1-Year', survey='person')
    if(dataset == "Folk"):
        CustomIncome = folktables.BasicProblem(
            features=[
                'AGEP',
                'COW',
                'SCHL',
                'MAR',
                'OCCP',
                'POBP',
                'RELP',
                'WKHP',
                'SEX',
                #'RAC1P',
            ],
            target='PINCP',
            target_transform=lambda x: x > 25000,    
            group='SEX',
            preprocess=folktables.adult_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )

        ca_data = data_source.get_data(states=[state],  download=True)

        ca_features, ca_labels, g = CustomIncome.df_to_pandas(ca_data, categories=ACS_categories, dummies=True)
    elif(dataset == "Employ"):
        CustomEmployment = folktables.BasicProblem(
            features=[
                'AGEP',
                'SCHL',
                'MAR',
                'RELP',
                'DIS',
                'ESP',
                'CIT',
                'MIG',
                'MIL',
                'ANC',
                'NATIVITY',
                'DEAR',
                'DEYE',
                'DREM',
                #'RAC1P',
                'SEX',
            ],
            target='ESR',
            target_transform=lambda x: x == 1,
            group='SEX',
            preprocess=lambda x: x,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        ca_data = data_source.get_data(states=[state],  download=True)
        ca_features, ca_labels, g = CustomEmployment.df_to_pandas(ca_data, categories=ACS_categories, dummies=True)  
    elif(dataset == "Insurance"):
        CustomInsurance = folktables.BasicProblem(
            features=[
                'AGEP',
                'SCHL',
                'MAR',
                'DIS',
                'ESP',
                'CIT',
                'MIG',
                'MIL',
                'ANC',
                'NATIVITY',
                'DEAR',
                'DEYE',
                'DREM',
                'PINCP',
                'ESR',
                'ST',
                'FER',
                'SEX',
            ],
            target='HINS2',
            target_transform=lambda x: x == 1,
            group='SEX',
            preprocess=lambda x: x,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        ca_data = data_source.get_data(states=[state],  download=True)
        ca_features, ca_labels, g = CustomInsurance.df_to_pandas(ca_data, categories=ACS_categories, dummies=True)
    elif(dataset == "Coverage"):
        def public_coverage_filter(data):
            df = data
            df = df[df['AGEP'] < 65]
            df = df[df['PINCP'] <= 30000]
            return df

        CustomCoverage = folktables.BasicProblem(
            features=[
                'AGEP',
                'SCHL',
                'MAR',
                'DIS',
                'ESP',
                'CIT',
                'MIG',
                'MIL',
                'ANC',
                'NATIVITY',
                'DEAR',
                'DEYE',
                'DREM',
                'PINCP',
                'ESR',
                'ST',
                'FER',
                'SEX',
                #'RAC1P',
            ],
            target='PUBCOV',
            target_transform=lambda x: x == 1,
            group='SEX',
            preprocess=public_coverage_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        ca_data = data_source.get_data(states=[state],  download=True)
        ca_features, ca_labels, g = CustomCoverage.df_to_pandas(ca_data, categories=ACS_categories, dummies=True)
        
    bin_indexes = []
    con_indexes = []
    i = 0
    for column in ca_features:
        if("_" in column):
            bin_indexes.append(i)
        else:
            con_indexes.append(i)
        i+=1
    ca_features = ca_features.to_numpy()
    ca_labels = ca_labels.to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(ca_features, ca_labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    y_val = np.squeeze(y_val)

    from sklearn.preprocessing import StandardScaler
    from sklearn.base import BaseEstimator, TransformerMixin
    class CustomScaler(BaseEstimator,TransformerMixin): 
        # note: returns the feature matrix with the binary columns ordered first  
        def __init__(self,bin_vars_index,cont_vars_index,copy=True,with_mean=True,with_std=True):
            self.scaler = StandardScaler()#(copy,with_mean,with_std)
            self.bin_vars_index = bin_vars_index
            self.cont_vars_index = cont_vars_index
        def fit(self, X, y=None):
            self.scaler.fit(X[:,self.cont_vars_index], y)
            return self

        def transform(self, X, y=None, copy=None):
            X_tail = self.scaler.transform(X[:,self.cont_vars_index],y)
            return np.concatenate((X_tail, X[:,self.bin_vars_index]), axis=1)

    if(scaler == None):
        scaler = CustomScaler(bin_indexes, con_indexes)
        scaled_data = scaler.fit_transform(X_train)
    else:
        scaler = scaler
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train = y_train.astype(int)
    y_val   = y_val.astype(int)
    y_test  = y_test.astype(int)
    
    X_train = X_train[:,:-1]
    X_test = X_test[:,:-1]
    X_val = X_val[:,:-1]

    lp_epsilon = FairCertModule.get_fairness_intervals(X_train, [-1],  metric="LP", use_sens=False, eps=1)
    lp_epsilon /= sum(lp_epsilon)
    lp_epsilon /= np.median(lp_epsilon)
    sr_epsilon = FairCertModule.get_fairness_intervals(X_train, [-1],  metric="SENSR", use_sens=False, eps=1)
    sr_epsilon /= sum(sr_epsilon)
    sr_epsilon /= np.median(sr_epsilon)

    group_train = X_train[:,-1]
    group_test = X_test[:,-1]
    group_val = X_val[:,-1]
    
    X_train = X_train[:,:-1]
    X_test = X_test[:,:-1]
    X_val = X_val[:,:-1]
    
    if(retscaler):
        return torch.tensor(X_train).float(), torch.tensor(X_test).float(), torch.tensor(X_val).float(), torch.Tensor(y_train).long(), torch.Tensor(y_test).long(), torch.Tensor(y_val).long(), lp_epsilon, sr_epsilon, scaler
    elif(groups):
        return torch.tensor(X_train).float(), torch.tensor(X_test).float(), torch.tensor(X_val).float(), torch.Tensor(y_train).long(), torch.Tensor(y_test).long(), torch.Tensor(y_val).long(), lp_epsilon, sr_epsilon, group_train, group_test, group_val
    else:
        return torch.tensor(X_train).float(), torch.tensor(X_test).float(), torch.tensor(X_val).float(), torch.Tensor(y_train).long(), torch.Tensor(y_test).long(), torch.Tensor(y_val).long(), lp_epsilon, sr_epsilon
    
  
    
from FairCertModule import * 

def fair_PGD_local(model, x_natural, lab, vec, eps, nclasses, iterations=10):
    x = x_natural.detach()
    eps_vec = vec*eps
    #noise = (eps_vec*torch.zeros_like(x).uniform_(0, 1))
    noise = (-2*eps_vec) * torch.zeros_like(x).uniform_(0, 1) + eps_vec
    #print("EPS  : ", eps_vec)
    #print("NOISE: ", abs(noise))
    x = x + (noise*eps)
    for i in range(iterations):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, lab)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x + (eps_vec/iterations * torch.sign(grad.detach()))
        #x = x.detach() + 0.5 * torch.sign(grad.detach())
        #x = torch.min(torch.max(x, x_natural - eps_vec), x_natural + eps_vec)
        x = torch.clip(x, x_natural-eps_vec, x_natural+eps_vec)  
    #print( abs(x_natural - x))
    #weights = [t for t in model.parameters()]
    #logit_l, logit_u = interval_bound_forward(model, weights, x, vec, 0.0)
    #print(logit_l)
    #print(logit_u)
    #print("PRED: ", F.softmax(logit_u))
    return x 

import torchmetrics
def evaluate_accuracy(model, X_test, y_test, average='micro'):
    y_pred = model.predict(X_test)
    y_pred = torch.Tensor(y_pred)
    a = torchmetrics.Accuracy(task='binary', average=average, num_classes=2)
    acc = a(torch.argmax(y_pred, -1), y_test)
    #acc = torch.sum(torch.eq(torch.argmax(y_pred, -1), y_test).to(torch.float32)) / len(y_test)
    return acc

def evaluate_delta_PGD(model, inp, lab, vec, eps, nclasses, iterations=10, ret_max=True):
    y_pred = F.softmax(model(inp)).detach().numpy()
    x_adv = fair_PGD_local(model, inp, lab, vec, eps, nclasses, iterations)
    y_adv = F.softmax(model(x_adv)).detach().numpy()
    #print("PRED Torch: ", y_adv)
    pgd_delta = np.max(np.abs(y_pred - y_adv), axis=1)
    if(ret_max):
        return np.mean(pgd_delta)
    else:
        return pgd_delta
    
def evaluate_delta_IBP(model, inp, lab, vec, eps, nclasses, ret_max=True):
    """
    This class only works for binary classification at the moment. Can be generalized
    with a bit of effort modifying the for loop.
    """
    weights = [t for t in model.parameters()]
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    worst_delta = 0
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
    min_logit = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    max_logit = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    min_i_softmax = F.softmax(min_logit, dim=-1)
    max_i_softmax = F.softmax(max_logit, dim=-1)
    delta = (max_i_softmax - min_i_softmax)
    delta = delta.detach().numpy()
    delta = np.abs(delta)
    if(ret_max):
        return np.mean(np.max(delta, axis=1))
    else:
        return np.max(np.squeeze(delta), axis=1)

from tqdm import trange

def I(model, inp, lab, vec, eps, nclasses, ret_max=True):
    """
    This class only works for binary classification at the moment. Can be generalized
    with a bit of effort modifying the for loop.
    """
    weights = [t for t in model.parameters()]
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    worst_delta = 0
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
    min_logit = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    max_logit = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    min_i_softmax = F.softmax(min_logit, dim=-1)
    max_i_softmax = F.softmax(max_logit, dim=-1)
    delta = (max_i_softmax - min_i_softmax)
    delta = torch.abs(delta)
    #if(ret_max):
    #    return np.max(np.max(delta, axis=1))
    #else:
    return torch.mean(delta, axis=1)

def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / rho
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    v = torch.squeeze(v)
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w


def compute_DIF_certification(model, f_epsilon, gamma, delta, X_test, y_test, N=1000, iters=500, 
                              rettrend=False, lr = 1, verbose=False, rand=False):
    convergence_trend = []
    #vg = torch.Tensor([[ (delta+gamma)*random.uniform(0, 1) for i in range(N)]]).T
    vg = torch.Tensor([[gamma for i in range(N)]]).T
    vg.requires_grad_()
    try:
        v = torch.Tensor([f_epsilon.numpy() for i in range(N)]).float()
    except:
        v = torch.Tensor([f_epsilon for i in range(N)]).float()
    if(rand):
        inds = np.random.choice(range(len(X_test)), N)
    else:
        inds = range(N)
    X = X_test[inds].float()
    y = y_test[inds].long()

    act = torch.nn.Softplus()
    invact = lambda y: torch.log(torch.exp(y)-1)
    #log(exp(x) - 1)
    vg_ = torch.nn.Parameter(invact(vg))
    opt = torch.optim.SGD([vg_], lr=lr, momentum=0.8)
    for i in trange(iters):
        with torch.enable_grad():
            eps = I(model, X, y, v, delta + act(vg_), 2) 
            loss = -1*eps #- (((torch.exp(vg_).mean() - gamma) /gamma)**2)
        opt.zero_grad()
        loss.mean().backward()
        opt.step()
        with torch.no_grad():
            # Projection onto the set of valid solutions
            n = torch.mean(act(vg_))
            if(n > gamma):
                #torch.exp(vg_) *= gamma/n
                vg_.data = invact(act(vg_.data)*gamma/n)
            #print(torch.exp(vg_.data))
        convergence_trend.append(float(eps.mean()))
    
    """
    for i in trange(iters):
        vg.requires_grad_()
        with torch.enable_grad():
            eps = I(model, X, y, v, delta + vg, 2) 
            loss = eps #- (((vg.mean() - gamma) /gamma)**2)
        loss.mean().backward()
        g = vg.grad
        
        # Gradient manipulation 
        g *= torch.where(vg==0, 0, 1)
        g -= (torch.sum(g)/torch.sum(torch.where(vg==0, 0, 1)))
        g *= torch.where(vg==0, 0, 1)
        
        if(verbose):
            print(torch.where(vg==0, 0, 1))
            print("Iter %s: "%(i), g)
            
        # Gradient-based update
        vg = vg.detach()
        vg = vg + lr*g

        # Projection onto the set of valid solutions
        n = torch.mean(vg)
        if(n >  gamma):
            vg /= n
            vg *= gamma
        vg = torch.clip(vg, 0, 1e3)
              
        convergence_trend.append(float(eps.mean()))
    """
    if(rettrend):
        eps = max(convergence_trend) #I(model, X, y, v, delta + act(vg_), 2) 
        return eps, convergence_trend, act(vg_) + delta
    else:
        eps = max(convergence_trend) #eps = I(model, X, y, v, delta + act(vg_), 2) 
        return eps
    
import copy
def compute_DIF_falsification(model, f_epsilon, gamma, delta, X_test, y_test, N=1000, iters=500, rettrend=False, rand = True, lr = 1.0):
    convergence_trend = []
    vg = torch.Tensor([[ delta for i in range(N)]]).T
    vg.requires_grad_()
    try:
        v = torch.Tensor([f_epsilon.numpy() for i in range(N)]).float()
    except:
        v = torch.Tensor([f_epsilon for i in range(N)]).float()
    if(rand):
        inds = np.random.choice(range(len(X_test)), N)
    else:
        inds = range(N)
    X = X_test[inds].float()
    X_orig = copy.deepcopy(X)
    y = y_test[inds].long()
    #lr = 100.0
    lr *= gamma
    for i in trange(iters):
        X.requires_grad_()
        with torch.enable_grad():
            eps = I(model, X, y, v, vg, 2) 
        eps.mean().backward()
        g = X.grad
        X = X.detach()
        X = X + (lr * g) #(lr*(g/torch.norm(g)))
        # n.b. this projection step is sound, but not perhaps as tight as it could be
        X = torch.clip(X, X_orig - 2*gamma, X_orig + 2*gamma) # clipping up any values that shrunk below delta
        convergence_trend.append(float(eps.mean()))
    if(rettrend):
        return eps.mean(), convergence_trend
    else:
        return eps.mean()
    


DATA_ADULT_TRAIN = 'data/adult.data.csv'
DATA_ADULT_TEST = 'data/adult.test.csv'
DATA_CRIME_FILENAME = 'data/crime.csv'
DATA_GERMAN_FILENAME = 'data/german.csv'

class TabularDataset:

    def __init__(self, X_raw, y_raw, sensitive_features=[], drop_columns=[], drop_first=False, drop_first_labels=True):
        """
        X_raw: features dataframe
        y_raw: labels dataframe or column label
        sensitive_features: the features considered sensitive
        drop_columns: the columns considered superfluous and to be deleted
        drop_first: whether to drop first when one-hot encoding features
        drop_first_labels: whether to drop first when one-hot encoding labels
        """

        self.sensitive_features = sensitive_features

        X_raw.drop(columns=drop_columns, inplace=True)
        self.X_raw = X_raw
        print("Raw cols: ")
        print(self.X_raw.shape)
        print(X_raw.columns.values.tolist())
        num_cols, cat_cols, sens_num_cols, sens_cat_cols = self.get_num_cat_columns_sorted(X_raw, sensitive_features)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.sens_num_cols = sens_num_cols
        self.sens_cat_cols = sens_cat_cols
        self.original_columns = X_raw.columns.values.tolist()

        X_df_all, y_df = self.prepare_dataset(X_raw, y_raw, drop_first=drop_first, drop_first_labels=drop_first_labels, drop_columns=drop_columns)

        self.y_df = y_df

        # Map from original column names to all new encoded ones
        all_columns_map = {}
        encoded_columns = X_df_all.columns.values.tolist()
        for c in self.original_columns:
            all_columns_map[c] = [ encoded_columns.index(e_c) for e_c in encoded_columns if e_c == c or e_c.startswith(c + '_') ]

        # List of list of the indexes of each sensitive features
        encoded_features = X_df_all.columns.values.tolist()
        sensitive_idxs = []
        sensitive_idxs_flat = []
        for sf in sensitive_features:
            sensitive_idxs.append(all_columns_map[sf])
            sensitive_idxs_flat.extend(all_columns_map[sf])
        all_idxs = [i for i in range(len(X_df_all.columns))]
        valid_idxs = [i for i in all_idxs if i not in sensitive_idxs_flat]

        # Datasets with one-hot encoded columns of each sensitive feature
        self.sensitive_dfs = [X_df_all.iloc[:, idxs] for idxs in sensitive_idxs]

        # Dataset with all features but the sensitive ones
        self.X_df = X_df_all.iloc[:, valid_idxs]

        self.columns_map = {}
        encoded_columns = self.X_df.columns.values.tolist()
        for c in num_cols + cat_cols:
            self.columns_map[c] = [ encoded_columns.index(e_c) for e_c in encoded_columns if e_c == c or e_c.startswith(c + '_') ]

    def get_num_cat_columns_sorted(self, X_df, sensitive_features):
        num_cols = []
        cat_cols = []

        sens_num_cols = []
        sens_cat_cols = []

        for c in X_df.columns:
            if c in sensitive_features:
                if X_df[c].dtype == 'object' or X_df[c].dtype.name == 'category':
                    sens_cat_cols.append(c)
                else:
                    sens_num_cols.append(c)
            else:
                if X_df[c].dtype == 'object' or X_df[c].dtype.name == 'category':
                    cat_cols.append(c)
                else:
                    num_cols.append(c)

        num_cols.sort()
        cat_cols.sort()
        sens_num_cols.sort()
        sens_cat_cols.sort()

        return num_cols, cat_cols, sens_num_cols, sens_cat_cols

    def scale_num_cols(self, X_df, num_cols):
        """
        X_df: features dataframe
        num_cols: name of all numerical columns to be scaled
        returns: feature dataframe with scaled numerical features
        """
        X_df_scaled = X_df.copy()
        scaler = MinMaxScaler()
        X_num = scaler.fit_transform(X_df_scaled[num_cols])

        for i, c in enumerate(num_cols):
            X_df_scaled[c] = X_num[:,i]

        return X_df_scaled

    def process_num_cat_columns(self, X_df, drop_first):
        """
        X_df: features dataframe
        returns: feature dataframe with scaled numerical features and one-hot encoded categorical features
        """
        num_cols = []
        cat_cols = []

        for c in X_df.columns:
            if X_df[c].dtype == 'object' or X_df[c].dtype.name == 'category':
                cat_cols.append(c)
            else:
                num_cols.append(c)

        # TODO: need to think about this drop_first
        X_df_encoded = pd.get_dummies(X_df, columns=cat_cols, drop_first=drop_first)

        cat_cols = list(set(X_df_encoded.columns) - set(num_cols))

        num_cols.sort()
        cat_cols.sort()

        X_df_encoded_scaled = self.scale_num_cols(X_df_encoded, num_cols)

        return X_df_encoded_scaled[num_cols + cat_cols]


    def process_labels(self, X_df, y_df, drop_first):
        X_processed = X_df.copy()
        if isinstance(y_df, str):
            prefix = y_df
            y_columns = [ c for c in X_processed.columns if c == prefix or c.startswith(prefix + '_') ]
            y_processed = X_df[y_columns]
            X_processed.drop(columns=y_columns, inplace=True)
        else:
            y_processed = pd.get_dummies(y_df, drop_first=drop_first)

        return X_processed, y_processed


    def prepare_dataset(self, X_df_original, y_df_original, drop_first, drop_first_labels, drop_columns=[]):
        """
        X_df_original: features dataframe
        y_df_original: labels dataframe
        returns:
            - feature dataframe with scaled numerical features and one-hot encoded categorical features
            - one hot encoded labels, with drop_first option
        """
        X_df = X_df_original.copy()

        X_processed = self.process_num_cat_columns(X_df, drop_first)

        X_processed, y_processed = self.process_labels(X_processed, y_df_original, drop_first_labels)

        return X_processed, y_processed
    
# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov,
#     Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th,
#     7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
#     Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
#     Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving,
#     Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany,
#     Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras,
#     Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France,
#     Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua,
#     Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
def get_adult_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    """
    train_path: path to training data
    test_path: path to test data
    returns: tuple of training features, training labels, test features and test labels
    """
    train_df = pd.read_csv(DATA_ADULT_TRAIN, na_values='?').dropna()
    print("Shape 1: ", train_df.shape)
    test_df = pd.read_csv(DATA_ADULT_TEST, na_values='?').dropna()
    print("Shape 2: ", test_df.shape)
    merged = pd.concat([train_df, test_df], axis=0)
    print("Shape 3: ", merged.shape)
    target = 'target'

    X_train = train_df.drop(columns=[target])
    y_train = train_df[[target]]
    X_test = merged.drop(columns=[target]) # test_df.drop(columns=[target])
    y_test = merged[[target]]
    #  ^ This is incorrect. Train data should not be in test set.

    train_ds = TabularDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = TabularDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds



# CREDIT DATASET:
# This research employed a binary variable, default payment (Yes = 1, No = 0), as the response
# variable. This study reviewed the literature and used the following 23 variables as explanatory
# variables:
#     x1: Amount of the given credit (NT dollar): it includes both the individual consumer
#         credit and his/her family (supplementary) credit.
#     x2: Gender (1 = male; 2 = female).
#     x3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
#     x4: Marital status (1 = married; 2 = single; 3 = others).
#     x5: Age (year).
#     x6 - x11: History of past payment. We tracked the past monthly payment records (from April to
#         September, 2005) as follows: x6 = the repayment status in September, 2005; x7 = the
#         repayment status in August, 2005; . . .;x11 = the repayment status in April, 2005. The
#         measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one
#         month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months;
#         9 = payment delay for nine months and above.
#     x12-x17: Amount of bill statement (NT dollar). x12 = amount of bill statement in September,
#         2005; x13 = amount of bill statement in August, 2005; . . .; x17 = amount of bill
#         statement in April, 2005.
#     x18-x23: Amount of previous payment (NT dollar). x18 = amount paid in September, 2005;
#         x19 = amount paid in August, 2005; . . .;x23 = amount paid in April, 2005.
def get_credit_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    """
    sensitive_features: features that should be considered sensitive when building the
        BiasedDataset object
    drop_columns: columns we can ignore and drop
    random_state: to pass to train_test_split
    return: two BiasedDataset objects, for training and test data respectively
    """
    credit_data = fetch_openml(data_id=42477, as_frame=True, data_home='./data/raw')

    # Force categorical data do be dtype: category
    features = credit_data.data
    categorical_features = ['x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']
    for cf in categorical_features:
        features[cf] = features[cf].astype(str).astype('category')

    # Encode output
    target = (credit_data.target == "1") * 1
    target = pd.DataFrame({'target': target})

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state)

    train_ds = TabularDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = TabularDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds



def get_data(data_df, target, sensitive_features, drop_columns=[], test_size=0.2, random_state=42):

    X = data_df.drop(columns=[target])
    y = data_df[[target]]

    return TabularDataset(
        X, y, sensitive_features=sensitive_features, drop_columns=drop_columns)
    # We split train/test outside, otherwise the number of categories may differ

def get_german_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    data_df = pd.read_csv(DATA_GERMAN_FILENAME, na_values='?').dropna()

    train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state)
    target = 'target'

    X_train = train_df.drop(columns=[target])
    y_train = train_df[[target]]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[[target]]

    train_ds = TabularDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = TabularDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds



def get_UCI_dataset(dataset, data_df, target, sensitive_features):
    ds = get_data(data_df, target, sensitive_features)
    ds.X_df = pd.concat([ds.X_df] + ds.sensitive_dfs, axis=1)
    # adding one-hot values, e.g., column-0 + column-2 can cause mistakes.

    X_df_train, X_df_test, y_df_train, y_df_test = train_test_split(ds.X_df, ds.y_df, test_size=0.2, random_state=42)

    X_train = X_df_train.to_numpy(); y_train = y_df_train.to_numpy()
    X_val = copy.deepcopy(X_df_test.to_numpy()); y_val = y_df_test.to_numpy()
    X_test = copy.deepcopy(X_df_test.to_numpy()); y_test = y_df_test.to_numpy()
    # ^ Although we don't really want this duplicate val/test sets, we leave it as it is.

    print("DATA SHAPES: ")
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print("****")
    
    lp_epsilon = FairCertModule.get_fairness_intervals(X_train, [-1],  metric="LP", use_sens=False, eps=1)
    lp_epsilon /= sum(lp_epsilon)
    lp_epsilon /= np.median(lp_epsilon)
    sr_epsilon = FairCertModule.get_fairness_intervals(X_train, [-1],  metric="SENSR", use_sens=False, eps=1)
    sr_epsilon /= sum(sr_epsilon)
    sr_epsilon /= np.median(sr_epsilon)
    
    X_train = X_train[:,:-1]
    X_test = X_test[:,:-1]
    X_val = X_val[:,:-1]
    
    y_train = torch.Tensor(np.squeeze(y_train)).long()
    y_test = torch.Tensor(np.squeeze(y_test)).long()
    y_val = torch.Tensor(np.squeeze(y_val)).long()
    
    return X_train, X_test, X_val, y_train, y_test, y_val, lp_epsilon, sr_epsilon
    