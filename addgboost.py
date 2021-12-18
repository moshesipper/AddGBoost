# AddGBoost
# copyright 2021 moshe sipper  
# www.moshesipper.com 

from string import ascii_lowercase
from random import choices
from sys import stdin
from pandas import read_csv
from statistics import median, mean
from mlxtend.evaluate import permutation_test
from pathlib import Path
from operator import itemgetter
from time import process_time
from argparse import ArgumentParser
from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_openml, make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split #GridSearchCV# #cross_val_score
from sklearn.datasets import load_boston, load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoLars, SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.experimental import enable_hist_gradient_boosting  # noqa  needed for HistGradientBoostingRegressor to work
from sklearn.ensemble import HistGradientBoostingRegressor#GradientBoostingRegressor, RandomForestRegressor, 
from sklearn.preprocessing import StandardScaler
import warnings
from pmlb import fetch_data, regression_dataset_names
from lightgbm import LGBMRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)#optuna.logging.WARNING)


MLAlgParams={ # 'categorical'/'int'/'float': for use by optuna
KernelRidge: [ ['categorical', 'kernel', ['linear', 'poly', 'rbf', 'sigmoid'] ],
                ['float', 'alpha', 1e-4, 1, 'log'], 
                ['float', 'gamma', 0.01, 10, 'log'] ],
LassoLars: [ ['float', 'alpha', 1e-04, 1, 'log'] ],       
SGDRegressor: [ ['float', 'alpha', 1e-05, 1, 'log'],
                ['categorical', 'penalty', ['l2', 'l1', 'elasticnet'] ] ],
LinearSVR: [ ['float', 'C', 1e-05, 1, 'log'],
              ['categorical', 'loss', ['epsilon_insensitive', 'squared_epsilon_insensitive'] ] ],
DecisionTreeRegressor: [['categorical', 'criterion', ['mse', 'friedman_mse', 'mae'] ],
                        ['categorical', 'splitter', ['best', 'random'] ],
                        ['categorical', 'max_features', ['sqrt','log2', None] ],
                        ['int', 'min_samples_leaf', 1, 5, 'nolog'] ],
HistGradientBoostingRegressor: [ ['int', 'max_iter', 10, 100, 'nolog'],
                ['float', 'learning_rate', 0.01, 0.3, 'nolog'],
                ['categorical', 'loss', ['least_squares', 'least_absolute_deviation'] ] ],
LGBMRegressor: [ ['float', 'lambda_l1', 1e-8, 10.0, 'log'],
                 ['float', 'lambda_l2', 1e-8, 10.0, 'log'],
                 ['int', 'num_leaves', 2, 256, 'nolog'] ] 
}
MLAlgs = list(MLAlgParams.keys())
MLAlgNames = [alg.__name__ for alg in MLAlgs]


def scoring(y, y_pred):
    return mean_absolute_error(y, np.nan_to_num(y_pred))

def rand_str(n): 
    return ''.join(choices(ascii_lowercase, k=n))

def fprint(fname, s):
    if stdin.isatty(): print(s) # running interactively 
    with open(Path(fname),'a') as f: f.write(s)

def save_params(fname, dsname, n_stages, n_trials, n_replicates, n_samples, n_features):
    fprint(fname, f' dsname: {dsname}\n n_samples: {n_samples}\n n_features: {n_features:}\n n_stages: {n_stages}\n n_trials: {n_trials} \n n_replicates: {n_replicates} \n\n')

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-resdir', dest='resdir', type=str, action='store', help='directory where results are placed')
    parser.add_argument('-dsname', dest='dsname', type=str, action='store', help='dataset name')
    parser.add_argument('-stages', dest='n_stages', type=int, action='store', help='number of AddGBoost stages')
    parser.add_argument('-trials', dest='n_trials', type=int, action='store', help='number of Optuna trials')
    parser.add_argument('-nrep', dest='n_replicates', type=int, action='store', help='number of replicate runs')
    args = parser.parse_args()
    if None in [getattr(args, arg) for arg in vars(args)]:
        parser.print_help()
        exit()
    resdir, dsname, n_stages, n_trials, n_replicates = args.resdir+'/', args.dsname, args.n_stages, args.n_trials, args.n_replicates                      
    # if not exists(resdir): makedirs(resdir)
    fname = resdir + dsname + '_' + rand_str(6) + '.txt'
    return fname, resdir, dsname, n_stages, n_trials, n_replicates

def get_dataset(dsname):
    version, openml = -1, False
    if dsname ==  'regtest':
        X, y = make_regression(n_samples=10, n_features=2, n_informative=1)
    elif dsname == 'boston':
        X, y = load_boston(return_X_y=True)
    elif dsname == 'diabetes':
        X, y = load_diabetes(return_X_y=True)
    elif dsname in regression_dataset_names: # PMLB datasets
        X, y = fetch_data(dsname, return_X_y=True, local_cache_dir='../datasets/pmlbreg/')
    else:
        try: # dataset from openml?
            data = fetch_openml(data_id=int(dsname), cache=True, data_home='../datasets/scikit_learn_data')
            X, y = data['data'], data['target']
            dsname = data['details']['name']
            version = data['details']['version']
            openml = True
        except:
            try: # a csv file in datasets folder?
                data = read_csv('../datasets/' + dsname + '.csv', sep=',')
                array = data.values
                X, y = array[:,0:-1], array[:,-1] # target is last col
                # X, y = array[:,1:], array[:,0] # target is 1st col
            except Exception as e: 
                print('looks like there is no such dataset')
                exit(e)
                
    n_samples, n_features = X.shape
    return X, y, n_samples, n_features, dsname, version, openml

class AddGBoost(BaseEstimator):
    def __init__(self, models=None):
        self.models = models
     
    def fit(self, X, y):
        y_res = deepcopy(y)
        for model in self.models: 
            model.fit(X, y_res)
            p = model.predict(X)
            y_res -= np.nan_to_num(p)
        return self

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for model in self.models: 
            pred += np.nan_to_num(model.predict(X))
        return np.nan_to_num(pred)
# end class AddGBoost

class Objective(object):
    def __init__(self, alg, X, y, n_stages=-1):
        self.alg = alg
        self.X = X
        self.y = y
        self.n_stages = n_stages

    def create_single_model(self, trial, alg, stage=-1):
        s = '' if stage==-1 else f'_{stage}'
        kwargs = {}
        for param in MLAlgParams[alg]:
            param_name = f'{alg.__name__}_{param[1]}{s}'
            if param[0] == 'categorical':
                p = trial.suggest_categorical(param_name, param[2])
            elif param[0] == 'int':
                p = trial.suggest_int(param_name, param[2], param[3], log=param[4]=='log')
            elif param[0] == 'float':
                p = trial.suggest_float(param_name, param[2], param[3], log=param[4]=='log')
            else:
                exit(f'create_model, unknown hyperparameter type: {param[0]}')
            kwargs.update({param[1]: p})
        model = alg(**kwargs)
        return model

    def create_model(self, trial):
        if self.alg==AddGBoost:
            models = []
            for stage in range(self.n_stages):
                name = trial.suggest_categorical(f'regressor_{stage}', MLAlgNames)
                alg = MLAlgs[MLAlgNames.index(name)]
                models.append(self.create_single_model(trial, alg, stage=stage))
            model = AddGBoost(models=models)
        else:
            model = self.create_single_model(trial, self.alg)
        
        return model

    def __call__(self, trial):
        model = self.create_model(trial)
        X, y = self.X, self.y 

        scores = []
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X):
            model.fit(X[train_index], y[train_index])            
            scores.append(scoring(y[test_index], model.predict(X[test_index])))
        
        final_score = mean(scores)
        
        model.fit(X, y)
        trial.set_user_attr(key='best_model', value=model)

        return final_score
# end class Objective

def optuna_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key='best_model', value=trial.user_attrs['best_model'])

# main 
def main():
    start_time = process_time()
    fname, resdir, dsname_id, n_stages, n_trials, n_replicates = get_args()
    X, y, n_samples, n_features, dsname, version, openml = get_dataset(dsname_id)
    print_ds = f'{dsname} ({dsname_id})' if openml else f'{dsname}' # openml datasets given as ints, get_dataset converts to string
    save_params(fname, print_ds, n_stages, n_trials, n_replicates, n_samples, n_features)
      
    allreps = dict.fromkeys(MLAlgs + [AddGBoost]) # for recording scores and params across all replicates
    for k in allreps: 
        allreps[k] = {'test_scores': [], 'params': []}
    
    for rep in range(1, n_replicates+1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        sc = StandardScaler() 
        X_train = sc.fit_transform(X_train) # scaled data has mean 0 and variance 1 (only over training set)
        X_test = sc.transform(X_test) # use same scaler as one fitted to training data

        for alg in MLAlgs + [AddGBoost]:
            # run Optuna
            objective = Objective(alg, X_train, y_train, n_stages=n_stages)
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials, callbacks=[optuna_callback])
            # get Optuna's best results
            best_model = study.user_attrs['best_model']
            allreps[alg]['params'].append(study.best_trial.params)
            # compute test score
            test_score = scoring(y_test, best_model.predict(X_test))
            allreps[alg]['test_scores'].append(test_score)
            fprint(fname, f'rep {rep}, {alg.__name__}, {round(test_score,3)}, {allreps[alg]["params"][-1]}\n')
        
    # done all replicate runs, compute and report final summary of experiment's stats  
    fprint(fname, f"\n*Summary of experiment's results over {n_replicates} replicates: \n")
    
    medians = [ [alg, median(allreps[alg]['test_scores'])] for alg in MLAlgs + [AddGBoost]]
    medians = sorted(medians, key=itemgetter(1), reverse=False)

    # 10,000-round permutation test to assess statistical significance of diff in test scores between 1st and 2nd places
    pval = permutation_test(allreps[medians[0][0]]['test_scores'], allreps[medians[1][0]]['test_scores'], method='approximate', num_rounds=10_000,\
                            func=lambda x, y: np.abs(np.median(x) - np.median(y))) 
    
    pp = ''
    if medians[0][0] == AddGBoost and pval<0.05: # ranked first, statistically significant
        pp = '!'
    elif medians[1][0] == AddGBoost and pval>=0.05: # ranked second, statistically insignificant
        pp = '=='

    s_res = f'*>> {dsname_id}, '
    for i, m in enumerate(medians): 
        s_res += f'#{i+1}: {m[0].__name__} {round(m[1],4)}'
        if m[0]==AddGBoost:
            s_res += pp
        s_res += ', '

    fprint(fname, s_res[:-2] + '\n')
    
    runtime = process_time() - start_time
    fprint(fname, f'*runtime {runtime}\n')

##############        
if __name__== "__main__":
  warnings.filterwarnings("ignore")
  main()

