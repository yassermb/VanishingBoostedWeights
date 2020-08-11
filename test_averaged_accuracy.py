import numpy as np
import pandas
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification,make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import seaborn as sns

import lightgbm
from catboost import CatBoostClassifier

import warnings
import time
import os
warnings.filterwarnings("ignore")



class my_LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=False, verbose=False, regul=1e-1):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.regul = regul
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y, theta):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        self.theta = theta #np.zeros(X.shape[1])
        ind = np.where(self.theta == 0)[0]
    
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta[ind] -= self.lr * gradient[ind] + self.regul*self.theta[ind]
        
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()
    
    

def do_expr(X_all, y_all, name, report_dict):
    start_time = time.time()
    print('Processing dataset: ' + name)
    NMC = 5
    #NEstimators = [1, 2, 3, 5, 7, 10, 20, 50, 100]
    NEstimators =  [1, 5, 10, 25, 50, 75, 100, 500, 1000, 5000]
    #NEstimators = [5, 10]
    
    acc_GB_train = np.zeros((NMC,len(NEstimators)))
    acc_light_train = np.zeros((NMC,len(NEstimators)))
    acc_cat_train = np.zeros((NMC,len(NEstimators)))
    acc_goss_train = np.zeros((NMC,len(NEstimators)))
    acc_OurGB_train = np.zeros((NMC,len(NEstimators)))
    acc_LR_train = np.zeros((NMC,len(NEstimators)))
    
    acc_GB_test = np.zeros((NMC,len(NEstimators)))
    acc_light_test = np.zeros((NMC,len(NEstimators)))
    acc_cat_test = np.zeros((NMC,len(NEstimators)))
    acc_goss_test = np.zeros((NMC,len(NEstimators)))
    acc_OurGB_test = np.zeros((NMC,len(NEstimators)))
    acc_LR_test = np.zeros((NMC,len(NEstimators)))
    
    for iMC in range(NMC):
            
        #ok
        #X_all, y_all = make_classification(n_samples=100, n_features=30, n_redundant=0, n_informative=2, n_clusters_per_class=1,flip_y=0.1)
        
        #ok
        #X_all, y_all = make_blobs(n_samples=100, n_features=50, cluster_std=10.0, centers=2)
        
        #X_all, y_all =  make_moons(n_samples=100, shuffle=True, noise=0.4, random_state=None)
        
        #X_all, y_all = make_classification(n_samples=500, n_features=200, n_redundant=30, n_informative=20, n_clusters_per_class=1,flip_y=0.3, shift = 0.)
    
        X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.3)
    
        for iNEst in range(len(NEstimators)):
    
            n_est = NEstimators[iNEst]
            #print(n_est)
            
            clfGB = GradientBoostingClassifier(n_estimators=n_est, max_depth=1)
            clfGB1 = GradientBoostingClassifier(n_estimators=n_est, max_depth=1)
            clfGB.fit(X_all,y_all)
            clfGB1.fit(X,y)
            
            # sklearn
            acc_GB_train[iMC,iNEst] =clfGB1.score(X, y)
            acc_GB_test[iMC,iNEst] =clfGB1.score(X_test, y_test)
            
            # light
            train_data = lightgbm.Dataset(X, label=y)
            test_data  = lightgbm.Dataset(X_test, label=y_test)
            parameters = {
                'application': 'binary',
                'metric': 'binary_logloss',
                'n_estimators': n_est,
                'boosting': 'gbdt',
                'num_leaves': 2,
                'learning_rate': 0.05,
                'verbose': -1,
                'num_iterations':20
            }
            model = lightgbm.train(parameters,
                                   train_data,
                                   valid_sets=test_data)
            preds = model.predict(X)
            y_pred = np.zeros((X.shape[0]))
            y_pred[preds>=.5] = 1
            acc_light_train[iMC,iNEst] = ((y == y_pred).mean())
            preds = model.predict(X_test)
            y_pred = np.zeros((X_test.shape[0]))
            y_pred[preds>=.5] = 1
            acc_light_test[iMC,iNEst] = ((y_pred == y_test).mean())
            
            # GOSS
            train_data = lightgbm.Dataset(X, label=y)
            test_data  = lightgbm.Dataset(X_test, label=y_test)
            parameters = {
                'application': 'binary',
                'metric': 'binary_logloss',
                'n_estimators': n_est,
                'boosting': 'goss',
                'num_leaves': 2,
                'learning_rate': 0.05,
                'verbose': -1,
                'num_iterations':20
            }
            model = lightgbm.train(parameters,
                                   train_data,
                                   valid_sets=test_data)
            preds = model.predict(X)
            y_pred = np.zeros((X.shape[0]))
            y_pred[preds>=.5] = 1
            acc_goss_train[iMC,iNEst] = ((y == y_pred).mean())
            preds = model.predict(X_test)
            y_pred = np.zeros((X_test.shape[0]))
            y_pred[preds>=.5] = 1
            acc_goss_test[iMC,iNEst] = ((y_pred == y_test).mean())
            
            # CatBoostClassifier
            model = CatBoostClassifier(learning_rate=0.05,
                                    eval_metric='Accuracy',n_estimators=n_est,max_depth=1,verbose=False)
            model.fit(X,y)
            preds = model.predict(X)
            y_pred = np.zeros((X.shape[0]))
            y_pred[preds>=.5] = 1
            acc_cat_train[iMC,iNEst] = ((y == y_pred).mean())
            preds = model.predict(X_test)
            y_pred = np.zeros((X_test.shape[0]))
            y_pred[preds>=.5] = 1
            acc_cat_test[iMC,iNEst] = ((y_pred == y_test).mean())
    
            
            D = set()
    
            n_classes, n_estimators = clfGB.estimators_.shape
            for c in range(n_classes):
                for t in range(n_estimators):
                    dtree = clfGB.estimators_[c, t]
       
                    rules = pandas.DataFrame({
                        'child_left': dtree.tree_.children_left,
                        'child_right': dtree.tree_.children_right,
                        'feature': dtree.tree_.feature,
                        'threshold': dtree.tree_.threshold,
                    })
                    tup = (rules.iloc[0,2],rules.iloc[0,3])
                    D.add(tup)
        
            first = 1
            for e in D:
                if (first):
                    feature, thres = e
                    tmp = X[:,feature] > thres
                    tmp_test = X_test[:,feature] > thres
                    dummy =pandas.get_dummies(tmp)
                    if (dummy.shape[1] == 1) and (dummy.iloc[1]==1).bool():
                        dummy = np.concatenate((dummy, 1-dummy), axis=1)
                    if (dummy.shape[1] == 1) and (dummy.iloc[1]==0).bool():
                        dummy = np.concatenate((dummy, 1+dummy), axis=1)
                    dummy_test =pandas.get_dummies(tmp_test)
                    if (dummy_test.shape[1] == 1) and (dummy_test.iloc[1]==1).bool():
                        dummy_test = np.concatenate((dummy_test, 1-dummy_test), axis=1)
                    if (dummy_test.shape[1] == 1) and (dummy_test.iloc[1]==0).bool():
                        dummy_test = np.concatenate((dummy_test, 1+dummy_test), axis=1)
                    data_discr = dummy
                    data_discr_test = dummy_test
                    first = 0
                else:
                    feature, thres = e
                    tmp = X[:,feature] > thres
                    dummy =pandas.get_dummies(tmp)
                    if (dummy.shape[1] == 1) and (dummy.iloc[1]==1).bool():
                        dummy = np.concatenate((dummy, 1-dummy), axis=1)
                    if (dummy.shape[1] == 1) and (dummy.iloc[1]==0).bool():
                        dummy = np.concatenate((dummy, 1+dummy), axis=1)
                    data_discr = np.concatenate((data_discr, dummy), axis=1)
                    tmp_test = X_test[:,feature] > thres
                    dummy_test =pandas.get_dummies(tmp_test)
                    if (dummy_test.shape[1] == 1) and (dummy_test.iloc[1]==1).bool():
                        dummy_test = np.concatenate((dummy_test, 1-dummy_test), axis=1)
                    if (dummy_test.shape[1] == 1) and (dummy_test.iloc[1]==0).bool():
                        dummy_test = np.concatenate((dummy_test, 1+dummy_test), axis=1)
                    data_discr_test = np.concatenate((data_discr_test, dummy_test), axis=1)
            #print(data_discr.shape[1])
            #print(data_discr_test.shape[1])
            
            T = 50
            for t in range(0,T):
                n = X.shape[0]//10
                ind =np.random.permutation(X.shape[0])
                ind_to_use = ind[0:n]
                logreg = LogisticRegression(C=1e1)
                logreg.fit(np.array(data_discr)[ind_to_use], y[ind_to_use])
                acc_LR_train[iMC,iNEst] =logreg.score(np.array(data_discr)[ind_to_use], y[ind_to_use])
                if t == 0:
                    theta_tmp = logreg.coef_
                else:
                    theta_tmp = theta_tmp + logreg.coef_
            
            logreg.coef_ = theta_tmp/T
            acc_LR_test[iMC,iNEst] =logreg.score(data_discr_test, y_test)
            acc_LR_train[iMC,iNEst] =logreg.score(data_discr, y)
            
    
    report_dict[name] = time.time() - start_time
    
    
    np.save('Accuracy_averaged_' + name + '_GB_train', acc_GB_train)
    np.save('Accuracy_averaged_' + name + '_GB_test', acc_GB_test)
    np.save('Accuracy_averaged_' + name + '_LR_train', acc_LR_train)
    np.save('Accuracy_averaged_' + name + '_LR_test', acc_LR_test)   
    np.save('Accuracy_averaged_' + name + '_light_train', acc_light_train)
    np.save('Accuracy_averaged_' + name + '_light_test', acc_light_test)
    np.save('Accuracy_averaged_' + name + '_cat_train', acc_cat_train)
    np.save('Accuracy_averaged_' + name + '_cat_test', acc_cat_test)
    np.save('Accuracy_averaged_' + name + '_goss_train', acc_goss_train)
    np.save('Accuracy_averaged_' + name + '_goss_test', acc_goss_test)
    
    # plot Accuracy
    fig=plt.figure()
    ax=fig.add_subplot(111)
                         
    ax.errorbar(range(len(NEstimators)), acc_GB_train.mean(0), acc_GB_train.std(0), linestyle='--', c='b', marker='_',label='GBoost train')
    ax.errorbar(range(len(NEstimators)), acc_GB_test.mean(0), acc_GB_test.std(0), linestyle='--', c='b', marker="D",label='GBoost test')
    
    ax.errorbar(range(len(NEstimators)), acc_light_train.mean(0), acc_light_train.std(0), linestyle='--', c='g', marker='_',label='LightGBM train')
    ax.errorbar(range(len(NEstimators)), acc_light_test.mean(0), acc_light_test.std(0), linestyle='--', c='g', marker="D",label='LightGBM test')
    
    ax.errorbar(range(len(NEstimators)), acc_cat_train.mean(0), acc_cat_train.std(0), linestyle='--', c='r', marker='_',label='Cat train')
    ax.errorbar(range(len(NEstimators)), acc_cat_test.mean(0), acc_cat_test.std(0), linestyle='--', c='r', marker="D",label='Cat test')
    
    ax.errorbar(range(len(NEstimators)), acc_goss_train.mean(0), acc_goss_train.std(0), linestyle='--', c='y', marker='_',label='GOSS train')
    ax.errorbar(range(len(NEstimators)), acc_goss_test.mean(0), acc_goss_test.std(0), linestyle='--', c='y', marker="D",label='GOSS test')
    
    ax.errorbar(range(len(NEstimators)), acc_LR_train.mean(0), acc_LR_train.std(0), linestyle='--', c='m', marker='_',label='Averaged train')
    ax.errorbar(range(len(NEstimators)), acc_LR_test.mean(0), acc_LR_test.std(0), linestyle='--', c='m', marker="D",label='Averaged test')
    
    plt.xticks(ticks=range(len(NEstimators)), labels=NEstimators)
    plt.xlabel('Nb estimators')
    plt.ylabel('Accuracy')
                         
    plt.legend(loc=0)
    plt.savefig('Accuracy_averaged_' + name + '.png')


use_multiprocessing = True
db_path = 'Rudin'
db_cases = []
for db_name in os.listdir(db_path):
    db_file = os.path.join(db_path, db_name)
    db_df = pandas.read_table(db_file, sep = ' ', error_bad_lines=False, header = None).sample(frac=1)
    y_all = (db_df.iloc[:,-1].to_numpy() + 1) // 2
    X_all = db_df.drop(db_df.columns[-1],axis=1).to_numpy()    
    
    db_cases.append((X_all, y_all, db_name.replace('.txt','')))

if use_multiprocessing:
    import multiprocessing
    max_cpus = 30
    manager = multiprocessing.Manager()
    report_dict = manager.dict()
    pool = multiprocessing.Pool(processes = min(max_cpus, multiprocessing.cpu_count()))
else:
    report_dict = dict()

for args in db_cases:
    args += (report_dict,)
    if use_multiprocessing:
        pool.apply_async(do_expr, args = args)
    else:
        do_expr(*args)

if use_multiprocessing:
    pool.close()
    pool.join()
    
print(dict(report_dict))