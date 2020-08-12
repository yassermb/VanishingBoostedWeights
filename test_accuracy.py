import numpy as np
import pandas
import matplotlib.pyplot as plt
import random
import os
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
from sklearn.exceptions import ConvergenceWarning
import seaborn as sns
import warnings
import time
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
    
        self.theta = theta
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
    NEstimators =  [1, 5, 10, 25, 50, 75, 100, 500, 1000, 5000, 7000, 10000, 15000, 30000, 50000]
    
    acc_GB_train = np.zeros((NMC,len(NEstimators)))
    acc_OurGB_train = np.zeros((NMC,len(NEstimators)))
    acc_LR_train = np.zeros((NMC,len(NEstimators)))
    
    acc_GB_test = np.zeros((NMC,len(NEstimators)))
    acc_OurGB_test = np.zeros((NMC,len(NEstimators)))
    acc_LR_test = np.zeros((NMC,len(NEstimators)))
        
    for iMC in range(NMC):
        X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.3)
    
        for iNEst in range(len(NEstimators)):
    
            n_est = NEstimators[iNEst]
            
            clfGB = GradientBoostingClassifier(n_estimators=n_est, max_depth=1)
            clfGB1 = GradientBoostingClassifier(n_estimators=n_est, max_depth=1)
            clfGB.fit(X_all,y_all)
            clfGB1.fit(X,y)
            acc_GB_train[iMC,iNEst] =clfGB1.score(X, y)
            acc_GB_test[iMC,iNEst] =clfGB1.score(X_test, y_test)
    
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
           
            logreg = LogisticRegression(C=1e1)
            logreg.fit(data_discr, y)
            acc_LR_train[iMC,iNEst] =logreg.score(data_discr, y)
            acc_LR_test[iMC,iNEst] =logreg.score(data_discr_test, y_test)
            
            used_n_est = len(D)
    
            T = 20*used_n_est
            my_clf = my_LogisticRegression(lr=0.1, num_iter=20)
            for t in range(T):
                e = random.sample(D,1)
                feature, thres = e[0]
                tmp = X[:,feature] > thres
                dummy = pandas.get_dummies(tmp)
                if (dummy.shape[1] == 1) and (dummy.iloc[1]==1).bool():
                    dummy = np.concatenate((dummy, 1-dummy), axis=1)
                if (dummy.shape[1] == 1) and (dummy.iloc[1]==0).bool():
                    dummy = np.concatenate((dummy, 1+dummy), axis=1)
                data_discr = np.concatenate((data_discr, dummy), axis=1)
    
                tmp_test = X_test[:,feature] > thres
                dummy_test = pandas.get_dummies(tmp_test)
                if (dummy_test.shape[1] == 1) and (dummy_test.iloc[1]==1).bool():
                    dummy_test = np.concatenate((dummy_test, 1-dummy_test), axis=1)
                if (dummy_test.shape[1] == 1) and (dummy_test.iloc[1]==0).bool():
                    dummy_test = np.concatenate((dummy_test, 1+dummy_test), axis=1)
                data_discr_test = np.concatenate((data_discr_test, dummy_test), axis=1)
    
    
                if t==0:
                    theta= np.zeros(data_discr.shape[1])
                    my_clf.fit(data_discr,y,theta)
                if t > 0:
                    theta_warm = np.zeros(data_discr.shape[1])
                    theta_warm[0:2*used_n_est+2*t] = my_clf.theta
                    theta = theta_warm
                    my_clf.fit(data_discr,y,theta)
                    
                preds = my_clf.predict(data_discr_test)
                acc_OurGB_test[iMC,iNEst] = ((preds == y_test).mean())
                
                preds = my_clf.predict(data_discr)
                acc_OurGB_train[iMC,iNEst] = ((preds == y).mean())
    
    report_dict[name] = time.time() - start_time
    
    np.save('Accuracy_' + name + '_GB_train', acc_GB_train)
    np.save('Accuracy_' + name + '_GB_test', acc_GB_test)
    np.save('Accuracy_' + name + '_LR_train', acc_LR_train)
    np.save('Accuracy_' + name + '_LR_test', acc_LR_test)   
    np.save('Accuracy_' + name + '_ourGB_train', acc_OurGB_train)
    np.save('Accuracy_' + name + '_ourGB_test', acc_OurGB_test)
    
    # plot Accuracy
    fig=plt.figure()
    ax=fig.add_subplot(111)
                         
    ax.errorbar(range(len(NEstimators)), acc_GB_train.mean(0), acc_GB_train.std(0), linestyle='--', c='b', marker='_',label='GBoost train')
    ax.errorbar(range(len(NEstimators)), acc_GB_test.mean(0), acc_GB_test.std(0), linestyle='--', c='b', marker="D",label='GBoost test')
    
    ax.errorbar(range(len(NEstimators)), acc_LR_train.mean(0), acc_LR_train.std(0), linestyle='--', c='g', marker='_',label='LogReg train')
    ax.errorbar(range(len(NEstimators)), acc_LR_test.mean(0), acc_LR_test.std(0), linestyle='--', c='g', marker="D",label='LogReg test')
    
    ax.errorbar(range(len(NEstimators)), acc_OurGB_train.mean(0), acc_OurGB_train.std(0), linestyle='--', c='r', marker='_',label='VBW train')
    ax.errorbar(range(len(NEstimators)), acc_OurGB_test.mean(0), acc_OurGB_test.std(0), linestyle='--', c='r', marker="D",label='VBW  test')
    
    plt.xticks(ticks=range(len(NEstimators)), labels=NEstimators)
    plt.xlabel('Nb estimators')
    plt.ylabel('Accuracy')
                         
    plt.legend(loc=4)
    plt.savefig('Accuracy_' + name + '.png')


use_multiprocessing = True
db_path = 'Data'
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
