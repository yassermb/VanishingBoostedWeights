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
from sklearn import mixture
from sklearn.cluster import KMeans
from random import randrange
import seaborn as sns
import warnings
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

    def fit(self, X, y, theta,weights=1):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        self.theta = theta #np.zeros(X.shape[1])
        ind = np.where(self.theta == 0)[0]
    
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(weights*X.T, (h - y)) / y.size
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
        
def weighted_predict(X, theta):
    z = np.dot(X, theta)
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid.round()
      
#def Our_Discretize(n_est, X_train, y_train, X_test, y_test):
def Our_Discretize(n_est, X_all, y_all):
     
    X = X_all
    X_test = X_all
    #y = y_train
    #X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.2)
    
    #X_all = np.concatenate((X_train, X_test), axis=0)
    #y_all = np.concatenate((y_train, y_test), axis=0)
    
    clfGB = GradientBoostingClassifier(n_estimators=n_est, max_depth=1)
    clfGB.fit(X_all,y_all)

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
    used_n_est = len(D)

    #T = 15*used_n_est
    #my_clf = my_LogisticRegression(lr=0.1, num_iter=2)
    #for t in range(T):
    #    e = random.sample(D,1)
    #    feature, thres = e[0]
    #    tmp = X[:,feature] > thres
    #    dummy = pandas.get_dummies(tmp)
    #    if (dummy.shape[1] == 1) and (dummy.iloc[1]==1).bool():
    #        dummy = np.concatenate((dummy, 1-dummy), axis=1)
    #    if (dummy.shape[1] == 1) and (dummy.iloc[1]==0).bool():
    #        dummy = np.concatenate((dummy, 1+dummy), axis=1)
    #         #print(dummy.shape)
    #    data_discr = np.concatenate((data_discr, dummy), axis=1)

    #    tmp_test = X_test[:,feature] > thres
    #    dummy_test = pandas.get_dummies(tmp_test)
    #    if (dummy_test.shape[1] == 1) and (dummy_test.iloc[1]==1).bool():
    #        dummy_test = np.concatenate((dummy_test, 1-dummy_test), axis=1)
    #    if (dummy_test.shape[1] == 1) and (dummy_test.iloc[1]==0).bool():
    #        dummy_test = np.concatenate((dummy_test, 1+dummy_test), axis=1)
    #    data_discr_test = np.concatenate((data_discr_test, dummy_test), axis=1)

    #    if t==0:
    #        theta= np.zeros(data_discr.shape[1])
    #        my_clf.fit(data_discr,y,theta,weights)
    #    if t > 0:
    #        theta_warm = np.zeros(data_discr.shape[1])
    #        theta_warm[0:2*used_n_est+2*t] = my_clf.theta
    #        theta = theta_warm
    #        my_clf.fit(data_discr,y,theta,weights)
                 
    #    preds = my_clf.predict(data_discr_test)
    #    #acc_OurGB_test[iMC,iNEst] = ((preds == y_test).mean())
             
    #    preds = my_clf.predict(data_discr)
    #    #acc_OurGB_train[iMC,iNEst] = ((preds == y).mean())

    return [data_discr,data_discr_test]

NMC = 50
NEstimators = [1, 3, 7, 10, 20, 50, 100, 1000]
#NEstimators =  [1, 5, 10, 25, 50, 75, 100, 500, 1000, 5000, 7000, 10000, 15000, 30000, 50000]
#NEstimators = [5, 10]

#acc_GB_train = np.zeros((NMC,len(NEstimators)))
#acc_OurGB_train = np.zeros((NMC,len(NEstimators)))
#acc_LR_train = np.zeros((NMC,len(NEstimators)))

#acc_GB_test = np.zeros((NMC,len(NEstimators)))
#acc_OurGB_test = np.zeros((NMC,len(NEstimators)))
#cc_LR_test = np.zeros((NMC,len(NEstimators)))

acc_OurGB_test_1_2 = np.zeros((NMC,len(NEstimators)))
acc_OurGB_test_1_3 = np.zeros((NMC,len(NEstimators)))
acc_OurGB_test_2_1 = np.zeros((NMC,len(NEstimators)))
acc_OurGB_test_2_3 = np.zeros((NMC,len(NEstimators)))
acc_OurGB_test_3_1 = np.zeros((NMC,len(NEstimators)))
acc_OurGB_test_3_2 = np.zeros((NMC,len(NEstimators)))

acc_OurGB_test_1 = np.zeros((NMC,len(NEstimators)))
acc_OurGB_test_2 = np.zeros((NMC,len(NEstimators)))
acc_OurGB_test_3 = np.zeros((NMC,len(NEstimators)))


NPerBlob = 25

#centers1 = [[1, 1], [2, 0], [1, -1]]
#centers2 = [[-1, 1], [-1, -1], [-2, 0]]
centers1 = [[1, 1],[-1, 1]]
centers2 = [[2, 0],[-1, -1]]
centers3 = [[1, -1],[-2, 0]]

for iMC in range(NMC):
    
    #X_all1, y_all1 = make_blobs(n_samples=50, n_features=30, cluster_std=1.5, centers=centers1,shuffle=False)
    
    X_all1, y_all1 = make_blobs(n_samples=50, n_features=100, cluster_std=5, centers=2,shuffle=False)
    
    #X_all2, y_all2 = make_blobs(n_samples=50, n_features=30, cluster_std=1.5, centers=centers2,shuffle=False)
    
    X_all2, y_all2 = make_blobs(n_samples=50, n_features=100, cluster_std=5, centers=2,shuffle=False)
    
    #X_all3, y_all3 = make_blobs(n_samples=50, n_features=30, cluster_std=1.5, centers=centers3,shuffle=False)

    X_all3, y_all3 = make_blobs(n_samples=50, n_features=100, cluster_std=5, centers=2,shuffle=False)


    X_all = np.concatenate((X_all1, X_all2,X_all3), axis=0)
    y_all = np.concatenate((y_all1, y_all2, y_all3), axis=0)
    
    
    # train on 1, test on 2, 3
    
    # train on 2, test on 1, 3
    
    # train on 3, test on 1, 2
    
    # Federate
    
    
    #print(X_all.shape)
    #print(y_all.shape)
    for iNEst in range(len(NEstimators)):
    
        n_est = NEstimators[iNEst]
        # discretize data
        [data_discr, data_discr_test] = Our_Discretize(n_est, X_all, y_all)
        
        # train on 1
        my_clf = my_LogisticRegression(lr=0.1, num_iter=20)
        theta= np.zeros(data_discr.shape[1])
        #print(data_discr.shape)
        #print(data_discr.iloc[0:50,:].shape)
        my_clf.fit(data_discr[0:50],y_all1,theta)
        theta1 = my_clf.theta
        preds = my_clf.predict(data_discr_test[50:100])
        acc_OurGB_test_1_2[iMC,iNEst] = ((preds == y_all2).mean())
        preds = my_clf.predict(data_discr_test[100:150])
        acc_OurGB_test_1_3[iMC,iNEst] = ((preds == y_all3).mean())
             
        # train on 2
        my_clf = my_LogisticRegression(lr=0.1, num_iter=20)
        theta= np.zeros(data_discr.shape[1])
        my_clf.fit(data_discr[50:100],y_all2,theta)
        theta2 = my_clf.theta
        preds = my_clf.predict(data_discr_test[100:150])
        acc_OurGB_test_2_3[iMC,iNEst] = ((preds == y_all3).mean())
        preds = my_clf.predict(data_discr_test[0:50])
        acc_OurGB_test_2_1[iMC,iNEst] = ((preds == y_all1).mean())
                
        # train on 3, test on 1,
        my_clf = my_LogisticRegression(lr=0.1, num_iter=20)
        theta= np.zeros(data_discr.shape[1])
        my_clf.fit(data_discr[100:150],y_all3,theta)
        theta3 = my_clf.theta
        preds = my_clf.predict(data_discr_test[0:50])
        acc_OurGB_test_3_1[iMC,iNEst] = ((preds == y_all1).mean())
        preds = my_clf.predict(data_discr_test[50:100])
        acc_OurGB_test_3_2[iMC,iNEst] = ((preds == y_all2).mean())

        
        # estimate weights
        clf_KMeans_1 = KMeans(n_clusters=5).fit(X_all1)
        pred_1 = clf_KMeans_1.predict(X_all1)
        weights1 = np.zeros((X_all1.shape[0]))
        for j in range(X_all1.shape[0]):
            weights1[j] = (np.sum(pred_1[j] == pred_1))
        weights1 = weights1/np.sum(weights1)
        
        clf_KMeans_2 = KMeans(n_clusters=5).fit(X_all2)
        pred_2 = clf_KMeans_2.predict(X_all2)
        weights2 = np.zeros((X_all2.shape[0]))
        for j in range(X_all2.shape[0]):
            weights2[j] = (np.sum(pred_2[j] == pred_2))
        weights2 = weights2/np.sum(weights2)

        clf_KMeans_3 = KMeans(n_clusters=5).fit(X_all3)
        pred_3 = clf_KMeans_3.predict(X_all3)
        weights3 = np.zeros((X_all3.shape[0]))
        for j in range(X_all3.shape[0]):
            weights3[j] = (np.sum(pred_3[j] == pred_3))
        weights3 = weights3/np.sum(weights3)

        #theta = (theta1*weights2/weights1 + theta1*weight3/weights1 + theta2*weight1/weights2 + theta2*weight3/weights2+theta3*weights2/weights3 + theta3*weight1/weights3)/6
        
        theta = (theta1 + theta2 + theta3)/3
        
        preds = weighted_predict(data_discr_test[0:50], theta)
        acc_OurGB_test_1[iMC,iNEst] = ((preds == y_all1).mean())
        
        preds = weighted_predict(data_discr_test[50:100], theta)
        acc_OurGB_test_2[iMC,iNEst] = ((preds == y_all2).mean())
        
        preds = weighted_predict(data_discr_test[100:150], theta)
        acc_OurGB_test_3[iMC,iNEst] = ((preds == y_all3).mean())
    
    #clf_KMeans_train = KMeans(n_clusters=8).fit(X)
    #clf_KMeans_test = KMeans(n_clusters=4).fit(X_test)
    #pred_train = clf_KMeans_train.predict(X)
    #pred_test = clf_KMeans_test.predict(X)
     
    #print(pred_train)
    #weights_train = np.zeros((X.shape[0]))
    #weights_test = np.zeros((X.shape[0]))
    #for j in range(X.shape[0]):
    #    weights_train[j] = (np.sum(pred_train[j] == pred_train))
    #for j in range(X.shape[0]):
    #    weights_test[j] = (np.sum(pred_train[j] == pred_train))
    
    #weights_train = weights_train/np.sum(weights_train)
    #weights_test = weights_test/np.sum(weights_test)
    
    #weights = weights_test/weights_train

        

# plot Accuracy
fig=plt.figure()
ax=fig.add_subplot(111)
                     
#ax.errorbar(range(len(NEstimators)), acc_OurGB_test_1_3.mean(0), acc_OurGB_test_1_3.std(0), linestyle='--', c='b', marker='_',label='1->2')
#ax.errorbar(range(len(NEstimators)), acc_OurGB_test_1_2.mean(0), acc_OurGB_test_1_2.std(0), linestyle='--', c='b', marker="D",label='1->2')

#ax.errorbar(range(len(NEstimators)), acc_OurGB_test_2_1.mean(0), acc_OurGB_test_2_1.std(0), linestyle='--', c='g', marker='_',label='2->1')
#ax.errorbar(range(len(NEstimators)), acc_OurGB_test_2_3.mean(0), acc_OurGB_test_2_3.std(0), linestyle='--', c='g', marker="D",label='2->3')

#ax.errorbar(range(len(NEstimators)), acc_OurGB_test_3_1.mean(0), acc_OurGB_test_3_1.std(0), linestyle='--', c='r', marker='_',label='3->1')
#ax.errorbar(range(len(NEstimators)), acc_OurGB_test_3_2.mean(0), acc_OurGB_test_3_2.std(0), linestyle='--', c='r', marker="D",label='3->2')

#ax.errorbar(range(len(NEstimators)), acc_OurGB_test_1.mean(0), acc_OurGB_test_1.std(0), linestyle='--', c='m', marker='_',label='f->1')

#ax.errorbar(range(len(NEstimators)), acc_OurGB_test_2.mean(0), acc_OurGB_test_2.std(0), linestyle='--', c='m', marker="D",label='f->2')

#ax.errorbar(range(len(NEstimators)), acc_OurGB_test_3_1.mean(0), acc_OurGB_test_3_1.std(0), linestyle='--', c='m', marker="d",label='f->3')

ax.errorbar(range(len(NEstimators)), ((acc_OurGB_test_1_3 + acc_OurGB_test_1_2)/2).mean(0), ((acc_OurGB_test_1_3 + acc_OurGB_test_1_2)/2).std(0), linestyle='--', c='g', marker='_',label='1->2,3')

ax.errorbar(range(len(NEstimators)), ((acc_OurGB_test_2_1 + acc_OurGB_test_2_3)/2).mean(0), ((acc_OurGB_test_2_3 + acc_OurGB_test_2_1)/2).std(0), linestyle='--', c='b', marker='_',label='2->1,3')

ax.errorbar(range(len(NEstimators)), ((acc_OurGB_test_3_1 + acc_OurGB_test_3_2)/2).mean(0), ((acc_OurGB_test_3_1 + acc_OurGB_test_3_2)/2).std(0), linestyle='--', c='k', marker='_',label='3->1,2')

ax.errorbar(range(len(NEstimators)), ((acc_OurGB_test_3 + acc_OurGB_test_2 + acc_OurGB_test_1)/3).mean(0), ((acc_OurGB_test_3 + acc_OurGB_test_2 + acc_OurGB_test_1)/3).std(0), linestyle='--', c='m', marker='_',label='f->1,2,3')



plt.xticks(ticks=range(len(NEstimators)), labels=NEstimators)
plt.xlabel('Nb estimators')
plt.ylabel('Accuracy')
                     
plt.legend(loc=4)
plt.savefig('simulated_federated_all.png')

