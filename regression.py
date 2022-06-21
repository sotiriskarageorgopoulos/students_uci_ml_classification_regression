from collect_data import DataCollection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np

class Regression:
    def __init__(self,regressor,path):
        self.__path = path
        self.__regressor = regressor
    
    def __str__(self):
        print(f"Regression with {self.__regressor}...",end="\n===============================================================\n")  
        with open("results.txt","a") as f:
            print(f"Regression with {self.__regressor}...",end="\n===============================================================\n",file=f)  
        return f"Classification with {self.__regressor}..."
     
    def regress(self,target,**kwargs):
        '''
        Predicts a target variable with Regression Algorithmns. \n
        Arguments:
        - target: Target variable \n
        Keyword Arguments:
        - iter: Number of iterations for Neural Network.
        - activation_func: activation function for neurons of Neural Network.
        '''
        if self.__regressor == 'decision_tree' and target is not None:
            self.__regress_DT(target)
        elif self.__regressor == 'random_forest' and target is not None:
            self.__regress_RF(target)
        elif self.__regressor == 'mlpr' and target is not None and kwargs.get('iter') is not None and kwargs.get('activation_func') is not None:
            iter = kwargs.get('iter')
            activation_func = kwargs.get('activation_func')
            self.__regress_MLP(target,iter,activation_func)
        else:
            raise Exception('The parameters are not defined properly...')
    
    def __split_features_targets(self,target_): 
        '''
        Split the data set to standardized features and target
        '''
        dc = DataCollection(self.__path)
        students_df = dc.collect_standardized_data()
        features = students_df.loc[:,students_df.columns != target_]
        target = students_df[target_]
        return features,target
    
    def __regress_DT(self,target_):
        '''
        Regress with Decision Tree Algorithm.
        '''
        features, target = self.__split_features_targets(target_)
        regressor = DecisionTreeRegressor(random_state=0)
        cv = KFold(n_splits=10)
        self.__evaluate_regression(regressor,features,target,cv)
    
    def __regress_MLP(self,target_,iter,activation_func):
        '''
        Regress with Multi Layer Perceptron Neural Network.
        '''
        features, target = self.__split_features_targets(target_)
        regressor = MLPRegressor(random_state=1, max_iter=iter, activation=activation_func, solver='lbfgs')
        cv = KFold(n_splits=10)
        self.__evaluate_regression(regressor,features,target,cv)
    
    def __regress_RF(self,target_):
        '''
        Regress with Random Forest Algorithm
        '''
        features, target = self.__split_features_targets(target_)
        regressor = RandomForestRegressor(random_state=0)
        cv = KFold(n_splits=10)
        self.__evaluate_regression(regressor,features,target,cv)
        
    def __evaluate_regression(self,regressor,features,target,cv):
        '''
        Evaluates regressor with K-Fold cross validation.\n
        The used measures are mean RMSE and mean R^2 score.
        '''
        RMSE_scores = cross_val_score(regressor,features,target,scoring='neg_root_mean_squared_error',cv=cv, n_jobs=-1)
        R2_scores = cross_val_score(regressor,features,target,scoring='r2',cv=cv,n_jobs=-1)
        
        mean_RMSE = -np.mean(RMSE_scores)
        mean_R2 = np.mean(R2_scores)
        
        print(f" Mean RMSE: {mean_RMSE} \n Mean R^2 score: {mean_R2}",end="\n===============================================================\n")
        
        with open('results.txt','a') as f:
            print(f" Mean RMSE: {mean_RMSE} \n Mean R^2 score: {mean_R2}",end="\n===============================================================\n",file=f)
    