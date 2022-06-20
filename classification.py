from collect_data import DataCollection
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer,f1_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
import numpy as np

class Classification:
    counter = 0
    
    def __init__(self,classifier,path,label_col,features_cols):
        self.__classifier = classifier
        self.__path = path
        self.__label_col = label_col
        self.__features_cols = features_cols
        
    def classify(self,**kwargs):
        if self.__classifier == 'decision_tree' and kwargs.get('max_depth') is not None:
            self.__classify_DT(kwargs.get('max_depth'))
        elif self.__classifier == 'kneighbors':
            if kwargs.get('k') is not None:
                self.__kneighbors(kwargs.get('k'))
            else:
                raise Exception('k parameter is missing...')
        elif self.__classifier == 'random_forest':
            self.__classify_RT()
        elif self.__classifier == 'svc':
            self.__classify_SVM()
        elif self.__classifier == 'mlpc' and kwargs.get('iter') is not None and kwargs.get('activation_func') is not None:
            self.__classify_MLP(kwargs.get('iter'),kwargs.get('activation_func'))
        else:
            raise Exception('The parameters are not defined properly...')
        
    def __split_labels_features(self):
        dc = DataCollection(self.__path)
        students_df = dc.collect(self.__label_col)
        labels = students_df[self.__label_col]
        features = students_df[self.__features_cols]
        return features,labels
    
    def __split_labels_features_std(self):
        dc = DataCollection(self.__path)
        students_df = dc.collect_standardized_data(self.__label_col)
        labels = students_df[self.__label_col]
        features = students_df[self.__features_cols]
        return features,labels
    
    def __classify_DT(self,max_depth):
        students_details,labels = self.__split_labels_features()
        cv = KFold(n_splits=10)
        dtc = DecisionTreeClassifier(max_depth=max_depth,random_state=0)
        self.__evaluate_classification(dtc,students_details.values,labels.values,cv)
        self.__visualize_DT(dtc.fit(students_details.values,labels.values),self.__features_cols,labels.values)
    
    def __classify_RT(self):
        students_details,labels = self.__split_labels_features()
        cv = KFold(n_splits=10)
        rfc = RandomForestClassifier(max_depth=5, random_state=0)
        self.__evaluate_classification(rfc,students_details.values,labels.values,cv)
        
    def __kneighbors(self,k):
        students_details,labels = self.__split_labels_features()
        cv = KFold(n_splits=10)
        knn = KNeighborsClassifier(n_neighbors=k) 
        self.__evaluate_classification(knn,students_details.values,labels.values,cv)
        
    def __classify_SVM(self):
        students_details,labels = self.__split_labels_features()
        cv = KFold(n_splits=10)
        svc = SVC(kernel='linear')
        self.__evaluate_classification(svc,students_details.values,labels.values,cv)
        
    def __classify_MLP(self,iterations,activation_func):
        students_details,labels = self.__split_labels_features_std()
        cv = KFold(n_splits=10)
        mlpc = MLPClassifier(random_state=1, max_iter=iterations, solver='lbfgs',activation=activation_func)
        self.__evaluate_classification(mlpc,students_details.values,labels.values,cv)
        
    def __evaluate_classification(self,classifier,features,labels,cv):
        uniq_labels = np.unique(labels)
        accuracy_scores = cross_val_score(classifier,features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
        f1_scores = cross_val_score(classifier,features, labels, scoring=make_scorer(f1_score, average='weighted', labels=uniq_labels, zero_division=0), cv=cv, n_jobs=-1)
        recall_scores = cross_val_score(classifier,features, labels, scoring=make_scorer(recall_score, average='weighted', labels=uniq_labels, zero_division=0), cv=cv, n_jobs=-1)

        mean_accuracy = np.mean(accuracy_scores)
        mean_f1_score = np.mean(f1_scores)
        mean_recall_score = np.mean(recall_scores)
        
        print(f" Mean Accuracy: {mean_accuracy} \n Mean f1 score: {mean_f1_score} \n Mean recall score: {mean_recall_score}",end="\n===============================================================\n")
    
        with open('results.txt','a') as f:
            print(f" Mean Accuracy: {mean_accuracy} \n Mean f1 score: {mean_f1_score} \n Mean recall score: {mean_recall_score}",end="\n===============================================================\n",file=f)
    
    def __visualize_DT(self,clf,features_cols,labels):
        uniq_labels = np.unique(labels)
        fig = plt.figure(figsize=(30,30))
        plot_tree(clf,feature_names=features_cols,class_names=uniq_labels,filled=True,rounded=True,fontsize=16)  
        Classification.counter += 1
        fig.savefig(f"diagrams/decision_tree({Classification.counter}).png")
