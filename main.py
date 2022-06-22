from classification import Classification
from regression import Regression

if __name__ == '__main__':
    MATHS_CLASS_PATH = './data/student-mat.csv'
    POR_CLASS_PATH = './data/student-por.csv'
    
    def display_classification_results(label,cols,iter_mat,iter_por,activation_mat,activation_por):
        '''
        Display the results of classification's validation in screen and store them to a txt file.
        '''
        print(f"RESULTS FOR {label} LABEL ON MATHS CLASS...",end="\n===============================================================\n")
        
        with open("results.txt","a") as f:
            print(f"RESULTS FOR {label} LABEL ON MATHS CLASS...",end="\n===============================================================\n",file=f)
        
        dt_mat_clf = Classification('decision_tree',MATHS_CLASS_PATH,label,cols)
        dt_mat_clf.__str__()
        dt_mat_clf.classify(max_depth=5)
        
        knn_mat_clf = Classification('kneighbors',MATHS_CLASS_PATH,label,cols)
        knn_mat_clf.__str__()
        knn_mat_clf.classify(k=5) 
        
        rf_mat_clf = Classification('random_forest',MATHS_CLASS_PATH,label,cols)
        rf_mat_clf.__str__()
        rf_mat_clf.classify()  
        
        svc_mat_clf = Classification('svc',MATHS_CLASS_PATH,label,cols)
        svc_mat_clf.__str__()
        svc_mat_clf.classify()
        
        mlpc_mat_clf = Classification('mlpc',MATHS_CLASS_PATH,label,cols)
        mlpc_mat_clf.__str__()
        mlpc_mat_clf.classify(iter=iter_mat,activation_func=activation_mat)  
        
        print(f"RESULTS FOR {label} LABEL ON PORTUGUESE CLASS...",end="\n===============================================================\n")
        
        with open("results.txt","a") as f:
            print(f"RESULTS FOR {label} LABEL ON PORTUGUESE CLASS...",end="\n===============================================================\n",file=f)
            
        dt_por_clf = Classification('decision_tree',POR_CLASS_PATH,label,cols)
        dt_por_clf.__str__()
        dt_por_clf.classify(max_depth=5)
        
        knn_por_clf = Classification('kneighbors',POR_CLASS_PATH,label,cols)
        knn_por_clf.__str__()
        knn_por_clf.classify(k=5) 
        
        rf_por_clf = Classification('random_forest',POR_CLASS_PATH,label,cols)
        rf_por_clf.__str__()
        rf_por_clf.classify()  
        
        svc_por_clf = Classification('svc',POR_CLASS_PATH,label,cols)
        svc_por_clf.__str__()
        svc_por_clf.classify()  
        
        mlpc_por_clf = Classification('mlpc',POR_CLASS_PATH,label,cols)
        mlpc_por_clf.__str__()
        mlpc_por_clf.classify(iter=iter_por,activation_func=activation_por)
      
    def display_regression_results():
        '''
        Display the results of regression's validation in screen and store them to a txt file.
        '''
        print("REGRESSION FOR MATHS CLASS...",end="\n===============================================================\n")
        with open("results.txt","a") as f:
            print("REGRESSION FOR MATHS CLASS...",end="\n===============================================================\n",file=f)
        
        dt_mat_reg = Regression('decision_tree',MATHS_CLASS_PATH)
        dt_mat_reg.__str__()
        dt_mat_reg.regress('G3')
        
        rf_mat_reg = Regression('random_forest',MATHS_CLASS_PATH)
        rf_mat_reg.__str__()
        rf_mat_reg.regress('G3')
        
        mlpr_mat_reg = Regression('mlpr',MATHS_CLASS_PATH)
        mlpr_mat_reg.__str__()
        mlpr_mat_reg.regress('G3',iter=21,activation_func='identity')
        
        mlr_mat_reg = Regression('mlr',MATHS_CLASS_PATH)
        mlr_mat_reg.__str__()
        mlr_mat_reg.regress('G3')
        
        print("REGRESSION FOR PORTUGUESE CLASS...",end="\n===============================================================\n")
        with open("results.txt","a") as f:
            print("REGRESSION FOR PORTUGUESE CLASS...",end="\n===============================================================\n",file=f)
        
        dt_por_reg = Regression('decision_tree',POR_CLASS_PATH)
        dt_por_reg.__str__()
        dt_por_reg.regress('G3')
        
        rf_por_reg = Regression('random_forest',POR_CLASS_PATH)
        rf_por_reg.__str__()
        rf_por_reg.regress('G3')
        
        mlpr_por_reg = Regression('mlpr',POR_CLASS_PATH)
        mlpr_por_reg.__str__()
        mlpr_por_reg.regress('G3',iter=18,activation_func='identity')
        
        mlr_mat_reg = Regression('mlr',POR_CLASS_PATH)
        mlr_mat_reg.__str__()
        mlr_mat_reg.regress('G3')
    
    cols_for_school = ['studytime','famsup','address','traveltime'] 
    display_classification_results('school',cols_for_school,9,7,'identity','identity')
    
    cols_for_higher = ['famsup','G3','school','address']
    display_classification_results('higher',cols_for_higher,9,9,'identity','identity')
    
    cols_for_grade = ['G1','G2','studytime','traveltime','Dalc','paid']
    display_classification_results('G3',cols_for_grade,40,36,'identity','identity')
    
    display_regression_results()
    