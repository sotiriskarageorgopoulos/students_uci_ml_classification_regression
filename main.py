from classification import Classification

if __name__ == '__main__':
    MATHS_CLASS_PATH = './data/student-mat.csv'
    POR_CLASS_PATH = './data/student-por.csv'
    
    def display_classification_results(label,cols,iter_mat,iter_por,activation_mat,activation_por):
        print(f"RESULTS FOR {label} LABEL ON LESSEON MATHS...",end="\n===============================================================\n")
        
        with open("results.txt","a") as f:
            print(f"RESULTS FOR {label} LABEL ON LESSEON MATHS...",end="\n===============================================================\n",file=f)
        
        dt_mat_clf = Classification('decision_tree',MATHS_CLASS_PATH,label,cols)
        dt_mat_clf.classify(max_depth=5)
        
        knn_mat_clf = Classification('kneighbors',MATHS_CLASS_PATH,label,cols)
        knn_mat_clf.classify(k=2) 
        
        rf_mat_clf = Classification('random_forest',MATHS_CLASS_PATH,label,cols)
        rf_mat_clf.classify()  
        
        svc_mat_clf = Classification('svc',MATHS_CLASS_PATH,label,cols)
        svc_mat_clf.classify()
        
        mlpc_mat_clf = Classification('mlpc',MATHS_CLASS_PATH,label,cols)
        mlpc_mat_clf.classify(iter=iter_mat,activation_func=activation_mat)  
        
        print(f"RESULTS FOR {label} LABEL ON LESSEON PORTUGUESE...",end="\n===============================================================\n")
        
        with open("results.txt","a") as f:
            print(f"RESULTS FOR {label} LABEL ON LESSEON PORTUGUESE...",end="\n===============================================================\n",file=f)
            
        dt_por_clf = Classification('decision_tree',POR_CLASS_PATH,label,cols)
        dt_por_clf.classify(max_depth=5)
        
        knn_por_clf = Classification('kneighbors',POR_CLASS_PATH,label,cols)
        knn_por_clf.classify(k=2) 
        
        rf_por_clf = Classification('random_forest',POR_CLASS_PATH,label,cols)
        rf_por_clf.classify()  
        
        svc_por_clf = Classification('svc',POR_CLASS_PATH,label,cols)
        svc_por_clf.classify()  
        
        mlpc_por_clf = Classification('mlpc',POR_CLASS_PATH,label,cols)
        mlpc_por_clf.classify(iter=iter_por,activation_func=activation_por)
       
       
    cols_for_school = ['studytime','famsup','address','traveltime'] 
    display_classification_results('school',cols_for_school,9,7,'identity','identity')
    
    cols_for_higher = ['famsup','G3','school','address']
    display_classification_results('higher',cols_for_higher,9,9,'identity','identity')
    
    cols_for_grade = ['G1','G2','studytime','traveltime','Dalc','paid']
    display_classification_results('G3',cols_for_grade,40,36,'identity','identity')
    