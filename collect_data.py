import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataCollection():
    def __init__(self,path):
        self.__path = path

    def get_path(self):
        return self.__path

    def set_path(self,path):
        self.__path = path

    def collect_standardized_data(self,label):
        students_df = pd.read_csv(self.__path,sep=";")
        if label == 'G3':
            students_df = self.__five_level_grade(students_df)
            
        nominal_cols = students_df.select_dtypes(['object']).columns
        labels = students_df[label].values
        nominal_cols_except_label = list(filter(lambda c: c != label,nominal_cols))
        for col in nominal_cols_except_label:
            students_df[col] = LabelEncoder().fit_transform(students_df[col].values)
        cols_except_label = list(filter(lambda c: c != label,students_df.columns))
        return self.__standardize(students_df,cols_except_label,label,labels)

    def collect(self,label):
        students_df = pd.read_csv(self.__path,sep=";")
        if label == 'G3':
            students_df = self.__five_level_grade(students_df)
            
        nominal_cols = students_df.select_dtypes(['object']).columns
        nominal_cols_except_label = list(filter(lambda c: c != label,nominal_cols))
        for col in nominal_cols_except_label:
            students_df[col] = LabelEncoder().fit_transform(students_df[col].values)
        return students_df

    def __standardize(self,df,cols,label_col,labels):
        standardized_df = (df[cols] - df[cols].mean())/ df[cols].std(ddof=0)
        standardized_df[label_col] = labels
        return standardized_df

    def __five_level_grade(self,df:pd.DataFrame):
        df = df.astype({'G1':'int','G2':'int'})
        df["G3"].replace([20,19,18,17,16],'A',inplace=True)
        df["G3"].replace([15,14],'B',inplace=True)
        df["G3"].replace([13,12],'C',inplace=True)
        df["G3"].replace([11,10],'D',inplace=True)
        df["G3"].replace([9,8,7,6,5,4,3,2,1,0],'F',inplace=True)
        df["G1"].replace([20,19,18,17,16],'A',inplace=True)
        df["G1"].replace([15,14],'B',inplace=True)
        df["G1"].replace([13,12],'C',inplace=True)
        df["G1"].replace([11,10],'D',inplace=True)
        df["G1"].replace([9,8,7,6,5,4,3,2,1,0],'F',inplace=True)
        df["G2"].replace([20,19,18,17,16],'A',inplace=True)
        df["G2"].replace([15,14],'B',inplace=True)
        df["G2"].replace([13,12],'C',inplace=True)
        df["G2"].replace([11,10],'D',inplace=True)
        df["G2"].replace([9,8,7,6,5,4,3,2,1,0],'F',inplace=True)
        return df

