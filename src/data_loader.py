import pandas as pd 
import joblib
from src.preprocess import preprocess_data
from sklearn.model_selection import train_test_split

def get_train_test_data():
    df=pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    df=preprocess_data(df)

    X=df.drop('churn',axis=1)
    y=df['chrun']
    
    return train_test_split(X,y,random_state=42,test_size=0.20,stratify=y)


