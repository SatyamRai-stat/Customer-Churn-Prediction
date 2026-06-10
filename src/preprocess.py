import pandas as pd 

def preprocess_data(df):
   df.columns=df.columns.str.lower()
   
   if "customerid" in df.columns:
       df=df.drop("customerid",axis=1)
       
   if "totalcharges" in df.columns:
        df['totalcharges']=pd.to_numeric(df['totalcharges'],errors="coerce")
        
        df["totalcharges"] = df["totalcharges"].fillna(df["totalcharges"].median())
        
   if "gender" in df.columns:
       df['gender'] = (
                    df['gender']
                    .str.strip()
                    .str.lower()
                    .map({
                       'female': 0,
                        'male': 1
                         })
                    )
    
   yes_or_no_cols=[
    "partner",
    "dependents",
    "phoneservice",
    "paperlessbilling"
]
   for col in yes_or_no_cols:
        if col in df.columns:
            df[col] = (
                    df[col]
                    .str.strip()
                    .str.lower()
                    .map({
                        'no': 0,
                        'yes': 1
                        })
                    )   
            
   if "churn" in df.columns:
       df['churn']=df['churn'].map({
           'No':0,
           'Yes':1
       })
   
   categorical_cols=[
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paymentmethod"
]
   existing_cols=[col for col in categorical_cols if col in df.columns]
   df=pd.get_dummies(
       df,
       columns=existing_cols,
       drop_first=True,
       dtype=int
   ) 
   
   return df