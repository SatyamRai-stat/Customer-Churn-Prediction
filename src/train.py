from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_loader import get_train_test_data

def train_models():
    X_train,X_test,y_train,y_test=get_train_test_data()
    from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_loader import get_train_test_data


def train_models():

    X_train, X_test, y_train, y_test = (
        get_train_test_data()
    )

    models = {
        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Decision Tree":
            DecisionTreeClassifier(random_state=42),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

        "XGBoost":
            XGBClassifier(
                random_state=42
            )
    }
    
    trained_models={}
    for name, model in models.item():
        model.fit(X_train,y_train)
        trained_models[name]=model
    return trained_models
