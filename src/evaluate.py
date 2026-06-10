import pandas as pd 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from src.data_loader import get_train_test_data
from src.train import train_models

def evaluate_model():
    X_train, X_test, y_train, y_test = get_train_test_data()
    trained_models=train_models()
    
    results = []

    for name, model in trained_models.items():

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]


            results.append({
        "Model": name,

        "Accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "Precision":
            precision_score(
                y_test,
                y_pred
            ),

        "Recall":
            recall_score(
                y_test,
                y_pred
            ),

        "F1":
            f1_score(
                y_test,
                y_pred
            ),

        "ROC_AUC":
            roc_auc_score(
                y_test,
                y_prob
            )
    })
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
            by="F1",
            ascending=False
        )

    return results_df


if __name__ == "__main__":
    print(evaluate_model())