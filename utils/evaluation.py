from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, model_name):
    print(f"Evaluation for {model_name}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))