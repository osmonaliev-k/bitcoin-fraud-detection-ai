from sklearn.naive_bayes import GaussianNB
from utils.evaluation import evaluate_model

def run_naive_bayes(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    evaluate_model(y_test, preds, "Naive Bayes")
    return preds