from sklearn.tree import DecisionTreeClassifier
from utils.evaluation import evaluate_model

def run_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    evaluate_model(y_test, preds, "Decision Tree")
    return preds