from preprocess import load_data
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from utils.evaluation import evaluate_model
import pandas as pd

X_train, X_test, y_train, y_test = load_data()

nb    = GaussianNB()
dt    = DecisionTreeClassifier(max_depth=10, random_state=42)
ga_dt = DecisionTreeClassifier(max_depth=12, min_samples_split=13, min_samples_leaf=16, random_state=42)

ensemble = VotingClassifier(
    estimators=[('nb', nb), ('dt', dt), ('ga_dt', ga_dt)],
    voting='soft'
)

nb.fit(X_train, y_train)
dt.fit(X_train, y_train)
ga_dt.fit(X_train, y_train)
ensemble.fit(X_train, y_train)

nb_preds       = nb.predict(X_test)
dt_preds       = dt.predict(X_test)
ga_preds       = ga_dt.predict(X_test)
ensemble_preds = ensemble.predict(X_test)

evaluate_model(y_test, ensemble_preds, "Ensemble (NB+DT+GA)")

df_out = pd.DataFrame({
    "true_label":     y_test.values,
    "nb_pred":        nb_preds,
    "dt_pred":        dt_preds,
    "ga_pred":        ga_preds,
    "ensemble_pred":  ensemble_preds
})
df_out.to_csv("predictions.csv", index=False)
print("Saved predictions.csv")