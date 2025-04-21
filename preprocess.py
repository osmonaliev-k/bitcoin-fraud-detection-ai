import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    # 1) Define column names
    feature_cols = ["node_id", "time_step"] + [f"f{i}" for i in range(1, 167)]
    
    # 2) Read features & classes as strings
    features_df = pd.read_csv(
        "data/elliptic_txs_features.csv",
        header=None,
        names=feature_cols,
        dtype=str,
        low_memory=False
    )
    classes_df = pd.read_csv(
        "data/elliptic_txs_classes.csv",
        header=None,
        names=["node_id", "class"],
        dtype=str
    )

    # 3) Trim whitespace on node_id & class
    features_df["node_id"] = features_df["node_id"].str.strip()
    classes_df["node_id"] = classes_df["node_id"].str.strip()
    classes_df["class"]   = classes_df["class"].str.strip()

    # 4) Merge on node_id (inner join automatically drops IDs missing labels)
    df = pd.merge(features_df, classes_df, on="node_id", how="inner")

    print("Merged data shape:", df.shape)
    print(df["class"].value_counts())

    # 5) Keep only the 1 (illicit) & 2 (licit) labels
    df = df[df["class"].isin(["1","2"])].copy()
    df["class"] = df["class"].astype(int)

    # 6) Coerce each feature column to float (bad strings → NaN)
    feat_cols = [f"f{i}" for i in range(1, 167)]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 7) **Fill** any NaNs in features with 0 (instead of dropping ALL rows)
    df[feat_cols] = df[feat_cols].fillna(0.0)

    print(f"Final dataset size: {df.shape}, label counts: {df['class'].value_counts().to_dict()}")

    # 8) Split X/y and train/test
    X = df[feat_cols]
    y = df["class"].replace({2: 0})   # licit→0, illicit→1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 9) Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test