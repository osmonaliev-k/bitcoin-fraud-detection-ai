# Bitcoin Fraud Detection using AI

This project uses machine learning to detect illicit Bitcoin transactions based on the Elliptic dataset. It implements:

- **Preprocessing** of raw CSVs into train/test arrays  
- **Naive Bayes** baseline  
- **Decision Tree** baseline  
- **GA‑optimized Decision Tree**  
- **Soft‑voting ensemble** of (NB + DT + GA‑optimized DT)  
- **Export** of all predictions to `predictions.csv`

---

## Tech Stack

- Python 3.8+  
- pandas, numpy  
- scikit‑learn  
- matplotlib, seaborn (optional for plots)

---

## Setup & Run

1. **Clone** the repo  
   ```bash
   git clone https://github.com/osmonaliev-k/bitcoin-fraud-detection-ai.git
   cd bitcoin-fraud-detection-ai

2. Virtualenv (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate

3. Install dependencies

   ```bash
   pip3 install -r requirements.txt
   ```
   or
   ```bash
   pip3 install pandas numpy scikit-learn matplotlib seaborn

4. Download the data
   data/elliptic_txs_features.csv
   data/elliptic_txs_classes.csv
   Place both under data/.
   link: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

5. Run the full pipeline:
   ```bash
   python run.py
   ```
   Trains Naive Bayes, Decision Tree, GA‑tuned tree, and ensemble. Prints evaluation for each. Saves predictions.csv with columns: true_label, nb_pred, dt_pred, ga_pred, ensemble_pred

What’s in predictions.csv:

| Column          | Description                                                   |
|-----------------|---------------------------------------------------------------|
| `true_label`    | Ground‑truth label (0 = licit transaction, 1 = illicit fraud) |
| `nb_pred`       | Naive Bayes prediction (0 = licit, 1 = illicit)               |
| `dt_pred`       | Decision Tree prediction (0 = licit, 1 = illicit)            |
| `ga_pred`       | GA‑optimized Tree prediction (0 = licit, 1 = illicit)         |
| `ensemble_pred` | Ensemble prediction (soft‑voting of NB, DT & GA‑DT; 0 or 1)   |    

