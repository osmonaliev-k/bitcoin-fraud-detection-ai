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
   git clone https://github.com/yourusername/bitcoin-fraud-detection-ai.git
   cd bitcoin-fraud-detection-ai

2. Virtualenv (recommended)
   python3 -m venv venv
   source venv/bin/activate

3. Install dependencies
   pip install -r requirements.txt
   - or install them manually 

4. Download the data
   data/elliptic_txs_features.csv
   data/elliptic_txs_classes.csv
   Place both under data/.
   link: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

5. Run the full pipeline:
   python run.py
   rains Naive Bayes, Decision Tree, GA‑tuned tree, and ensemble
   Prints evaluation for each
   Saves predictions.csv with columns: true_label, nb_pred, dt_pred, ga_pred, ensemble_pred

    What’s in predictions.csv:
    Column	        Description
    true_label	    0 = licit, 1 = illicit
    nb_pred	        Naive Bayes prediction (0 or 1)
    dt_pred	        Decision Tree prediction
    ga_pred	        GA‑optimized tree prediction
    ensemble_pred   Soft‑voting ensemble prediction

