# Project Brief: GPU-Accelerated Anti-Money Laundering (AML) Detection

This file is the project brief for Claude Code. Read it fully before doing anything.

## About me

- I'm Rishita, a Year 2 student at Republic Polytechnic (Diploma in Applied AI and Data Analytics), based in Singapore.
- I know Python, pandas, basic ML, Node.js/Express/MySQL, and have trained YOLOv8 models.
- This is a portfolio project for GitHub and LinkedIn, aimed at roles at NVIDIA and big banks (DBS, OCBC, UOB, global banks).
- I need to understand every part of this project well enough to explain it in an interview.

## How to work with me

1. Work one phase at a time (phases are listed below). Do not jump ahead.
2. At the start of each phase, explain in plain language what we're about to do and why, then list the steps.
3. After writing code, explain what it does in simple terms, like you're teaching a student.
4. Explain any new concept the first time it comes up (e.g. data leakage, PR-AUC, SHAP, PageRank).
5. Keep code simple and readable. Add comments. Prefer clear code over clever code.
6. When editing text I wrote myself (README, notes, comments), only add to it. Do not rewrite or remove my wording unless I ask.
7. Written text (README, docs) should sound plain and student-written, not like marketing.
8. Suggest a git commit with a clear message at the end of each phase.
9. Never commit data files, API keys, or `kaggle.json`.
10. If something is ambiguous, ask me before guessing.

## The goal

Build an end-to-end system that detects money laundering in bank transactions, and show:

- Careful data cleaning and feature engineering
- Graph features (money laundering happens across networks of accounts)
- Hyperparameter tuning with Optuna, optimised for business value, not just accuracy
- Explainability with SHAP (banks must explain why something was flagged)
- CPU vs GPU speed benchmarks using NVIDIA RAPIDS (the NVIDIA angle)
- A Streamlit dashboard for a fraud/AML analyst
- An honest README with results and limitations

Honesty rule: the dataset is synthetic. Never overclaim results. Report limitations clearly.

## Dataset

Primary dataset: IBM Transactions for Anti Money Laundering (AML), on Kaggle.

- Kaggle slug: `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`
- Start with `HI-Small_Trans.csv` (HI = higher share of illicit transactions, Small = ~5 million rows).
- `HI-Small_Patterns.txt` describes the laundering patterns (fan-in, fan-out, cycles, etc.). Useful for analysis.
- Expected columns: `Timestamp`, `From Bank`, `Account`, `To Bank`, `Account.1`, `Amount Received`, `Receiving Currency`, `Amount Paid`, `Payment Currency`, `Payment Format`, `Is Laundering`.
- Check the actual columns after downloading, and tell me if they differ.

Download with the Kaggle API (I'll set up `kaggle.json` myself):

```
kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -p data/raw --unzip
```

## Where things run

- My laptop likely has no NVIDIA GPU. Everything must run on CPU first.
- GPU parts (RAPIDS cuDF, cuML, cuGraph / nx-cugraph, XGBoost on CUDA) will run in Google Colab with a free GPU.
- Write code with a `USE_GPU` setting (in `config.yaml`) so the same code runs on both.
- Notebooks must work in Colab: include a setup cell that clones the repo, installs requirements, and downloads data.
- For RAPIDS in Colab: `cudf.pandas` can be enabled with `%load_ext cudf.pandas`. If a RAPIDS library is missing, check the official RAPIDS install docs rather than guessing versions.
- Add a `SAMPLE_FRAC` setting in config so I can develop on a small sample before running on everything.

## Repo structure

```
aml-detection/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
├── data/
│   ├── raw/            # downloaded data (gitignored)
│   └── processed/      # cleaned parquet files (gitignored)
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_tuning.ipynb
│   ├── 06_explainability.ipynb
│   └── 07_gpu_benchmark.ipynb
├── src/
│   ├── config.py
│   ├── data_cleaning.py
│   ├── features.py
│   ├── graph_features.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   ├── explain.py
│   └── benchmark.py
├── app/
│   └── dashboard.py    # Streamlit app
├── results/
│   ├── figures/
│   ├── metrics/
│   └── benchmarks/
├── models/             # saved models (gitignored if large)
└── tests/
```

Logic lives in `src/`. Notebooks call functions from `src/` and show results. This keeps it clean and testable.

## Libraries

pandas, numpy, pyarrow, scikit-learn, xgboost, lightgbm (optional), optuna, shap, networkx, matplotlib, plotly, streamlit, pyyaml, kaggle, jupyter, pytest.
GPU extras (Colab only): cudf, cuml, cugraph or nx-cugraph.

---

## Phases

### Phase 0: Setup

1. Create the folder structure above.
2. Create `.gitignore` (Python, Jupyter checkpoints, `data/`, `models/*.pkl` if large, `kaggle.json`, `.env`, Optuna sqlite files if large).
3. Create `requirements.txt` and `config.yaml` (paths, `SAMPLE_FRAC`, `USE_GPU`, random seed, split ratios, cost assumptions).
4. Write a first README with just the project title, one-line goal, and "work in progress".
5. Explain to me how to set up the Kaggle API and download the data.

Done when: repo structure exists, data downloads, first commit is pushed.

### Phase 1: Data cleaning (`src/data_cleaning.py`, notebook 01)

1. Load the CSV. Report shape, dtypes, memory usage, missing values, duplicates.
2. Rename columns to clear snake_case: `from_bank`, `from_account`, `to_bank`, `to_account`, `amount_received`, `receiving_currency`, `amount_paid`, `payment_currency`, `payment_format`, `is_laundering`.
3. Account IDs may only be unique within a bank. Create unique IDs: `from_id = from_bank + "_" + from_account`, same for `to_id`. Check and explain whether this is needed.
4. Parse `timestamp` to datetime. Sort by time.
5. Currencies: amounts are in different currencies. Convert everything to USD. Try inferring exchange rates from the data itself (rows where paid and received currencies differ). Explain the approach and check it looks sensible.
6. Check for and handle: exact duplicates, negative or zero amounts, self-transfers (same from and to account), extreme outliers. Don't delete things blindly. Explain each decision.
7. Convert categorical columns to category dtype.
8. Save cleaned data to `data/processed/transactions_clean.parquet`.
9. Write a cleaning log (what was found, what was changed, row counts before and after) to `results/metrics/cleaning_report.md`.

Done when: cleaned parquet exists and the cleaning report explains every decision.

### Phase 2: Exploratory data analysis (notebook 02)

Make clear charts and save them to `results/figures/`:

1. Class imbalance: % of laundering transactions.
2. Laundering rate by payment format and by currency.
3. Amount distributions (log scale) for laundering vs normal.
4. Transactions and laundering over time (by hour, by day).
5. Account activity: how many transactions per account, laundering vs normal accounts.
6. Use `HI-Small_Patterns.txt` to summarise which laundering patterns exist.

Write 3 to 5 plain-language key findings at the end of the notebook.

### Phase 3: Feature engineering (`src/features.py`, `src/graph_features.py`, notebook 03)

VERY IMPORTANT: avoid data leakage. Every feature for a transaction must only use information from before that transaction's timestamp. Explain leakage to me with an example before starting.

1. Time-based split first: sort by time, then train (first ~60%), validation (next ~20%), test (last ~20%). Never random split. Ratios come from config.
2. Transaction features: amount in USD, log amount, hour, day of week, payment format, currency, cross-currency flag, cross-bank flag.
3. Account behaviour features (computed from past transactions only, e.g. using shift or rolling windows that exclude the current row):
   - number of transactions sent and received in the last 1 day / 7 days
   - average and max amount sent recently
   - amount compared to the account's usual amount (ratio or z-score)
   - number of unique counterparties recently
   - time since the account's previous transaction
4. Graph features (`graph_features.py`): build a directed graph of accounts with networkx.
   - in-degree, out-degree, unique senders and receivers (fan-in / fan-out)
   - total money in vs out
   - PageRank
   - build the graph for each split using only edges available up to that point, so the test set doesn't leak
   - explain each graph feature and why it relates to laundering patterns
5. Save feature tables for train, val, test as parquet.

Done when: feature tables exist and there's a short written check confirming no future information is used.

### Phase 4: Baseline model (`src/train.py`, `src/evaluate.py`, notebook 04)

1. Train XGBoost with default settings on train, evaluate on validation.
2. Also train a simple logistic regression for comparison.
3. Metrics (explain each one and why accuracy is useless here):
   - PR-AUC (average precision) as the main metric
   - ROC-AUC
   - precision, recall, F1 for the laundering class
   - recall at a fixed false positive rate (e.g. 1%)
4. Business cost metric in `evaluate.py`, using assumptions from config:
   - `cost_false_negative`: cost of missing a laundering transaction
   - `cost_false_positive`: cost of an analyst reviewing a false alert
   - total cost = FN × cost_fn + FP × cost_fp
   - these are assumptions and must be labelled as such everywhere
5. Choose the decision threshold on the validation set by minimising business cost.
6. Save metrics to `results/metrics/baseline.json`.

### Phase 5: Hyperparameter tuning (`src/tune.py`, notebook 05)

1. Use Optuna (TPE sampler) to tune XGBoost. Tune on train, score on validation.
2. Objective: maximise validation PR-AUC. (Optionally a second study that minimises business cost.)
3. Search space (explain each parameter): `max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`, `scale_pos_weight`.
4. Compare imbalance handling: no weighting vs `scale_pos_weight` vs undersampling normal transactions. Report which works best.
5. Use early stopping and Optuna pruning to save time.
6. Save the study to sqlite so it can be resumed.
7. Save Optuna plots: optimisation history, parameter importance, parallel coordinate.
8. Retrain with best params, evaluate ONCE on the test set at the end. Explain why the test set is only touched once.
9. Make a before vs after table: baseline vs tuned (PR-AUC, recall, precision, F1, business cost).

### Phase 6: Explainability (`src/explain.py`, notebook 06)

1. Use SHAP TreeExplainer on the tuned model.
2. Global: summary plot and top features.
3. Local: explain 3 to 5 individual flagged transactions (true positives and false positives) in plain language.
4. Write a short section on why explainability matters for banks, mentioning Singapore's MAS FEAT principles (Fairness, Ethics, Accountability, Transparency). Keep it factual and short.

### Phase 7: GPU benchmark (`src/benchmark.py`, notebook 07, runs in Colab)

1. Benchmark the same steps on CPU vs GPU:
   - loading and cleaning: pandas vs cuDF (or `cudf.pandas`)
   - feature engineering: pandas vs cuDF
   - graph features: networkx vs cuGraph / nx-cugraph
   - training: XGBoost CPU (`device="cpu"`) vs XGBoost GPU (`device="cuda"`, `tree_method="hist"`)
   - Optuna tuning: time for N trials on CPU vs GPU
2. Run each timing a few times and report the median.
3. Record hardware used (CPU type, GPU type from `nvidia-smi`).
4. Save results to `results/benchmarks/benchmarks.csv` and make a bar chart of speedups.
5. Confirm GPU and CPU models give similar accuracy.
6. Be honest if some steps don't speed up much, and explain why.

### Phase 8: Streamlit dashboard (`app/dashboard.py`)

A simple tool for an AML analyst:

1. Overview page: number of transactions, number flagged, estimated cost saved vs no model.
2. Flagged transactions table, sortable by risk score.
3. Click a transaction to see its SHAP explanation in plain words.
4. Account view: show an account's recent transactions and a small network graph of its connections.
5. A threshold slider that updates flagged count, precision, recall, and cost.
6. Must run on a small sample so it works on Streamlit Community Cloud's free tier.

### Phase 9: Tests and cleanup

1. Add pytest tests for key functions, especially one that checks features don't use future data.
2. Remove dead code, make sure notebooks run top to bottom.
3. Add type hints and docstrings to `src/` functions.

### Phase 10: README and LinkedIn

README sections:

1. Title and one-paragraph summary
2. The problem (why AML matters to banks)
3. Dataset (source, size, that it's synthetic)
4. Pipeline diagram (can be a Mermaid diagram)
5. Data cleaning highlights
6. Key EDA findings with 2 to 3 charts
7. Features, including graph features
8. Results: baseline vs tuned table, business cost
9. Explainability example
10. CPU vs GPU benchmark chart and table
11. Dashboard screenshot / GIF and link
12. How to run it
13. Limitations and what I'd do next
14. What I learned

Then help me draft a short LinkedIn post (plain tone, not hype) with 1 to 2 charts and the repo link.

---

## Interview prep (for later)

At the end, create `INTERVIEW_NOTES.md` with questions an interviewer might ask and short answers based on this project, for example:

- Why not use accuracy?
- How did you avoid data leakage?
- Why a time-based split?
- How did you choose the threshold?
- What do the graph features capture?
- Why did the GPU speed up some steps more than others?
- What are the limitations of synthetic data?
- How would this work in a real bank?

Don't write the answers for me until I've finished the phases, so I understand them properly first.
