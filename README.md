# Mini-VAEP-project-_-Data-analysis
An end-to-end soccer analytics engine using PyTorch and Socceraction to value player actions via VAEP. It ingests raw StatsBomb event data, converts it to SPADL format, trains deep neural networks on spatial-temporal game states, and evaluates offensive/defensive contributions per 90 minutes alongside traditional match metrics.



# `README.md`

```markdown
# VAEP Analytics Engine with PyTorch & Socceraction

An end-to-end Python pipeline for soccer action valuation using **Valuing Actions by Estimating Probabilities (VAEP)**. Built on **PyTorch** and **Socceraction**, this engine downloads open-source event data from StatsBomb, transforms raw events into SPADL actions, trains neural network models to predict scoring ($P_{scores}$) and conceding ($P_{concedes}$) probabilities, and evaluates player performance across metrics like VAEP per 90 minutes and individual action contributions.

---

## Features

- **Automated Data Retrieval & Caching**: Multi-threaded downloader for StatsBomb open match, event, and lineup data[cite: 2].
- **SPADL Conversion**: Standardized spatial-temporal event format conversion via `socceraction.spadl`[cite: 2].
- **Feature Engineering**: Game state construction using action-type one-hot encoding, start/end locations, spatial movements, and deltas[cite: 2].
- **Deep Learning Framework**: Dual PyTorch neural networks with batch normalization, dropout, and `AdamW` optimizer for probability estimation[cite: 2].
- **Advanced Player Analytics**:
  - Top individual actions valuation (excluding penalties)[cite: 2].
  - Defensive action impact analysis[cite: 2].
  - Per-90-minute normalized metrics filtered by playing time thresholds[cite: 2].
  - VAEP correlation against traditional stats (Goals, Tackles, Interceptions)[cite: 2].

---

## Installation

Ensure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install numpy pandas torch tqdm requests socceraction

```

---

## Project Structure

```text
├── run_vaep.py             # Main pipeline script
├── README.md               # Project documentation
└── statsbomb_data/         # Automatically created local cache directory
    ├── events/
    ├── matches/
    └── lineups/

```

---

## Data Pipeline & Datasets

### Training Set

The model trains on open event data across multiple major competitions:

* FIFA World Cup 2018


* UEFA Euro 2020


* UEFA Champions League 2018/19


* La Liga 2017/18


* Premier League 2003/04



### Evaluation Set

Evaluated out-of-sample on **La Liga 2018/19**.

---

## Usage

Run the entire pipeline from data ingestion to analytics output using:

```bash
python run_vaep.py

```

---

## Output Insights

When executed, `run_vaep.py` performs the following steps and prints detailed analytical tables:

1. **Training Phase**: Converts training matches into SPADL representation, extracts features and labels, and trains $P_{scores}$ and $P_{concedes}$ PyTorch classifiers.


2. **Evaluation Phase**: Scores out-of-sample actions from La Liga 2018/19 and merges player metadata.


3. **Analytics Generated**:
* **Top 10 Players by VAEP/90** (Minimum 450 minutes played).


* **Top 10 Individual Non-Penalty Actions**.


* **Top 10 Defensive Actions**.


* **VAEP vs Traditional Metrics Comparison** (Goals, Tackles, Interceptions).





```

```
