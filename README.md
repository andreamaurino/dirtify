# Dirtify

**Dirtify** is a Python tool suite for systematically evaluating the sensitivity of machine learning models to data quality errors. It implements the **Error Sensitivity Profile (ESP)** framework, which quantifies how model performance changes as data corruption increases — revealing whether a model degrades gracefully, collapses abruptly, or even improves under certain error types.


## What is ESP?

The **Error Sensitivity Profile** characterises the relationship between data corruption and model performance through three complementary components:

| Component | Description |
|-----------|-------------|
| **EPC** — Error Performance Correlation | Global linear trend between corruption level and performance |
| **AEPC** — Area under the Error–Performance Curve | Cumulative deviation from baseline across all corruption levels |
| **Slope vector** | Local regime structure: thresholds, turning points, saturation effects |

A positive AEPC means corruption *improves* model performance relative to the clean baseline — a counterintuitive but empirically documented phenomenon. A negative AEPC indicates degradation.

![ESP canonical representation](esp_canonical_example.png)

---

## Architecture

Dirtify consists of three loosely coupled tools:

```
┌─────────────────┐     JSON config     ┌─────────────────┐     results     ┌──────────────────┐
│   Configurator  │ ─────────────────▶  │     Trainer     │ ──────────────▶ │   ESP Builder    │
│                 │                     │                 │                 │                  │
│ Define error    │                     │ Inject errors   │                 │ Compute EPC,     │
│ strategy via    │                     │ via PuckTrick   │                 │ AEPC, slopes     │
│ wizard UI       │                     │ Train models    │                 │ Plot & cluster   │
│                 │                     │ via scikit-learn│                 │ ESP curves       │
└─────────────────┘                     └─────────────────┘                 └──────────────────┘
```

Data corruption is injected via [PuckTrick](https://github.com/andreamaurino/pucktrick), which supports five error types:

| Error type | Mechanism |
|------------|-----------|
| **Outlier** | Values replaced by $\bar{X} \pm 3\sigma$ boundaries (3-sigma method) |
| **Missing** | Values replaced by NaN (MCAR) |
| **Duplicate** | Rows copied and appended to training partition |
| **Noise** | Values randomly distorted within feature range |
| **Labels**| Swapping values

---

## Installation

```bash
git clone https://github.com/andeamaurino/dirtify.git
cd dirtify
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Python 3.11 is required.** The following package versions are pinned for compatibility:

```
scikit-learn==1.3.2
imbalanced-learn==0.11.0
pycaret==3.3.2
hdbscan==0.8.33
```

---

## Quick Start

### Step 1 — Configure your experiment

Choose or Save the dataset into the datasetRoot directory
Run the Configurator to define which dataset, task, models, error types, and features to analyse:

```bash
python Configurator.py
```

This produces a JSON configuration file (e.g., `./json/config_mydata.json`) that specifies the full experimental grid.

### Step 2 — Run the experiments

```bash
python Trainer.py ./json/config_mydata.json
```

The Trainer injects errors at each corruption level, trains each model, and stores performance metrics in `./experiments/`.

### Step 3 — Compute and visualise the ESP

```bash
python ESP_Builder.py --dataset mydata.csv --metric F1 --aepc 0.05
```

This produces:
- A CSV with significant ESP scenarios (filtered by Wilcoxon + BY-FDR + |AEPC| > threshold)
- ESP canonical plots for each significant scenario
- A cluster plot of ESP curve shapes


## Supported Models
### Classification (via PyCaret)
 
| Label | Model |
|-------|-------|
| SVM | SVM — Linear Kernel |
| ET | Extra Trees Classifier |
| RF | Random Forest Classifier |
| KN | K Neighbors Classifier |
| LDA | Linear Discriminant Analysis |
| MLP | Multilayer Perceptron Classifier |
| LR | Logistic Regression |
| NB | Naive Bayes |
| DT | Decision Tree Classifier |
| QDA | Quadratic Discriminant Analysis |
| SGD | Stochastic Gradient Descent Classifier |
| RC | Ridge Classifier |
| ADA | AdaBoost |
| XG | XGBoost |
 
### Clustering (via scikit-learn)
 
| Label | Model |
|-------|-------|
| KM | K-Means |
| HDB | HDBSCAN |
| GMM | Gaussian Mixture Model |
| BIRCH | BIRCH |
| HR | Hierarchical Clustering (Ward linkage) |
 



## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

Andrea Maurino — [andrea.maurino@unimib.it](mailto:andrea.maurino@unimib.it)  
Università degli Studi di Milano-Bicocca, Dipartimento di Informatica, Sistemistica e Comunicazione


