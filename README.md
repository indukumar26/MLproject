# Project Overview: Lithium-Ion Battery SOH & RUL Prognostics

An end-to-end deep learning framework for **State of Health (SOH, %)** and **Remaining Useful Life (RUL, cycles)** estimation of lithium-ion batteries using Long Short-Term Memory (LSTM) neural networks and cross-domain transfer learning.

---

## 🎯 Benchmark Results

* **NASA Benchmark (Battery_04 / B0018):**
  * **SOH MAE:** **2.51%** (RMSE: 3.02%, $R^2 = \mathbf{0.8566}$)
  * **RUL MAE:** **8.74 cycles** (RMSE: 11.05 cycles, $R^2 = \mathbf{0.7094}$)
* **CALCE Transfer Learning (CS2_21):**
  * **SOH MAE:** **4.60%** ($R^2 = \mathbf{0.9416}$)
  * **RUL MAE:** **0.81 cycles**

---

## 📓 Notebook Pipeline (01 to 12)

The experimental workflow is structured into 12 sequentially numbered Jupyter notebooks located in `notebooks/`:

| Notebook | Name | Stage / Focus | Output Artifacts |
|:---:|:---|:---|:---|
| **01** | `01_raw_mat_to_csv_conversion.ipynb` | Telemetry Ingestion | `data/intermediate/converted_data/` |
| **02** | `02_exploratory_data_analysis.ipynb` | EDA & Sensor Profiling | `outputs/02_eda/` |
| **03** | `03_cycle_feature_extraction.ipynb` | Feature Extraction | `data/intermediate/preprocessed_data/` |
| **04** | `04_soh_and_rul_target_formulation.ipynb` | Target Formulation | `data/intermediate/processed_battery_data/` |
| **05** | `05_master_dataset_construction.ipynb` | Initial Dataset Compilation | `data/intermediate/final_dataset/` |
| **06** | `06_data_quality_audit_and_target_correction.ipynb` | **Quality Audit & Target Correction** | `data/final/corrected_data/` |
| **07** | `07_lstm_sequence_engineering.ipynb` | Sliding Windows & Scaling | `data/final/filtered_lstm_data/` |
| **08** | `08_baseline_lstm_modeling.ipynb` | Baseline Multi-Task LSTM | Initial baseline metrics |
| **09** | `09_performance_diagnosis_and_error_attribution.ipynb` | Error Diagnosis & Audits | `outputs/08_diagnosis/` |
| **10** | `10_optimized_multi_task_lstm_training.ipynb` | **Optimized Model Training** | `models/nasa_lstm/` |
| **11** | `11_calce_dataset_preparation.ipynb` | External CALCE Ingestion | `data/intermediate/calce_prepared_data/` |
| **12** | `12_calce_transfer_learning.ipynb` | **Cross-Domain Transfer Learning** | `models/calce_transfer/` |

---

## 📁 Repository Directory Structure

```
project ml/
├── notebooks/          # 12 clean sequentially numbered notebooks (01 to 12)
├── outputs/            # 42 publication-grade figures, plots, and charts
│   ├── 02_eda/
│   ├── 03_preprocessing/
│   ├── 04_soh_rul/
│   ├── 05_dataset/
│   ├── 06_lstm_prep/
│   ├── 07_model/
│   ├── 08_diagnosis/
│   ├── 09_quality/
│   ├── 10_calce_prep/
│   └── 11_transfer/
├── models/             # Trained models, metrics, configs
│   ├── nasa_lstm/      # Best NASA LSTM weights, scaler, test predictions
│   └── calce_transfer/ # Fine-tuned CALCE model weights & metrics
├── data/
│   └── final/          # Clean 3MB final datasets (filtered & corrected CSVs)
├── analysis/           # Condition reports, validation logs, integrity audits
├── scripts/            # Python scripts for reproducible execution
└── .gitignore          # Excludes ~3.33 GB of intermediate data; tracks 27.4 MB
```

---

## 🚀 Quick Start

1. Install dependencies:
   ```bash
   pip install torch pandas numpy scikit-learn matplotlib joblib
   ```
2. Train the optimized multi-output LSTM:
   ```bash
   python scripts/train_quality_filtered_lstm.py
   ```
3. Run transfer learning on CALCE:
   ```bash
   python scripts/07_transfer_learning_calce.py
   ```
