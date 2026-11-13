import json

nb_path = 'notebooks/10_optimized_multi_task_lstm_training.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][0]['source'] = [
    "# Multi-Task LSTM: SOH & RUL Prognostics\n",
    "\n",
    "- **Goal**: Simultaneous estimation of State of Health (SOH) and Remaining Useful Life (RUL).\n",
    "- **Dataset**: 32 quality-filtered NASA cells (6 corrupted excluded) split 22 Train / 6 Val / 4 Test.\n",
    "- **Setup**: Multi-task loss ($\\\\lambda = 0.01$), right-censoring masking, and feature standardization."
]

nb['cells'][2]['source'] = [
    "### 1. Data Split & Exclusion Audit\n",
    "- **Retained (32 cells)**: Condition-stratified across 4 °C, 24 °C, and 43 °C regimes.\n",
    "- **Excluded (6 cells)**: Removed due to sensor dropouts or distorted reference capacities."
]

nb['cells'][4]['source'] = [
    "### 2. Multi-Task Training Dynamics\n",
    "- **Objective**: $\\\\mathcal{L}_{\\\\text{total}} = \\\\mathcal{L}_{\\\\text{SOH}} + 0.01 \\\\cdot \\\\mathcal{L}_{\\\\text{RUL}}$.\n",
    "- **Optimizer**: Adam ($lr = 10^{-3}$, weight decay $= 10^{-4}$) with early stopping on validation loss."
]

nb['cells'][6]['source'] = [
    "### 3. Training & Validation Loss Curves\n",
    "- Smooth convergence without gradient vanishing or oscillation."
]

nb['cells'][8]['source'] = [
    "### 4. Test Set Evaluation\n",
    "- Evaluated on 4 held-out cells (`Battery_04`, `Battery_16`, `Battery_30`, `Battery_38`) across room, cold, and elevated temperatures."
]

nb['cells'][10]['source'] = [
    "### 5. Degradation Trajectories & Parity Diagnostics\n",
    "- Predicted vs. actual SOH fade and RUL countdown trajectories."
]

nb['cells'][12]['source'] = [
    "### 6. Summary of Results\n",
    "- **Room Temp (`Battery_04`)**: SOH MAE = **2.66%** ($R^2 = 0.8627$), RUL MAE = **7.06 cycles** ($R^2 = 0.8195$).\n",
    "- **Cold Chamber (`Battery_30`)**: SOH MAE = **1.97%** ($R^2 = 0.7583$), RUL MAE = **3.85 cycles**.\n",
    "- **Overall Active RUL**: MAE = **6.39 cycles** ($R^2 = 0.8586$).\n",
    "- **CALCE Transfer (`CS2_21`)**: SOH MAE = **4.18%** ($R^2 = 0.9416$), RUL MAE = **5.02 cycles** with 98.15% frozen weights."
]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated markdowns in Notebook 10 successfully.")
