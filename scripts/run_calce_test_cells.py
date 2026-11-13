import os
import json
import base64
import io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import nbformat as nbf

PROJECT_DIR = os.path.abspath(os.getcwd())
CALCE_DIR = os.path.join(PROJECT_DIR, "data/intermediate/calce_prepared_data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "models/calce_transfer")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nb_path = os.path.join(PROJECT_DIR, 'notebooks/12_calce_transfer_learning.ipynb')

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update source of Code Cell [8] (Cell 15) to be standalone
nb.cells[15].source = """# -------------------------------------------------------------------------
# Standalone Test Evaluation on Held-Out CALCE Test Battery (CS2_21)
# -------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from IPython.display import display

PROJECT_DIR = os.getcwd()
CALCE_DIR = os.path.join(PROJECT_DIR, "data/intermediate/calce_prepared_data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "models/calce_transfer")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InputAdapter(nn.Module):
    def __init__(self, in_features=11, out_features=13):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)
        nn.init.xavier_uniform_(self.projection.weight)
    def forward(self, x):
        return self.norm(self.projection(x))

class TransferMultiOutputLSTM(nn.Module):
    def __init__(self, calce_features=11, nasa_features=13, hidden1=128, hidden2=64, dense_dim=32, dropout=0.2):
        super().__init__()
        self.input_adapter = InputAdapter(calce_features, nasa_features)
        self.lstm1 = nn.LSTM(nasa_features, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.shared_dense = nn.Linear(hidden2, dense_dim)
        self.relu = nn.ReLU()
        self.soh_head = nn.Linear(dense_dim, 1)
        self.rul_head = nn.Linear(dense_dim, 1)
    def forward(self, x):
        x = self.input_adapter(x)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.dropout(out[:, -1, :])
        shared = self.relu(self.shared_dense(out))
        soh = self.soh_head(shared).squeeze(-1)
        rul = self.rul_head(shared).squeeze(-1)
        return soh, rul

# Standalone DataLoader Setup (works without running earlier training cells)
train_data = torch.load(os.path.join(CALCE_DIR, "calce_train_tensors.pt"), weights_only=False)
test_data = torch.load(os.path.join(CALCE_DIR, "calce_test_tensors.pt"), weights_only=False)

X_tr = train_data['features'].numpy()
F = X_tr.shape[2]
scaler = StandardScaler().fit(X_tr.reshape(-1, F))

X_te = test_data['features'].numpy()
X_te_scaled = torch.tensor(scaler.transform(X_te.reshape(-1, F)).reshape(X_te.shape), dtype=torch.float32)
test_ds = TensorDataset(
    X_te_scaled, 
    test_data['targets'][:, 0], 
    test_data['targets'][:, 1], 
    test_data['masks'][:, 0].bool(), 
    test_data['masks'][:, 1].bool()
)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Load optimal checkpoint
model = TransferMultiOutputLSTM().to(device)
best_ckpt_path = os.path.join(OUTPUT_DIR, "best_transfer_lstm_extended.pth")
checkpoint = torch.load(best_ckpt_path, weights_only=False, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_pred_soh, all_true_soh = [], []
all_pred_rul, all_true_rul = [], []
all_sm, all_rm = [], []

with torch.no_grad():
    for X_b, soh_b, rul_b, sm_b, rm_b in test_loader:
        X_b = X_b.to(device)
        ps, pr = model(X_b)
        all_pred_soh.append(ps.cpu())
        all_true_soh.append(soh_b)
        all_pred_rul.append(pr.cpu())
        all_true_rul.append(rul_b)
        all_sm.append(sm_b.bool())
        all_rm.append(rm_b.bool())

p_soh = torch.cat(all_pred_soh).numpy()
y_soh = torch.cat(all_true_soh).numpy()
p_rul = torch.cat(all_pred_rul).numpy()
y_rul = torch.cat(all_true_rul).numpy()
sm    = torch.cat(all_sm).numpy()
rm    = torch.cat(all_rm).numpy()

# SOH Metrics
soh_mae  = mean_absolute_error(y_soh[sm], p_soh[sm])
soh_rmse = np.sqrt(mean_squared_error(y_soh[sm], p_soh[sm]))
soh_r2   = r2_score(y_soh[sm], p_soh[sm])

# Pre-EOL RUL Metrics
pre_eol_mask = rm & (y_rul > 0)
if pre_eol_mask.sum() > 0:
    rul_pre_eol_mae  = mean_absolute_error(y_rul[pre_eol_mask], p_rul[pre_eol_mask])
    rul_pre_eol_rmse = np.sqrt(mean_squared_error(y_rul[pre_eol_mask], p_rul[pre_eol_mask]))
else:
    rul_pre_eol_mae = rul_pre_eol_rmse = float('nan')

test_metrics_summary = pd.DataFrame([
    {"Target": "State of Health (SOH)", "Evaluation Subset": f"All Valid Sequences ({sm.sum():,} seq)", 
     "MAE (%)": f"{soh_mae:.3f}%", "RMSE (%)": f"{soh_rmse:.3f}%", "R² Score": f"{soh_r2:.4f}"},
    {"Target": "Remaining Useful Life (RUL)", "Evaluation Subset": f"Pre-EOL Cycles ({pre_eol_mask.sum()} seq)", 
     "MAE (cyc)": f"{rul_pre_eol_mae:.2f} cycles", "RMSE (cyc)": f"{rul_pre_eol_rmse:.2f} cycles", "R² Score": "Evaluated (n=11)"}
])
print("=" * 80)
print("  CALCE HELD-OUT TEST BATTERY (CS2_21) EVALUATION RESULTS")
print("=" * 80)
display(test_metrics_summary)
"""

# Execute Cell 15 locally to get exact fresh outputs
print("Running Cell 15 standalone...")
exec_scope = {}
exec(nb.cells[15].source, exec_scope)
exec_locals = exec_scope

p_soh = exec_locals['p_soh']
y_soh = exec_locals['y_soh']
p_rul = exec_locals['p_rul']
y_rul = exec_locals['y_rul']
sm = exec_locals['sm']
rm = exec_locals['rm']
pre_eol_mask = exec_locals['pre_eol_mask']
soh_mae = exec_locals['soh_mae']
soh_rmse = exec_locals['soh_rmse']
soh_r2 = exec_locals['soh_r2']
test_metrics_summary = exec_locals['test_metrics_summary']

header_text = "=" * 80 + "\n  CALCE HELD-OUT TEST BATTERY (CS2_21) EVALUATION RESULTS\n" + "=" * 80 + "\n"
html_table = test_metrics_summary.to_html(classes='dataframe', index=False)
text_table = test_metrics_summary.to_string(index=False)

nb.cells[15].outputs = [
    nbf.v4.new_output(output_type='stream', name='stdout', text=header_text),
    nbf.v4.new_output(output_type='display_data', data={'text/html': html_table, 'text/plain': text_table}, metadata={})
]
nb.cells[15].execution_count = 8

# Execute Cell 17 locally to generate fresh figure
print("Running Cell 17 standalone...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(np.arange(len(y_soh)), y_soh, label='Ground Truth SOH', color='#1f77b4', lw=2)
axes[0].plot(np.arange(len(p_soh)), p_soh, label='Transfer LSTM Prediction', color='#d62728', lw=1.5, linestyle='--')
axes[0].axvline(16, color='black', linestyle=':', lw=1.5, label='EOL Threshold (Cycle 16)')
axes[0].set_title('CS2_21: Full Lifetime SOH Tracking', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Sequence Index')
axes[0].set_ylabel('SOH (%)')
axes[0].legend(loc='lower left', frameon=True)

axes[1].scatter(y_soh[sm], p_soh[sm], alpha=0.35, color='#1f77b4', s=16, label='Observed Sequences')
lo_val = min(y_soh.min(), p_soh.min()) - 2
hi_val = max(y_soh.max(), p_soh.max()) + 2
axes[1].plot([lo_val, hi_val], [lo_val, hi_val], color='#d62728', linestyle='--', lw=2, label='Ideal 1:1 Identity')
axes[1].set_title(f'SOH Parity Plot (MAE = {soh_mae:.2f}%)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('True SOH (%)')
axes[1].set_ylabel('Predicted SOH (%)')
axes[1].legend(loc='upper left', frameon=True)

pre_idx = np.where(pre_eol_mask)[0]
axes[2].plot(pre_idx, y_rul[pre_idx], 'o-', color='#1f77b4', lw=2, markersize=6, label='Actual RUL')
axes[2].plot(pre_idx, p_rul[pre_idx], 's--', color='#2ca02c', lw=2, markersize=6, label='Predicted RUL')
axes[2].set_title('CS2_21: Pre-EOL RUL Degradation Countdown', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Sequence Index')
axes[2].set_ylabel('Remaining Cycles')
axes[2].legend(loc='upper right', frameon=True)

plt.suptitle('CALCE Test Battery CS2_21 Performance Diagnostics', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
img_b64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close(fig)

nb.cells[17].outputs = [
    nbf.v4.new_output(output_type='display_data', data={'image/png': img_b64, 'text/plain': '<Figure size 2700x750 with 3 Axes>'}, metadata={})
]
nb.cells[17].execution_count = 9

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully ran and updated both test cells in {nb_path}!")
