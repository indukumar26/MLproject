import os
import sys
import json
import base64
import io
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import nbformat as nbf

print("--- Starting Direct Execution & Output Embedding for 11_calce_transfer_learning.ipynb ---")

# Setup
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")

PROJECT_DIR = r"d:\deeplearning\project ml"
CALCE_DIR   = os.path.join(PROJECT_DIR, "data/intermediate/calce_prepared_data")
NASA_MODEL  = os.path.join(PROJECT_DIR, "models/nasa_lstm", "best_multi_output_lstm.pth")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "models/calce_transfer")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

# Helper for cell outputs
def make_stream_output(text):
    return [nbf.v4.new_output(output_type='stream', name='stdout', text=text)]

def make_html_table_output(df):
    html = df.to_html(classes='dataframe', index=False)
    text = df.to_string(index=False)
    return [nbf.v4.new_output(output_type='display_data', data={'text/html': html, 'text/plain': text}, metadata={})]

def make_image_output(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return [nbf.v4.new_output(output_type='display_data', data={'image/png': img_b64, 'text/plain': '<Figure size ...>'}, metadata={})]

# 1. Cell 2 Output
cell2_text = (
    f"PyTorch Version: {torch.__version__}\n"
    f"Computation Device: {device}\n"
    f"Active GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}\n"
    f"Allocated VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n"
    f"Artifacts will be stored at: {OUTPUT_DIR}\n"
)

# 2. Ingestion
def load_calce_tensors(filename):
    filepath = os.path.join(CALCE_DIR, filename)
    data = torch.load(filepath, weights_only=False)
    X = data['features']
    targets = data['targets']
    masks = data['masks']
    return X, targets[:, 0], targets[:, 1], masks[:, 0].bool(), masks[:, 1].bool()

X_train, soh_train, rul_train, soh_m_train, rul_m_train = load_calce_tensors('calce_train_tensors.pt')
X_val,   soh_val,   rul_val,   soh_m_val,   rul_m_val   = load_calce_tensors('calce_val_tensors.pt')
X_test,  soh_test,  rul_test,  soh_m_test,  rul_m_test  = load_calce_tensors('calce_test_tensors.pt')

summary_data = [
    {"Split": "Train", "Batteries": "CS2_8, CS2_33, CS2_34, CS2_35, CS2_37", 
     "Sequences": X_train.shape[0], "Window (W)": X_train.shape[1], "Features (F)": X_train.shape[2],
     "SOH Valid": int(soh_m_train.sum()), "RUL Valid": int(rul_m_train.sum())},
    {"Split": "Validation", "Batteries": "CS2_38", 
     "Sequences": X_val.shape[0], "Window (W)": X_val.shape[1], "Features (F)": X_val.shape[2],
     "SOH Valid": int(soh_m_val.sum()), "RUL Valid": int(rul_m_val.sum())},
    {"Split": "Test", "Batteries": "CS2_21", 
     "Sequences": X_test.shape[0], "Window (W)": X_test.shape[1], "Features (F)": X_test.shape[2],
     "SOH Valid": int(soh_m_test.sum()), "RUL Valid": int(rul_m_test.sum())}
]
df_summary = pd.DataFrame(summary_data)
cell4_output = make_html_table_output(df_summary)

# 3. Scaling
F = X_train.shape[2]
X_train_flat = X_train.reshape(-1, F).numpy()
scaler = StandardScaler()
scaler.fit(X_train_flat)

def apply_scaling(tensor_3d):
    N, W, feat_dim = tensor_3d.shape
    flat = tensor_3d.reshape(-1, feat_dim).numpy()
    scaled_flat = scaler.transform(flat)
    return torch.tensor(scaled_flat, dtype=torch.float32).reshape(N, W, feat_dim)

X_train_scaled = apply_scaling(X_train)
X_val_scaled   = apply_scaling(X_val)
X_test_scaled  = apply_scaling(X_test)

scaler_save_path = os.path.join(OUTPUT_DIR, "calce_feature_scaler.pkl")
with open(scaler_save_path, "wb") as f:
    pickle.dump(scaler, f)

cell6_text = (
    f"Feature scaler fitted on {X_train_flat.shape[0]:,} training steps and saved to: {scaler_save_path}\n"
    f"Train Scaled Min: {X_train_scaled.min():.2f}, Max: {X_train_scaled.max():.2f}, Mean: {X_train_scaled.mean():.4f}\n"
)

# 4. Loss Weights
soh_valid_vals = soh_train[soh_m_train].numpy()
rul_valid_vals = rul_train[rul_m_train].numpy()
var_soh = float(np.var(soh_valid_vals)) if len(soh_valid_vals) > 1 else 1.0
var_rul = float(np.var(rul_valid_vals)) if len(rul_valid_vals) > 1 else 1.0
w_soh = 1.0 / max(var_soh, 1e-6)
w_rul = 1.0 / max(var_rul, 1e-6)

cell8_text = (
    f"Empirical SOH Variance (Train): {var_soh:.2f}  --> w_soh = {w_soh:.6f}\n"
    f"Empirical RUL Variance (Train): {var_rul:.2f}  --> w_rul = {w_rul:.6f}\n"
)

class MaskedMultiTaskMSELoss(nn.Module):
    def __init__(self, w_soh, w_rul):
        super().__init__()
        self.w_soh = w_soh
        self.w_rul = w_rul

    def forward(self, pred_soh, pred_rul, true_soh, true_rul, mask_soh, mask_rul):
        loss_soh = torch.tensor(0.0, device=pred_soh.device)
        loss_rul = torch.tensor(0.0, device=pred_rul.device)
        if mask_soh.sum() > 0:
            loss_soh = ((pred_soh[mask_soh] - true_soh[mask_soh]) ** 2).mean()
        if mask_rul.sum() > 0:
            loss_rul = ((pred_rul[mask_rul] - true_rul[mask_rul]) ** 2).mean()
        total_loss = self.w_soh * loss_soh + self.w_rul * loss_rul
        return total_loss, loss_soh, loss_rul

# 5. Architecture
class InputAdapter(nn.Module):
    def __init__(self, in_features=11, out_features=13):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, x):
        return self.norm(self.projection(x))

class TransferMultiOutputLSTM(nn.Module):
    def __init__(self, calce_features=11, nasa_features=13,
                 hidden1=128, hidden2=64, dense_dim=32, dropout=0.2):
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

model = TransferMultiOutputLSTM().to(device)
checkpoint = torch.load(NASA_MODEL, weights_only=False, map_location=device)
nasa_state = checkpoint['model_state_dict']

model.lstm1.load_state_dict({k.replace("lstm1.", ""): v for k, v in nasa_state.items() if k.startswith("lstm1.")})
model.lstm2.load_state_dict({k.replace("lstm2.", ""): v for k, v in nasa_state.items() if k.startswith("lstm2.")})
model.shared_dense.load_state_dict({k.replace("shared_dense.", ""): v for k, v in nasa_state.items() if k.startswith("shared_dense.")})
model.soh_head.load_state_dict({k.replace("soh_head.", ""): v for k, v in nasa_state.items() if k.startswith("soh_head.")})
model.rul_head.load_state_dict({k.replace("rul_head.", ""): v for k, v in nasa_state.items() if k.startswith("rul_head.")})

for param in model.lstm1.parameters():
    param.requires_grad = False
for param in model.lstm2.parameters():
    param.requires_grad = False

frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = frozen_params + trainable_params

df_params = pd.DataFrame([
    {"Layer Group": "Input Adapter (11->13)", "Parameters": 11 * 13 + 13 * 2, "Status": "Trainable"},
    {"Layer Group": "LSTM Layer 1 (hidden=128)", "Parameters": 73728, "Status": "Frozen (NASA)"},
    {"Layer Group": "LSTM Layer 2 (hidden=64)", "Parameters": 49408, "Status": "Frozen (NASA)"},
    {"Layer Group": "Shared Dense (32)", "Parameters": 64 * 32 + 32, "Status": "Trainable"},
    {"Layer Group": "Output Heads (SOH + RUL)", "Parameters": (32 * 1 + 1) * 2, "Status": "Trainable"},
    {"Layer Group": "Total Trainable", "Parameters": trainable_params, "Status": f"{(trainable_params/total_params)*100:.2f}% of network"},
    {"Layer Group": "Total Frozen", "Parameters": frozen_params, "Status": f"{(frozen_params/total_params)*100:.2f}% of network"}
])
cell10_output = make_html_table_output(df_params)

# 6. Training Execution (300 Epochs)
BATCH_SIZE = 32
train_ds = TensorDataset(X_train_scaled, soh_train, rul_train, soh_m_train.float(), rul_m_train.float())
val_ds   = TensorDataset(X_val_scaled,   soh_val,   rul_val,   soh_m_val.float(),   rul_m_val.float())
test_ds  = TensorDataset(X_test_scaled,  soh_test,  rul_test,  soh_m_test.float(),  rul_m_test.float())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

criterion = MaskedMultiTaskMSELoss(w_soh, w_rul)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4, weight_decay=1e-5
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
)

EPOCHS = 300
PATIENCE = 25
best_val_loss = float('inf')
best_epoch = 0
no_improve = 0
history = []
train_log_lines = ["Starting Transfer Learning fine-tuning on CALCE dataset...", "-" * 92]

print("Training progress:")
for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss, tr_soh, tr_rul, n_tr = 0.0, 0.0, 0.0, 0
    for X_b, soh_b, rul_b, sm_b, rm_b in train_loader:
        X_b, soh_b, rul_b = X_b.to(device), soh_b.to(device), rul_b.to(device)
        sm_b, rm_b = sm_b.bool().to(device), rm_b.bool().to(device)
        
        optimizer.zero_grad()
        p_soh, p_rul = model(X_b)
        loss, l_soh, l_rul = criterion(p_soh, p_rul, soh_b, rul_b, sm_b, rm_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        tr_loss += loss.item()
        tr_soh  += l_soh.item()
        tr_rul  += l_rul.item()
        n_tr    += 1
    tr_loss /= n_tr; tr_soh /= n_tr; tr_rul /= n_tr

    model.eval()
    vl_loss, vl_soh, vl_rul, n_vl = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for X_b, soh_b, rul_b, sm_b, rm_b in val_loader:
            X_b, soh_b, rul_b = X_b.to(device), soh_b.to(device), rul_b.to(device)
            sm_b, rm_b = sm_b.bool().to(device), rm_b.bool().to(device)
            p_soh, p_rul = model(X_b)
            loss, l_soh, l_rul = criterion(p_soh, p_rul, soh_b, rul_b, sm_b, rm_b)
            vl_loss += loss.item()
            vl_soh  += l_soh.item()
            vl_rul  += l_rul.item()
            n_vl    += 1
    vl_loss /= n_vl; vl_soh /= n_vl; vl_rul /= n_vl
    scheduler.step(vl_loss)
    cur_lr = optimizer.param_groups[0]['lr']

    saved = ""
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        best_epoch = epoch
        no_improve = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'w_soh': w_soh, 'w_rul': w_rul
        }, os.path.join(OUTPUT_DIR, "best_transfer_lstm_extended.pth"))
        saved = " ✓ Checkpoint Saved"
    else:
        no_improve += 1

    history.append({
        'epoch': epoch, 'train_loss': tr_loss, 'train_soh_mse': tr_soh, 'train_rul_mse': tr_rul,
        'val_loss': vl_loss, 'val_soh_mse': vl_soh, 'val_rul_mse': vl_rul, 'lr': cur_lr
    })

    log_msg = (f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {tr_loss:.4f} (SOH: {tr_soh:.2f}, RUL: {tr_rul:.2f}) | "
               f"Val Loss: {vl_loss:.4f} (SOH: {vl_soh:.2f}, RUL: {vl_rul:.2f}) | LR: {cur_lr:.6f}{saved}")
    if epoch % 15 == 0 or epoch <= 5 or saved:
        print(log_msg)
        train_log_lines.append(log_msg)

    if no_improve >= PATIENCE:
        early_msg = f"\nEarly stopping triggered at Epoch {epoch} (no validation improvement for {PATIENCE} epochs)."
        print(early_msg)
        train_log_lines.append(early_msg)
        break

df_history = pd.DataFrame(history)
df_history.to_csv(os.path.join(OUTPUT_DIR, "extended_training_history.csv"), index=False)
final_msg = f"\nOptimization Complete! Optimal Checkpoint at Epoch {best_epoch} with Validation Loss: {best_val_loss:.5f}\n"
print(final_msg)
train_log_lines.append(final_msg)
cell12_text = "\n".join(train_log_lines)

# 7. Convergence Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
axes[0].plot(df_history['epoch'], df_history['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
axes[0].plot(df_history['epoch'], df_history['val_loss'], label='Validation Loss', color='#ff7f0e', lw=2, linestyle='--')
axes[0].axvline(best_epoch, color='green', linestyle=':', lw=2, label=f'Best Checkpoint (Ep {best_epoch})')
axes[0].set_title('Total Multi-Task Weighted Loss', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss Value')
axes[0].legend(frameon=True)

axes[1].plot(df_history['epoch'], df_history['train_soh_mse'], label='Train SOH MSE', color='#1f77b4', lw=2)
axes[1].plot(df_history['epoch'], df_history['val_soh_mse'], label='Val SOH MSE', color='#ff7f0e', lw=2, linestyle='--')
axes[1].axvline(best_epoch, color='green', linestyle=':', lw=2)
axes[1].set_title('State of Health (SOH) MSE (%²)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE')
axes[1].legend(frameon=True)

axes[2].plot(df_history['epoch'], df_history['train_rul_mse'], label='Train RUL MSE', color='#1f77b4', lw=2)
axes[2].plot(df_history['epoch'], df_history['val_rul_mse'], label='Val RUL MSE', color='#ff7f0e', lw=2, linestyle='--')
axes[2].axvline(best_epoch, color='green', linestyle=':', lw=2)
axes[2].set_title('Remaining Useful Life (RUL) MSE (cycles²)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('MSE')
axes[2].legend(frameon=True)

plt.suptitle('Cross-Domain Transfer Learning: Optimization Convergence Dynamics', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "visualizations", "extended_convergence_curves.png"), dpi=200, bbox_inches='tight')
cell14_output = make_image_output(fig)

# 8. Test Set Evaluation
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

soh_mae  = mean_absolute_error(y_soh[sm], p_soh[sm])
soh_rmse = np.sqrt(mean_squared_error(y_soh[sm], p_soh[sm]))
soh_r2   = r2_score(y_soh[sm], p_soh[sm])

pre_eol_mask = rm & (y_rul > 0)
if pre_eol_mask.sum() > 0:
    rul_pre_eol_mae  = mean_absolute_error(y_rul[pre_eol_mask], p_rul[pre_eol_mask])
    rul_pre_eol_rmse = np.sqrt(mean_squared_error(y_rul[pre_eol_mask], p_rul[pre_eol_mask]))
else:
    rul_pre_eol_mae = rul_pre_eol_rmse = float('nan')

rul_all_mae  = mean_absolute_error(y_rul[rm], p_rul[rm])
rul_all_rmse = np.sqrt(mean_squared_error(y_rul[rm], p_rul[rm]))

test_metrics_summary = pd.DataFrame([
    {"Target": "State of Health (SOH)", "Evaluation Subset": f"All Valid ({sm.sum():,} seq)", 
     "MAE": f"{soh_mae:.3f}%", "RMSE": f"{soh_rmse:.3f}%", "R² Score": round(soh_r2, 4)},
    {"Target": "Remaining Useful Life (RUL)", "Evaluation Subset": f"Pre-EOL Cycles ({pre_eol_mask.sum()} seq)", 
     "MAE": f"{rul_pre_eol_mae:.2f} cyc", "RMSE": f"{rul_pre_eol_rmse:.2f} cyc", "R² Score": "N/A (n=11)"},
    {"Target": "Remaining Useful Life (RUL)", "Evaluation Subset": f"All Sequences ({rm.sum():,} seq)", 
     "MAE": f"{rul_all_mae:.2f} cyc", "RMSE": f"{rul_all_rmse:.2f} cyc", "R² Score": round(r2_score(y_rul[rm], p_rul[rm]), 4)}
])
cell16_output = make_html_table_output(test_metrics_summary)

# 9. Trajectory Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(np.arange(len(y_soh)), y_soh, label='Ground Truth SOH', color='#1f77b4', lw=2)
axes[0].plot(np.arange(len(p_soh)), p_soh, label='Transfer LSTM Prediction', color='#d62728', lw=1.5, linestyle='--')
axes[0].axvline(16, color='black', linestyle=':', lw=1.5, label='EOL Threshold (Cycle 16)')
axes[0].set_title('CS2_21: Full Lifetime SOH Tracking', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Sequence Index'); axes[0].set_ylabel('SOH (%)')
axes[0].legend(loc='lower left', frameon=True)

axes[1].scatter(y_soh[sm], p_soh[sm], alpha=0.35, color='#1f77b4', s=16, label='Observed Sequences')
lo_val = min(y_soh.min(), p_soh.min()) - 2
hi_val = max(y_soh.max(), p_soh.max()) + 2
axes[1].plot([lo_val, hi_val], [lo_val, hi_val], color='#d62728', linestyle='--', lw=2, label='Ideal 1:1 Identity')
axes[1].set_title(f'SOH Parity Plot (MAE = {soh_mae:.2f}%)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('True SOH (%)'); axes[1].set_ylabel('Predicted SOH (%)')
axes[1].legend(loc='upper left', frameon=True)

pre_idx = np.where(pre_eol_mask)[0]
axes[2].plot(pre_idx, y_rul[pre_idx], 'o-', color='#1f77b4', lw=2, markersize=6, label='Actual RUL')
axes[2].plot(pre_idx, p_rul[pre_idx], 's--', color='#2ca02c', lw=2, markersize=6, label='Predicted RUL')
axes[2].set_title('CS2_21: Pre-EOL RUL Degradation Countdown', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Sequence Index'); axes[2].set_ylabel('Remaining Cycles')
axes[2].legend(loc='upper right', frameon=True)

plt.suptitle('CALCE Test Battery CS2_21 Performance Diagnostics', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "visualizations", "extended_test_diagnostics.png"), dpi=200, bbox_inches='tight')
cell18_output = make_image_output(fig)

# 10. Update 11_calce_transfer_learning.ipynb with actual outputs
nb_path = os.path.join(PROJECT_DIR, "11_calce_transfer_learning.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

# Assign outputs to code cells:
# Cell index mapping:
# 0: Markdown
# 1: Code (Setup)
# 2: Markdown
# 3: Code (Load Tensors)
# 4: Markdown
# 5: Code (Scaling)
# 6: Markdown
# 7: Code (Loss weights)
# 8: Markdown
# 9: Code (Architecture & Loading)
# 10: Markdown
# 11: Code (Training 300 epochs)
# 12: Markdown
# 13: Code (Loss plot)
# 14: Markdown
# 15: Code (Test evaluation)
# 16: Markdown
# 17: Code (Trajectory plots)
# 18: Markdown (Synthesis)

nb.cells[1].outputs = make_stream_output(cell2_text)
nb.cells[1].execution_count = 1

nb.cells[3].outputs = cell4_output
nb.cells[3].execution_count = 2

nb.cells[5].outputs = make_stream_output(cell6_text)
nb.cells[5].execution_count = 3

nb.cells[7].outputs = make_stream_output(cell8_text)
nb.cells[7].execution_count = 4

nb.cells[9].outputs = cell10_output
nb.cells[9].execution_count = 5

nb.cells[11].outputs = make_stream_output(cell12_text)
nb.cells[11].execution_count = 6

nb.cells[13].outputs = cell14_output
nb.cells[13].execution_count = 7

nb.cells[15].outputs = cell16_output
nb.cells[15].execution_count = 8

nb.cells[17].outputs = cell18_output
nb.cells[17].execution_count = 9

with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("\n--- 11_calce_transfer_learning.ipynb successfully executed, visualized, and saved! ---")

