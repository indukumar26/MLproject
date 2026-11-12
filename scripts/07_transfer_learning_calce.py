"""
Transfer Learning: NASA LSTM → CALCE Battery Dataset
======================================================
Strategy: Input Adapter (11 → 13 features) + Frozen LSTM Core + Fine-tuned Output Heads

Architecture:
  [CALCE 11-feat input]
       ↓  (learnable, randomly initialized)
  InputAdapter (Linear 11 → 13, no bias, LayerNorm)
       ↓  (FROZEN weights from NASA pre-training)
  lstm1  (128 hidden)  ← FROZEN
  lstm2  (64 hidden)   ← FROZEN
       ↓  (fine-tuned)
  shared_dense (32, ReLU)
  soh_head (1)
  rul_head (1)

Fine-tuned layers: InputAdapter, shared_dense, soh_head, rul_head
Frozen layers:     lstm1, lstm2
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR   = r"d:\deeplearning\project ml"
CALCE_DIR     = os.path.join(PROJECT_DIR, "data/intermediate/calce_prepared_data")
NASA_MODEL    = os.path.join(PROJECT_DIR, "models/nasa_lstm", "best_multi_output_lstm.pth")
OUTPUT_DIR    = os.path.join(PROJECT_DIR, "models/calce_transfer")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

CALCE_FEATURES   = 11      # CALCE has 11 compatible features
NASA_FEATURES    = 14      # NASA model input size (14 features with cumulative Ah)
SEQ_LEN          = 5
HIDDEN1          = 128
HIDDEN2          = 64
DENSE_DIM        = 32
DROPOUT          = 0.2

BATCH_SIZE       = 32
LEARNING_RATE    = 1e-4    # Low LR for fine-tuning
WEIGHT_DECAY     = 1e-5
EPOCHS           = 150
EARLY_STOP_PAT   = 20
LR_PATIENCE      = 8
LR_FACTOR        = 0.5
MIN_LR           = 1e-6
SEED             = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load CALCE tensors
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading CALCE tensors ...")

def load_split(fname):
    d = torch.load(os.path.join(CALCE_DIR, fname), weights_only=False)
    X    = d['features']          # (N, 5, 11)
    tgt  = d['targets']           # (N, 2)  col0=SOH, col1=RUL
    mask = d['masks']             # (N, 2)  col0=soh_mask, col1=rul_mask
    return X, tgt[:, 0], tgt[:, 1], mask[:, 0].bool(), mask[:, 1].bool()

X_train, soh_train, rul_train, soh_mask_train, rul_mask_train = load_split("calce_train_tensors.pt")
X_val,   soh_val,   rul_val,   soh_mask_val,   rul_mask_val   = load_split("calce_val_tensors.pt")
X_test,  soh_test,  rul_test,  soh_mask_test,  rul_mask_test  = load_split("calce_test_tensors.pt")

print(f"  Train: {X_train.shape}  SOH valid={soh_mask_train.sum().item()}  RUL valid={rul_mask_train.sum().item()}")
print(f"  Val:   {X_val.shape}    SOH valid={soh_mask_val.sum().item()}    RUL valid={rul_mask_val.sum().item()}")
print(f"  Test:  {X_test.shape}   SOH valid={soh_mask_test.sum().item()}   RUL valid={rul_mask_test.sum().item()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fit feature scaler on CALCE training data only
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Fitting CALCE feature scaler on training data only ...")
X_train_2d = X_train.reshape(-1, CALCE_FEATURES).numpy()
calce_scaler = StandardScaler()
calce_scaler.fit(X_train_2d)

def scale_split(X):
    N, S, F = X.shape
    X_np = X.reshape(-1, F).numpy()
    X_sc = calce_scaler.transform(X_np)
    return torch.tensor(X_sc, dtype=torch.float32).reshape(N, S, F)

X_train_sc = scale_split(X_train)
X_val_sc   = scale_split(X_val)
X_test_sc  = scale_split(X_test)

scaler_path = os.path.join(OUTPUT_DIR, "calce_feature_scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(calce_scaler, f)
print(f"  Scaler saved → {scaler_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Compute variance-normalized loss weights from CALCE training targets
# ─────────────────────────────────────────────────────────────────────────────
soh_valid_train = soh_train[soh_mask_train].numpy()
rul_valid_train = rul_train[rul_mask_train].numpy()
var_soh = float(np.var(soh_valid_train)) if len(soh_valid_train) > 1 else 1.0
var_rul = float(np.var(rul_valid_train)) if len(rul_valid_train) > 1 else 1.0
w_soh = 1.0 / max(var_soh, 1e-6)
w_rul = 1.0 / max(var_rul, 1e-6)
print(f"\n[3] CALCE Loss weights → w_soh={w_soh:.6f} (var={var_soh:.2f}), w_rul={w_rul:.6f} (var={var_rul:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model Definition
# ─────────────────────────────────────────────────────────────────────────────
class InputAdapter(nn.Module):
    """Learnable linear projection from CALCE 11-feature space to NASA 13-feature space."""
    def __init__(self, in_features=11, out_features=13):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, x):
        # x: (B, seq_len, 11)
        projected = self.projection(x)   # → (B, seq_len, 13)
        return self.norm(projected)


class TransferMultiOutputLSTM(nn.Module):
    def __init__(self, calce_features=11, nasa_features=13,
                 hidden1=128, hidden2=64, dense_dim=32, dropout=0.2):
        super().__init__()
        # Input adapter (fine-tuned)
        self.input_adapter = InputAdapter(calce_features, nasa_features)
        # LSTM core (frozen — loaded from NASA checkpoint)
        self.lstm1 = nn.LSTM(nasa_features, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # Shared dense + heads (fine-tuned)
        self.shared_dense = nn.Linear(hidden2, dense_dim)
        self.relu = nn.ReLU()
        self.soh_head = nn.Linear(dense_dim, 1)
        self.rul_head = nn.Linear(dense_dim, 1)

    def forward(self, x):
        # x: (B, 5, 11)
        x = self.input_adapter(x)        # (B, 5, 13)
        out, _ = self.lstm1(x)           # (B, 5, 128)
        out, _ = self.lstm2(out)         # (B, 5, 64)
        out = self.dropout(out[:, -1, :])  # (B, 64) — last timestep
        shared = self.relu(self.shared_dense(out))  # (B, 32)
        soh = self.soh_head(shared).squeeze(-1)      # (B,)
        rul = self.rul_head(shared).squeeze(-1)      # (B,)
        return soh, rul


# ─────────────────────────────────────────────────────────────────────────────
# 5. Load NASA weights and freeze LSTM core
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Loading NASA pre-trained weights ...")
model = TransferMultiOutputLSTM(
    calce_features=CALCE_FEATURES,
    nasa_features=NASA_FEATURES,
    hidden1=HIDDEN1, hidden2=HIDDEN2,
    dense_dim=DENSE_DIM, dropout=DROPOUT
).to(device)

checkpoint = torch.load(NASA_MODEL, weights_only=False, map_location=device)
nasa_state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

# Load NASA weights into LSTM core + shared layers + heads
# (input_adapter is randomly initialized — this is intentional)
model.lstm1.load_state_dict({
    k.replace("lstm1.", ""): v for k, v in nasa_state.items() if k.startswith("lstm1.")
})
model.lstm2.load_state_dict({
    k.replace("lstm2.", ""): v for k, v in nasa_state.items() if k.startswith("lstm2.")
})
model.shared_dense.load_state_dict({
    k.replace("shared_dense.", ""): v for k, v in nasa_state.items() if k.startswith("shared_dense.")
})
model.soh_head.load_state_dict({
    k.replace("soh_head.", ""): v for k, v in nasa_state.items() if k.startswith("soh_head.")
})
model.rul_head.load_state_dict({
    k.replace("rul_head.", ""): v for k, v in nasa_state.items() if k.startswith("rul_head.")
})
print("  NASA weights loaded into: lstm1, lstm2, shared_dense, soh_head, rul_head")

# Freeze LSTM core
for param in model.lstm1.parameters():
    param.requires_grad = False
for param in model.lstm2.parameters():
    param.requires_grad = False

frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Frozen parameters:    {frozen_params:,}  (lstm1 + lstm2)")
print(f"  Trainable parameters: {trainable_params:,}  (adapter + dense + heads)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Loss Function
# ─────────────────────────────────────────────────────────────────────────────
class MaskedMultiTaskMSELoss(nn.Module):
    def __init__(self, w_soh, w_rul):
        super().__init__()
        self.w_soh = w_soh
        self.w_rul = w_rul

    def forward(self, pred_soh, pred_rul, true_soh, true_rul, mask_soh, mask_rul):
        loss_soh = 0.0
        loss_rul = 0.0
        if mask_soh.sum() > 0:
            loss_soh = ((pred_soh[mask_soh] - true_soh[mask_soh]) ** 2).mean()
        if mask_rul.sum() > 0:
            loss_rul = ((pred_rul[mask_rul] - true_rul[mask_rul]) ** 2).mean()
        return self.w_soh * loss_soh + self.w_rul * loss_rul, loss_soh, loss_rul


# ─────────────────────────────────────────────────────────────────────────────
# 7. DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
def make_loader(X, soh, rul, soh_m, rul_m, shuffle=False):
    ds = TensorDataset(X, soh, rul, soh_m.float(), rul_m.float())
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train_sc, soh_train, rul_train, soh_mask_train, rul_mask_train, shuffle=True)
val_loader   = make_loader(X_val_sc,   soh_val,   rul_val,   soh_mask_val,   rul_mask_val)
test_loader  = make_loader(X_test_sc,  soh_test,  rul_test,  soh_mask_test,  rul_mask_test)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Optimizer and Scheduler
# ─────────────────────────────────────────────────────────────────────────────
criterion = MaskedMultiTaskMSELoss(w_soh, w_rul)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(loader, training=False):
    if training:
        model.train()
    else:
        model.eval()
    total_loss = soh_mse_sum = rul_mse_sum = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_b, soh_b, rul_b, sm_b, rm_b in loader:
            X_b   = X_b.to(device)
            soh_b = soh_b.to(device)
            rul_b = rul_b.to(device)
            sm_b  = sm_b.bool().to(device)
            rm_b  = rm_b.bool().to(device)
            pred_soh, pred_rul = model(X_b)
            loss, l_soh, l_rul = criterion(pred_soh, pred_rul, soh_b, rul_b, sm_b, rm_b)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss   += loss.item()
            soh_mse_sum  += l_soh.item() if isinstance(l_soh, torch.Tensor) else l_soh
            rul_mse_sum  += l_rul.item() if isinstance(l_rul, torch.Tensor) else l_rul
            n_batches    += 1
    return total_loss / n_batches, soh_mse_sum / n_batches, rul_mse_sum / n_batches


print("\n[5] Starting Transfer Learning Fine-Tuning ...")
print(f"    Fine-tuning: InputAdapter + shared_dense + soh_head + rul_head")
print(f"    Frozen:      lstm1 + lstm2")
print(f"    LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Max Epochs={EPOCHS}")
print(f"    Early stopping patience={EARLY_STOP_PAT}")
print("-" * 90)

history = []
best_val_loss = float('inf')
best_epoch    = 0
no_improve    = 0

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_soh, tr_rul = run_epoch(train_loader, training=True)
    vl_loss, vl_soh, vl_rul = run_epoch(val_loader,   training=False)
    scheduler.step(vl_loss)
    cur_lr = optimizer.param_groups[0]['lr']

    saved = ""
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        best_epoch    = epoch
        no_improve    = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'w_soh': w_soh, 'w_rul': w_rul
        }, os.path.join(OUTPUT_DIR, "best_transfer_lstm.pth"))
        saved = " ✓ Saved"
    else:
        no_improve += 1

    history.append({
        'epoch': epoch, 'train_loss': tr_loss,
        'train_soh_mse': tr_soh, 'train_rul_mse': tr_rul,
        'val_loss': vl_loss, 'val_soh_mse': vl_soh,
        'val_rul_mse': vl_rul, 'lr': cur_lr
    })

    if epoch % 10 == 0 or epoch <= 5 or saved:
        print(f"  Ep {epoch:3d} | TrLoss={tr_loss:.4f} TrSOH={tr_soh:.2f} TrRUL={tr_rul:.2f} "
              f"| VlLoss={vl_loss:.4f} VlSOH={vl_soh:.2f} VlRUL={vl_rul:.2f} "
              f"| LR={cur_lr:.6f}{saved}")

    if no_improve >= EARLY_STOP_PAT:
        print(f"\n  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PAT} epochs)")
        break

hist_df = pd.DataFrame(history)
hist_df.to_csv(os.path.join(OUTPUT_DIR, "transfer_training_history.csv"), index=False)
print(f"\n  Best checkpoint: Epoch {best_epoch}  Val Loss={best_val_loss:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Load Best Checkpoint and Evaluate on Test Set
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Evaluating on CALCE Test Set ...")
ckpt = torch.load(os.path.join(OUTPUT_DIR, "best_transfer_lstm.pth"), weights_only=False, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

all_pred_soh, all_true_soh, all_pred_rul, all_true_rul = [], [], [], []
all_sm, all_rm = [], []
meta_test = pd.read_csv(os.path.join(CALCE_DIR, "calce_test_sequences_metadata.csv"))
battery_ids = []

with torch.no_grad():
    for idx, (X_b, soh_b, rul_b, sm_b, rm_b) in enumerate(test_loader):
        X_b = X_b.to(device)
        ps, pr = model(X_b)
        all_pred_soh.append(ps.cpu())
        all_true_soh.append(soh_b)
        all_pred_rul.append(pr.cpu())
        all_true_rul.append(rul_b)
        all_sm.append(sm_b.bool())
        all_rm.append(rm_b.bool())

pred_soh = torch.cat(all_pred_soh).numpy()
true_soh = torch.cat(all_true_soh).numpy()
pred_rul = torch.cat(all_pred_rul).numpy()
true_rul = torch.cat(all_true_rul).numpy()
sm       = torch.cat(all_sm).numpy().astype(bool)
rm       = torch.cat(all_rm).numpy().astype(bool)

# Overall metrics
soh_mae  = mean_absolute_error(true_soh[sm], pred_soh[sm])
soh_rmse = np.sqrt(mean_squared_error(true_soh[sm], pred_soh[sm]))
soh_r2   = r2_score(true_soh[sm], pred_soh[sm])
print(f"  SOH → MAE={soh_mae:.3f}%  RMSE={soh_rmse:.3f}%  R²={soh_r2:.4f}  (n={sm.sum()})")

if rm.sum() > 0:
    rul_mae  = mean_absolute_error(true_rul[rm], pred_rul[rm])
    rul_rmse = np.sqrt(mean_squared_error(true_rul[rm], pred_rul[rm]))
    rul_r2   = r2_score(true_rul[rm], pred_rul[rm])
    print(f"  RUL → MAE={rul_mae:.3f}  RMSE={rul_rmse:.3f}  R²={rul_r2:.4f}  (n={rm.sum()})")
else:
    rul_mae = rul_rmse = rul_r2 = float('nan')
    print(f"  RUL → No valid RUL sequences in test set (CS2_21 EOL at cycle 16)")

# Save predictions
pred_df = pd.DataFrame({
    'sequence_id': meta_test['sequence_id'].values[:len(pred_soh)],
    'Battery_ID':  meta_test['Battery_ID'].values[:len(pred_soh)],
    'true_SOH':    true_soh,
    'pred_SOH':    pred_soh,
    'soh_valid':   sm,
    'true_RUL':    true_rul,
    'pred_RUL':    pred_rul,
    'rul_valid':   rm
})
pred_df.to_csv(os.path.join(OUTPUT_DIR, "transfer_test_predictions.csv"), index=False)

metrics_df = pd.DataFrame([
    {'Target': 'SOH_percent', 'Evaluated_Sequences': int(sm.sum()), 'MAE': round(soh_mae, 4), 'RMSE': round(soh_rmse, 4), 'R2_Score': round(soh_r2, 4)},
    {'Target': 'RUL_cycles',  'Evaluated_Sequences': int(rm.sum()), 'MAE': round(rul_mae, 4), 'RMSE': round(rul_rmse, 4), 'R2_Score': round(rul_r2, 4)},
])
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "transfer_test_metrics.csv"), index=False)
print(f"\n  Predictions → {os.path.join(OUTPUT_DIR, 'transfer_test_predictions.csv')}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Visualizations
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

vis_dir = os.path.join(OUTPUT_DIR, "visualizations")

# --- Training loss curves ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(hist_df['epoch'], hist_df['train_loss'],  label='Train Loss',  color='steelblue')
axes[0].plot(hist_df['epoch'], hist_df['val_loss'],    label='Val Loss',    color='tomato', linestyle='--')
axes[0].axvline(best_epoch, color='green', linestyle=':', label=f'Best (Ep {best_epoch})')
axes[0].set_title('Total Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()

axes[1].plot(hist_df['epoch'], hist_df['train_soh_mse'], label='Train SOH MSE', color='steelblue')
axes[1].plot(hist_df['epoch'], hist_df['val_soh_mse'],   label='Val SOH MSE',   color='tomato', linestyle='--')
axes[1].set_title('SOH MSE'); axes[1].set_xlabel('Epoch'); axes[1].legend()

axes[2].plot(hist_df['epoch'], hist_df['train_rul_mse'], label='Train RUL MSE', color='steelblue')
axes[2].plot(hist_df['epoch'], hist_df['val_rul_mse'],   label='Val RUL MSE',   color='tomato', linestyle='--')
axes[2].set_title('RUL MSE'); axes[2].set_xlabel('Epoch'); axes[2].legend()

plt.suptitle('Transfer Learning: NASA → CALCE  |  Training Curves', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'transfer_training_curves.png'), dpi=150)
plt.close()

# --- SOH prediction trajectory on test set ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
valid_idx = np.where(sm)[0]
axes[0].plot(valid_idx, true_soh[sm], label='Ground Truth SOH', color='steelblue', lw=2)
axes[0].plot(valid_idx, pred_soh[sm], label='Predicted SOH',    color='tomato',    lw=1.5, linestyle='--')
axes[0].set_title('CS2_21 — SOH Trajectory'); axes[0].set_xlabel('Sequence Index')
axes[0].set_ylabel('SOH (%)'); axes[0].legend()

# Parity plot
axes[1].scatter(true_soh[sm], pred_soh[sm], alpha=0.4, color='steelblue', s=10)
lo = min(true_soh[sm].min(), pred_soh[sm].min()) - 2
hi = max(true_soh[sm].max(), pred_soh[sm].max()) + 2
axes[1].plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect prediction')
axes[1].set_xlabel('Actual SOH (%)'); axes[1].set_ylabel('Predicted SOH (%)')
axes[1].set_title(f'SOH Parity Plot  (MAE={soh_mae:.2f}%)'); axes[1].legend()

plt.suptitle('Transfer Learning: NASA → CALCE  |  CS2_21 Test Evaluation', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'transfer_soh_evaluation.png'), dpi=150)
plt.close()

print(f"\n  Visualizations saved → {vis_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Save configuration
# ─────────────────────────────────────────────────────────────────────────────
config = {
    "model_name": "TransferMultiOutputLSTM",
    "strategy": "InputAdapter (11→13) + Frozen LSTM Core + Fine-tuned Heads",
    "pretrained_on": "NASA PCoE Quality-Filtered Dataset (32 batteries)",
    "fine_tuned_on": "CALCE CS2 Dataset (7 batteries)",
    "frozen_layers": ["lstm1", "lstm2"],
    "fine_tuned_layers": ["input_adapter", "shared_dense", "soh_head", "rul_head"],
    "frozen_params": frozen_params,
    "trainable_params": trainable_params,
    "calce_features": CALCE_FEATURES,
    "nasa_features": NASA_FEATURES,
    "sequence_length": SEQ_LEN,
    "training": {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "w_soh": w_soh, "w_rul": w_rul
    },
    "test_metrics": {
        "SOH_MAE_%": round(soh_mae, 4),
        "SOH_RMSE_%": round(soh_rmse, 4),
        "SOH_R2": round(soh_r2, 4),
        "RUL_MAE_cycles": round(rul_mae, 4) if not np.isnan(rul_mae) else "N/A (CS2_21 EOL at cycle 16)",
        "RUL_note": "CS2_21 reached EOL at cycle 16; only 11 valid RUL sequences in test set"
    }
}

with open(os.path.join(OUTPUT_DIR, "transfer_model_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 70)
print("TRANSFER LEARNING COMPLETE")
print("=" * 70)
print(f"  Best Epoch:      {best_epoch}")
print(f"  Best Val Loss:   {best_val_loss:.5f}")
print(f"  Test SOH MAE:    {soh_mae:.3f}%")
print(f"  Test SOH RMSE:   {soh_rmse:.3f}%")
print(f"  Test SOH R²:     {soh_r2:.4f}")
print(f"  Frozen params:   {frozen_params:,}")
print(f"  Trainable params:{trainable_params:,}")
print(f"\n  Output directory: {OUTPUT_DIR}")

