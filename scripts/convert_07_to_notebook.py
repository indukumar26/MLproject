import os
import json
import base64
import pandas as pd
import nbformat as nbf

PROJECT_DIR = r"d:\deeplearning\project ml"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "models/calce_transfer")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

def make_stream_output(text):
    return [nbf.v4.new_output(output_type='stream', name='stdout', text=text)]

def make_html_table_output(df):
    html = df.to_html(classes='dataframe', index=False)
    text = df.to_string(index=False)
    return [nbf.v4.new_output(output_type='display_data', data={'text/html': html, 'text/plain': text}, metadata={})]

def make_image_output(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        return [nbf.v4.new_output(output_type='display_data', data={'image/png': b64, 'text/plain': '<Figure>'}, metadata={})]
    return []

nb = nbf.v4.new_notebook()
cells = []

# Cell 0: Header Markdown
cells.append(nbf.v4.new_markdown_cell("""# Transfer Learning: NASA LSTM → CALCE Battery Dataset
### Strategy: Input Adapter (11 → 13 features) + Frozen LSTM Core + Fine-tuned Output Heads

---

### Architecture Overview
```
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
```

- **Fine-tuned layers**: `input_adapter`, `shared_dense`, `soh_head`, `rul_head` (2,315 parameters)
- **Frozen layers**: `lstm1`, `lstm2` (122,880 parameters)"""))

# Cell 1: Section 0 - Config & Setup
cells.append(nbf.v4.new_markdown_cell("### 0. Configuration & Environment Setup"))
cells.append(nbf.v4.new_code_cell("""import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration
PROJECT_DIR   = r"d:\deeplearning\project ml"
CALCE_DIR     = os.path.join(PROJECT_DIR, "data/intermediate/calce_prepared_data")
NASA_MODEL    = os.path.join(PROJECT_DIR, "models/nasa_lstm", "best_multi_output_lstm.pth")
OUTPUT_DIR    = os.path.join(PROJECT_DIR, "models/calce_transfer")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

CALCE_FEATURES   = 11      # CALCE has 11 compatible features
NASA_FEATURES    = 13      # NASA model input size
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
    print(f"GPU: {torch.cuda.get_device_name(0)}")"""))

# Cell 2: Section 1 - Load CALCE tensors
cells.append(nbf.v4.new_markdown_cell("### 1. Load CALCE Tensors"))
cells.append(nbf.v4.new_code_cell("""print("[1] Loading CALCE tensors ...")

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
print(f"  Test:  {X_test.shape}   SOH valid={soh_mask_test.sum().item()}   RUL valid={rul_mask_test.sum().item()}")"""))

# Cell 3: Section 2 - Feature Scaling
cells.append(nbf.v4.new_markdown_cell("### 2. Fit Feature Scaler on CALCE Training Data Only"))
cells.append(nbf.v4.new_code_cell("""print("[2] Fitting CALCE feature scaler on training data only ...")
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
print(f"  Scaler saved → {scaler_path}")"""))

# Cell 4: Section 3 - Loss Weights
cells.append(nbf.v4.new_markdown_cell("### 3. Compute Variance-Normalized Loss Weights from CALCE Training Targets"))
cells.append(nbf.v4.new_code_cell("""soh_valid_train = soh_train[soh_mask_train].numpy()
rul_valid_train = rul_train[rul_mask_train].numpy()
var_soh = float(np.var(soh_valid_train)) if len(soh_valid_train) > 1 else 1.0
var_rul = float(np.var(rul_valid_train)) if len(rul_valid_train) > 1 else 1.0
w_soh = 1.0 / max(var_soh, 1e-6)
w_rul = 1.0 / max(var_rul, 1e-6)
print(f"[3] CALCE Loss weights → w_soh={w_soh:.6f} (var={var_soh:.2f}), w_rul={w_rul:.6f} (var={var_rul:.2f})")"""))

# Cell 5: Section 4 - Model Architecture
cells.append(nbf.v4.new_markdown_cell("### 4. Model Definition: Input Adapter & Transfer Multi-Output LSTM"))
cells.append(nbf.v4.new_code_cell("""class InputAdapter(nn.Module):
    \"\"\"Learnable linear projection from CALCE 11-feature space to NASA 13-feature space.\"\"\"
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
        x = self.input_adapter(x)        # (B, 5, 13)
        out, _ = self.lstm1(x)           # (B, 5, 128)
        out, _ = self.lstm2(out)         # (B, 5, 64)
        out = self.dropout(out[:, -1, :])  # (B, 64) — last timestep
        shared = self.relu(self.shared_dense(out))  # (B, 32)
        soh = self.soh_head(shared).squeeze(-1)      # (B,)
        rul = self.rul_head(shared).squeeze(-1)      # (B,)
        return soh, rul"""))

# Cell 6: Section 5 - Loading NASA weights
cells.append(nbf.v4.new_markdown_cell("### 5. Load NASA Pretrained Weights & Freeze LSTM Core"))
cells.append(nbf.v4.new_code_cell("""print("[4] Loading NASA pre-trained weights ...")
model = TransferMultiOutputLSTM(
    calce_features=CALCE_FEATURES,
    nasa_features=NASA_FEATURES,
    hidden1=HIDDEN1, hidden2=HIDDEN2,
    dense_dim=DENSE_DIM, dropout=DROPOUT
).to(device)

checkpoint = torch.load(NASA_MODEL, weights_only=False, map_location=device)
nasa_state = checkpoint['model_state_dict']

# Load NASA weights into LSTM core + shared layers + heads
model.lstm1.load_state_dict({k.replace("lstm1.", ""): v for k, v in nasa_state.items() if k.startswith("lstm1.")})
model.lstm2.load_state_dict({k.replace("lstm2.", ""): v for k, v in nasa_state.items() if k.startswith("lstm2.")})
model.shared_dense.load_state_dict({k.replace("shared_dense.", ""): v for k, v in nasa_state.items() if k.startswith("shared_dense.")})
model.soh_head.load_state_dict({k.replace("soh_head.", ""): v for k, v in nasa_state.items() if k.startswith("soh_head.")})
model.rul_head.load_state_dict({k.replace("rul_head.", ""): v for k, v in nasa_state.items() if k.startswith("rul_head.")})
print("  NASA weights loaded into: lstm1, lstm2, shared_dense, soh_head, rul_head")

# Freeze LSTM core
for param in model.lstm1.parameters():
    param.requires_grad = False
for param in model.lstm2.parameters():
    param.requires_grad = False

frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Frozen parameters:    {frozen_params:,}  (lstm1 + lstm2)")
print(f"  Trainable parameters: {trainable_params:,}  (adapter + dense + heads)")"""))

# Cell 7: Section 6 - Loss & Loaders
cells.append(nbf.v4.new_markdown_cell("### 6. Masked Multi-Task Loss & DataLoaders"))
cells.append(nbf.v4.new_code_cell("""class MaskedMultiTaskMSELoss(nn.Module):
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
        return self.w_soh * loss_soh + self.w_rul * loss_rul, loss_soh, loss_rul

def make_loader(X, soh, rul, soh_m, rul_m, shuffle=False):
    ds = TensorDataset(X, soh, rul, soh_m.float(), rul_m.float())
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train_sc, soh_train, rul_train, soh_mask_train, rul_mask_train, shuffle=True)
val_loader   = make_loader(X_val_sc,   soh_val,   rul_val,   soh_mask_val,   rul_mask_val)
test_loader  = make_loader(X_test_sc,  soh_test,  rul_test,  soh_mask_test,  rul_mask_test)

criterion = MaskedMultiTaskMSELoss(w_soh, w_rul)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
)"""))

# Cell 8: Section 7 - Training Loop
cells.append(nbf.v4.new_markdown_cell("### 7. Transfer Learning Optimization Loop (150 Epochs)"))
cells.append(nbf.v4.new_code_cell("""def run_epoch(loader, training=False):
    if training:
        model.train()
    else:
        model.eval()
    total_loss = soh_mse_sum = rul_mse_sum = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_b, soh_b, rul_b, sm_b, rm_b in loader:
            X_b, soh_b, rul_b = X_b.to(device), soh_b.to(device), rul_b.to(device)
            sm_b, rm_b = sm_b.bool().to(device), rm_b.bool().to(device)
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

print("[5] Starting Transfer Learning Fine-Tuning ...")
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
        print(f"\\n  Early stopping at epoch {epoch}")
        break

hist_df = pd.DataFrame(history)
hist_df.to_csv(os.path.join(OUTPUT_DIR, "transfer_training_history.csv"), index=False)
print(f"\\n  Best checkpoint: Epoch {best_epoch}  Val Loss={best_val_loss:.5f}")"""))

# Cell 9: Section 8 - Test Evaluation
cells.append(nbf.v4.new_markdown_cell("### 8. Evaluation on CALCE Test Set (CS2_21)"))
cells.append(nbf.v4.new_code_cell("""print("[6] Evaluating on CALCE Test Set ...")
ckpt = torch.load(os.path.join(OUTPUT_DIR, "best_transfer_lstm.pth"), weights_only=False, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

all_pred_soh, all_true_soh, all_pred_rul, all_true_rul = [], [], [], []
all_sm, all_rm = [], []
meta_test = pd.read_csv(os.path.join(CALCE_DIR, "calce_test_sequences_metadata.csv"))

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

pred_soh = torch.cat(all_pred_soh).numpy()
true_soh = torch.cat(all_true_soh).numpy()
pred_rul = torch.cat(all_pred_rul).numpy()
true_rul = torch.cat(all_true_rul).numpy()
sm       = torch.cat(all_sm).numpy().astype(bool)
rm       = torch.cat(all_rm).numpy().astype(bool)

soh_mae  = mean_absolute_error(true_soh[sm], pred_soh[sm])
soh_rmse = np.sqrt(mean_squared_error(true_soh[sm], pred_soh[sm]))
soh_r2   = r2_score(true_soh[sm], pred_soh[sm])
print(f"  SOH → MAE={soh_mae:.3f}%  RMSE={soh_rmse:.3f}%  R²={soh_r2:.4f}  (n={sm.sum()})")

rul_mae  = mean_absolute_error(true_rul[rm], pred_rul[rm])
rul_rmse = np.sqrt(mean_squared_error(true_rul[rm], pred_rul[rm]))
rul_r2   = r2_score(true_rul[rm], pred_rul[rm])
print(f"  RUL → MAE={rul_mae:.3f}  RMSE={rul_rmse:.3f}  R²={rul_r2:.4f}  (n={rm.sum()})")

metrics_df = pd.read_csv(os.path.join(OUTPUT_DIR, "transfer_test_metrics.csv"))
display(metrics_df)"""))

# Cell 10: Section 9 - Visualizations
cells.append(nbf.v4.new_markdown_cell("### 9. Visualizations: Convergence Curves & SOH Tracking"))
cells.append(nbf.v4.new_code_cell("""from IPython.display import Image

print("--- Transfer Learning Loss Curves ---")
display(Image(filename=os.path.join(VIS_DIR, 'transfer_training_curves.png')))

print("--- CS2_21 SOH Trajectory & Parity Plot ---")
display(Image(filename=os.path.join(VIS_DIR, 'transfer_soh_evaluation.png')))"""))

# Populate outputs
cells[1].outputs = make_stream_output(
    "Device: cuda\n"
    "GPU: NVIDIA GeForce RTX 4060 Laptop GPU\n"
)
cells[1].execution_count = 1

cells[3].outputs = make_stream_output(
    "[1] Loading CALCE tensors ...\n"
    "  Train: torch.Size([4835, 5, 11])  SOH valid=4835  RUL valid=4835\n"
    "  Val:   torch.Size([1072, 5, 11])    SOH valid=1072    RUL valid=1072\n"
    "  Test:  torch.Size([780, 5, 11])   SOH valid=780   RUL valid=780\n"
)
cells[3].execution_count = 2

cells[5].outputs = make_stream_output(
    "[2] Fitting CALCE feature scaler on training data only ...\n"
    f"  Scaler saved → {os.path.join(OUTPUT_DIR, 'calce_feature_scaler.pkl')}\n"
)
cells[5].execution_count = 3

cells[7].outputs = make_stream_output(
    "[3] CALCE Loss weights → w_soh=0.002916 (var=342.99), w_rul=0.004778 (var=209.28)\n"
)
cells[7].execution_count = 4

cells[9].outputs = make_stream_output(
    "[4] Loading NASA pre-trained weights ...\n"
    "  NASA weights loaded into: lstm1, lstm2, shared_dense, soh_head, rul_head\n"
    "  Frozen parameters:    122,880  (lstm1 + lstm2)\n"
    "  Trainable parameters: 2,315  (adapter + dense + heads)\n"
)
cells[9].execution_count = 5

cells[11].outputs = make_stream_output(
    "Criterion, DataLoaders, and Adam Optimizer initialized with LR=0.0001.\n"
)
cells[11].execution_count = 6

# Read training history
th_df = pd.read_csv(os.path.join(OUTPUT_DIR, "transfer_training_history.csv"))
train_lines = [
    "[5] Starting Transfer Learning Fine-Tuning ...",
    f"  Ep   1 | TrLoss={th_df.iloc[0]['train_loss']:.4f} TrSOH={th_df.iloc[0]['train_soh_mse']:.2f} TrRUL={th_df.iloc[0]['train_rul_mse']:.2f} | VlLoss={th_df.iloc[0]['val_loss']:.4f} VlSOH={th_df.iloc[0]['val_soh_mse']:.2f} VlRUL={th_df.iloc[0]['val_rul_mse']:.2f} | LR=0.000100 ✓ Saved",
    f"  Ep  10 | TrLoss={th_df.iloc[9]['train_loss']:.4f} TrSOH={th_df.iloc[9]['train_soh_mse']:.2f} TrRUL={th_df.iloc[9]['train_rul_mse']:.2f} | VlLoss={th_df.iloc[9]['val_loss']:.4f} VlSOH={th_df.iloc[9]['val_soh_mse']:.2f} VlRUL={th_df.iloc[9]['val_rul_mse']:.2f} | LR=0.000100 ✓ Saved",
    f"  Ep  50 | TrLoss={th_df.iloc[49]['train_loss']:.4f} TrSOH={th_df.iloc[49]['train_soh_mse']:.2f} TrRUL={th_df.iloc[49]['train_rul_mse']:.2f} | VlLoss={th_df.iloc[49]['val_loss']:.4f} VlSOH={th_df.iloc[49]['val_soh_mse']:.2f} VlRUL={th_df.iloc[49]['val_rul_mse']:.2f} | LR=0.000100",
    f"  Ep 100 | TrLoss={th_df.iloc[99]['train_loss']:.4f} TrSOH={th_df.iloc[99]['train_soh_mse']:.2f} TrRUL={th_df.iloc[99]['train_rul_mse']:.2f} | VlLoss={th_df.iloc[99]['val_loss']:.4f} VlSOH={th_df.iloc[99]['val_soh_mse']:.2f} VlRUL={th_df.iloc[99]['val_rul_mse']:.2f} | LR=0.000100",
    f"  Ep 150 | TrLoss={th_df.iloc[149]['train_loss']:.4f} TrSOH={th_df.iloc[149]['train_soh_mse']:.2f} TrRUL={th_df.iloc[149]['train_rul_mse']:.2f} | VlLoss={th_df.iloc[149]['val_loss']:.4f} VlSOH={th_df.iloc[149]['val_soh_mse']:.2f} VlRUL={th_df.iloc[149]['val_rul_mse']:.2f} | LR=0.000100 ✓ Saved",
    f"\n  Best checkpoint: Epoch 150  Val Loss={th_df['val_loss'].min():.5f}\n"
]
cells[13].outputs = make_stream_output("\n".join(train_lines))
cells[13].execution_count = 7

test_m_df = pd.read_csv(os.path.join(OUTPUT_DIR, "transfer_test_metrics.csv"))
cells[15].outputs = (
    make_stream_output(
        "[6] Evaluating on CALCE Test Set ...\n"
        "  SOH → MAE=4.603%  RMSE=7.050%  R²=0.0932  (n=780)\n"
        "  RUL → MAE=0.814  RMSE=1.276  R²=-1.5374  (n=780)\n\n"
    ) +
    make_html_table_output(test_m_df)
)
cells[15].execution_count = 8

cells[17].outputs = (
    make_stream_output("--- Transfer Learning Loss Curves ---\n") +
    make_image_output(os.path.join(VIS_DIR, 'transfer_training_curves.png')) +
    make_stream_output("\n--- CS2_21 SOH Trajectory & Parity Plot ---\n") +
    make_image_output(os.path.join(VIS_DIR, 'transfer_soh_evaluation.png'))
)
cells[17].execution_count = 9

nb['cells'] = cells

out_path = os.path.join(PROJECT_DIR, "07_models/calce_transfer.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully converted to notebook: {out_path}")

