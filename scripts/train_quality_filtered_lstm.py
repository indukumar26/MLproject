import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs('models/nasa_lstm', exist_ok=True)
os.makedirs('outputs/07_model', exist_ok=True)
os.makedirs('outputs/09_quality', exist_ok=True)

# 1. Load dataset
df = pd.read_csv('data/final/filtered_lstm_data/filtered_nasa_battery_dataset.csv')

# 2. Fix Cycle 1 conditioning glitch for B36, B37, B38
glitch_mask = (df['Battery_ID'].isin(['Battery_36', 'Battery_37', 'Battery_38'])) & (df['Cycle_Number'] == 1)
df.loc[glitch_mask, 'Is_Glitch'] = True
df.loc[glitch_mask, 'SOH_percent'] = np.nan

# 3. Ground truth EOL & RUL (strictly pre-EOL evaluation)
eol_map = {
    'Battery_01': 101, 'Battery_02': 61, 'Battery_03': 124, 'Battery_04': 75,
    'Battery_06': 6, 'Battery_10': 6, 'Battery_24': 42, 'Battery_25': 42,
    'Battery_26': 42, 'Battery_27': 4, 'Battery_28': 18, 'Battery_29': 18,
    'Battery_30': 22, 'Battery_36': 41, 'Battery_37': 41
}

for b, eol in eol_map.items():
    cycles = df.loc[df['Battery_ID'] == b, 'Cycle_Number']
    df.loc[df['Battery_ID'] == b, 'EOL_Cycle'] = eol
    rul = eol - cycles
    df.loc[df['Battery_ID'] == b, 'RUL_cycles'] = rul
    df.loc[df['Battery_ID'] == b, 'Has_RUL'] = (cycles <= eol)

no_eol_bats = [b for b in df['Battery_ID'].unique() if b not in eol_map]
for b in no_eol_bats:
    df.loc[df['Battery_ID'] == b, 'EOL_Cycle'] = np.nan
    df.loc[df['Battery_ID'] == b, 'RUL_cycles'] = np.nan
    df.loc[df['Battery_ID'] == b, 'Has_RUL'] = False

df['cumulative_ah'] = df.groupby('Battery_ID')['capacity_Ahr'].transform(lambda x: x.fillna(0).cumsum())

feature_cols = [
    'Cycle_Number', 'discharge_duration_s',
    'voltage_measured_mean', 'voltage_measured_min', 'voltage_measured_max', 'voltage_measured_std',
    'current_measured_mean', 'current_measured_min',
    'temperature_measured_mean', 'temperature_measured_max', 'temperature_measured_min',
    'voltage_load_mean', 'current_load_mean', 'cumulative_ah'
]

seq_len = 5

def build_sequences(battery_ids):
    X, soh, rul, m_s, m_r, meta = [], [], [], [], [], []
    for b in battery_ids:
        b_df = df[df['Battery_ID'] == b].sort_values('Cycle_Number').reset_index(drop=True)
        feat = b_df[feature_cols].values
        s = b_df['SOH_percent'].values
        r = b_df['RUL_cycles'].values
        glitch = b_df['Is_Glitch'].values
        has_r = b_df['Has_RUL'].values
        
        for i in range(len(b_df) - seq_len + 1):
            w_feat = feat[i:i+seq_len]
            t_s = s[i+seq_len-1]
            t_r = r[i+seq_len-1]
            t_g = glitch[i+seq_len-1]
            t_hr = has_r[i+seq_len-1]
            
            # SOH mask
            if np.isnan(t_s) or t_g:
                m_s.append(0.0)
                soh.append(0.0)
            else:
                m_s.append(1.0)
                soh.append(t_s)
                
            # RUL mask
            if np.isnan(t_r) or not t_hr or t_r < 0:
                m_r.append(0.0)
                rul.append(0.0)
            else:
                m_r.append(1.0)
                rul.append(min(t_r, 100.0))
                
            X.append(w_feat)
            meta.append({'Battery_ID': b, 'Cycle': b_df['Cycle_Number'].iloc[i+seq_len-1]})
            
    return np.array(X), np.array(soh), np.array(rul), np.array(m_s), np.array(m_r), pd.DataFrame(meta)

train_bats = ['Battery_01', 'Battery_02', 'Battery_03']
val_bats   = ['Battery_06', 'Battery_15', 'Battery_25', 'Battery_29']
test_bats  = ['Battery_04', 'Battery_16', 'Battery_30', 'Battery_38']

X_tr, y_s_tr, y_r_tr, m_s_tr, m_r_tr, meta_tr = build_sequences(train_bats)
X_va, y_s_va, y_r_va, m_s_va, m_r_va, meta_va = build_sequences(val_bats)
X_te, y_s_te, y_r_te, m_s_te, m_r_te, meta_te = build_sequences(test_bats)

N_tr, S, F = X_tr.shape
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr.reshape(-1, F)).reshape(N_tr, S, F)
X_va_sc = scaler.transform(X_va.reshape(-1, F)).reshape(len(X_va), S, F)
X_te_sc = scaler.transform(X_te.reshape(-1, F)).reshape(len(X_te), S, F)

joblib.dump(scaler, 'models/nasa_lstm/feature_scaler.pkl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Architecture with standard layer naming compatible with transfer learning
class MultiOutputLSTM(nn.Module):
    def __init__(self, in_features=14, h1=128, h2=64, dense=32, drop=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(in_features, h1, batch_first=True)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.shared_dense = nn.Linear(h2, dense)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.soh_head = nn.Linear(dense, 1)
        self.rul_head = nn.Linear(dense, 1)
        
    def forward(self, x):
        o1, _ = self.lstm1(x)
        o2, _ = self.lstm2(o1)
        h = o2[:, -1, :]
        feat = self.dropout(self.relu(self.shared_dense(h)))
        return self.soh_head(feat), self.rul_head(feat)

torch.manual_seed(42)
np.random.seed(42)

model = MultiOutputLSTM(in_features=len(feature_cols)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

X_tr_t = torch.tensor(X_tr_sc, dtype=torch.float32).to(device)
ys_tr_t = torch.tensor(y_s_tr / 100.0, dtype=torch.float32).unsqueeze(1).to(device)
yr_tr_t = torch.tensor(y_r_tr / 100.0, dtype=torch.float32).unsqueeze(1).to(device)
ms_tr_t = torch.tensor(m_s_tr, dtype=torch.float32).unsqueeze(1).to(device)
mr_tr_t = torch.tensor(m_r_tr, dtype=torch.float32).unsqueeze(1).to(device)

loader = DataLoader(TensorDataset(X_tr_t, ys_tr_t, yr_tr_t, ms_tr_t, mr_tr_t), batch_size=16, shuffle=True)

history = []
best_val_loss = 999.0
best_state = None
best_ep = 1

for ep in range(1, 46):
    model.train()
    tot_loss, tot_soh, tot_rul = 0.0, 0.0, 0.0
    for bx, bys, byr, bms, bmr in loader:
        optimizer.zero_grad()
        ps, pr = model(bx)
        ls = ((ps - bys)**2 * bms).sum() / (bms.sum() + 1e-8)
        lr = ((pr - byr)**2 * bmr).sum() / (bmr.sum() + 1e-8)
        loss = ls + lr
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        tot_soh  += ls.item()
        tot_rul  += lr.item()
        
    n_b = len(loader)
    train_loss = tot_loss / n_b
    train_soh_mse = (tot_soh / n_b) * 10000.0
    train_rul_mse = (tot_rul / n_b) * 10000.0
    
    # Validation
    model.eval()
    with torch.no_grad():
        X_va_t = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
        ys_va_t = torch.tensor(y_s_va / 100.0, dtype=torch.float32).unsqueeze(1).to(device)
        yr_va_t = torch.tensor(y_r_va / 100.0, dtype=torch.float32).unsqueeze(1).to(device)
        ms_va_t = torch.tensor(m_s_va, dtype=torch.float32).unsqueeze(1).to(device)
        mr_va_t = torch.tensor(m_r_va, dtype=torch.float32).unsqueeze(1).to(device)
        vps, vpr = model(X_va_t)
        vls = ((vps - ys_va_t)**2 * ms_va_t).sum() / (ms_va_t.sum() + 1e-8)
        vlr = ((vpr - yr_va_t)**2 * mr_va_t).sum() / (mr_va_t.sum() + 1e-8)
        val_loss = (vls + vlr).item()
        val_soh_mse = vls.item() * 10000.0
        val_rul_mse = vlr.item() * 10000.0
        
    scheduler.step(val_loss)
    history.append({
        'epoch': ep,
        'train_loss': train_loss,
        'train_soh_mse': train_soh_mse,
        'train_rul_mse': train_rul_mse,
        'val_loss': val_loss,
        'val_soh_mse': val_soh_mse,
        'val_rul_mse': val_rul_mse,
        'learning_rate': optimizer.param_groups[0]['lr']
    })
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict().copy()
        best_ep = ep

# Save both state_dict and full checkpoint format for maximum compatibility
checkpoint_pkg = {
    'epoch': best_ep,
    'model_state_dict': best_state,
    'best_val_loss': best_val_loss
}
torch.save(checkpoint_pkg, 'models/nasa_lstm/best_multi_output_lstm.pth')
pd.DataFrame(history).to_csv('models/nasa_lstm/training_history.csv', index=False)
print("Saved best model weights and training history!")

# Test Evaluation
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    X_te_t = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
    ps, pr = model(X_te_t)
    pred_s = (ps.cpu().numpy() * 100.0).flatten()
    pred_r = (pr.cpu().numpy() * 100.0).flatten()

meta_te['SOH_Target'] = y_s_te
meta_te['RUL_Target'] = y_r_te
meta_te['Pred_SOH']   = pred_s
meta_te['Pred_RUL']   = pred_r
meta_te['Mask_SOH']   = m_s_te
meta_te['Mask_RUL']   = m_r_te
meta_te['Split']      = 'Test'
meta_te.to_csv('models/nasa_lstm/test_predictions.csv', index=False)

# Primary Benchmark Metrics (Battery_04: Standard NASA Benchmark)
b04 = meta_te[meta_te['Battery_ID'] == 'Battery_04']
b04_s = b04[b04['Mask_SOH'] > 0]
b04_r = b04[b04['Mask_RUL'] > 0]

b04_soh_mae = mean_absolute_error(b04_s['SOH_Target'], b04_s['Pred_SOH'])
b04_soh_rmse = np.sqrt(mean_squared_error(b04_s['SOH_Target'], b04_s['Pred_SOH']))
b04_soh_r2  = r2_score(b04_s['SOH_Target'], b04_s['Pred_SOH'])

b04_rul_mae = mean_absolute_error(b04_r['RUL_Target'], b04_r['Pred_RUL'])
b04_rul_rmse = np.sqrt(mean_squared_error(b04_r['RUL_Target'], b04_r['Pred_RUL']))
b04_rul_r2  = r2_score(b04_r['RUL_Target'], b04_r['Pred_RUL'])

print("\n" + "="*60)
print("=== NASA BENCHMARK TEST RESULTS (Battery_04 / B0018) ===")
print("="*60)
print(f"SOH MAE  : {b04_soh_mae:.2f}%")
print(f"SOH RMSE : {b04_soh_rmse:.2f}%")
print(f"SOH R2   : {b04_soh_r2:.4f}")
print(f"RUL MAE  : {b04_rul_mae:.2f} cycles")
print(f"RUL RMSE : {b04_rul_rmse:.2f} cycles")
print(f"RUL R2   : {b04_rul_r2:.4f}")

overall_metrics = [
    {
        'Target': 'SOH_percent (Benchmark B0018)',
        'Evaluated_Sequences': len(b04_s),
        'MAE': round(b04_soh_mae, 4),
        'RMSE': round(b04_soh_rmse, 4),
        'R2_Score': round(b04_soh_r2, 4)
    },
    {
        'Target': 'RUL_cycles (Benchmark B0018)',
        'Evaluated_Sequences': len(b04_r),
        'MAE': round(b04_rul_mae, 4),
        'RMSE': round(b04_rul_rmse, 4),
        'R2_Score': round(b04_rul_r2, 4)
    }
]
pd.DataFrame(overall_metrics).to_csv('models/nasa_lstm/test_evaluation_metrics.csv', index=False)

per_bat_rows = []
for b, g in meta_te.groupby('Battery_ID'):
    gs = g[g['Mask_SOH'] > 0]
    gr = g[g['Mask_RUL'] > 0]
    row = {
        'Battery_ID': b,
        'Sequences': len(g),
        'SOH_Eval_Count': len(gs),
        'SOH_MAE_%': round(mean_absolute_error(gs['SOH_Target'], gs['Pred_SOH']), 3),
        'SOH_RMSE_%': round(np.sqrt(mean_squared_error(gs['SOH_Target'], gs['Pred_SOH'])), 3),
        'SOH_R2': round(r2_score(gs['SOH_Target'], gs['Pred_SOH']), 4),
        'RUL_Eval_Count': len(gr),
        'RUL_MAE_cyc': round(mean_absolute_error(gr['RUL_Target'], gr['Pred_RUL']), 2) if len(gr) > 0 else np.nan,
        'RUL_RMSE_cyc': round(np.sqrt(mean_squared_error(gr['RUL_Target'], gr['Pred_RUL'])), 2) if len(gr) > 0 else np.nan,
        'RUL_R2': round(r2_score(gr['RUL_Target'], gr['Pred_RUL']), 4) if len(gr) > 0 else np.nan
    }
    per_bat_rows.append(row)

pd.DataFrame(per_bat_rows).to_csv('models/nasa_lstm/per_battery_test_metrics.csv', index=False)

# Generate publication-grade comparison plots
plt.figure(figsize=(14, 5))

# SOH Plot
plt.subplot(1, 2, 1)
plt.plot(b04_s['Cycle'], b04_s['SOH_Target'], 'b-', lw=2.5, label='Actual SOH (%)')
plt.plot(b04_s['Cycle'], b04_s['Pred_SOH'], 'r--', lw=2.0, label='Predicted SOH (%)')
plt.title(f'Battery_04 SOH Estimation (MAE: {b04_soh_mae:.2f}%, R²: {b04_soh_r2:.4f})', fontsize=12, fontweight='bold')
plt.xlabel('Discharge Cycle Number', fontsize=11)
plt.ylabel('State of Health (%)', fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# RUL Plot
plt.subplot(1, 2, 2)
plt.plot(b04_r['Cycle'], b04_r['RUL_Target'], 'g-', lw=2.5, label='Actual RUL (Cycles)')
plt.plot(b04_r['Cycle'], b04_r['Pred_RUL'], 'm--', lw=2.0, label='Predicted RUL (Cycles)')
plt.title(f'Battery_04 RUL Prognostics (MAE: {b04_rul_mae:.2f} cyc, R²: {b04_rul_r2:.4f})', fontsize=12, fontweight='bold')
plt.xlabel('Discharge Cycle Number', fontsize=11)
plt.ylabel('Remaining Useful Life (Cycles)', fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig('outputs/07_model/nasa_battery04_soh_rul_results.png', dpi=300)
plt.savefig('outputs/09_quality/nasa_battery04_soh_rul_results.png', dpi=300)
plt.close()

# Save Model Architecture Config
config = {
    "model_name": "MultiOutputLSTM",
    "dataset_version": "NASA PCoE Standard Benchmark Cohort + Corrected Pre-EOL Targets",
    "architecture": {
        "input_size": len(feature_cols),
        "hidden_dim1": 128,
        "hidden_dim2": 64,
        "dense_dim": 32,
        "dropout": 0.2
    },
    "features": feature_cols,
    "metrics_benchmark_b04": {
        "SOH_MAE_%": round(b04_soh_mae, 4),
        "SOH_R2": round(b04_soh_r2, 4),
        "RUL_MAE_cycles": round(b04_rul_mae, 4),
        "RUL_R2": round(b04_rul_r2, 4)
    }
}
with open('models/nasa_lstm/model_architecture_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Saved publication plots to outputs/07_model and outputs/09_quality!")
print("ALL DONE SUCCESSFULLY!")
