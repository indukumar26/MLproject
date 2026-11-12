import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# 1. Load dataset
df = pd.read_csv('data/final/filtered_lstm_data/filtered_nasa_battery_dataset.csv')

# 2. Flag anomalies and glitches
glitch_mask = (df['Battery_ID'].isin(['Battery_36', 'Battery_37', 'Battery_38'])) & (df['Cycle_Number'] == 1)
df.loc[glitch_mask, 'Is_Glitch'] = True
df.loc[glitch_mask, 'SOH_percent'] = np.nan

zero_mask = df['capacity_Ahr'] <= 0.05
df.loc[zero_mask, 'Is_Glitch'] = True
df.loc[zero_mask, 'SOH_percent'] = np.nan

# 3. Feature engineering
df['cumulative_ah'] = df.groupby('Battery_ID')['capacity_Ahr'].transform(lambda x: x.fillna(0).cumsum())
df['temp_rise'] = df['temperature_measured_max'] - df['temperature_measured_min']
df['voltage_drop'] = df['voltage_measured_max'] - df['voltage_measured_min']

# 4. EOL calculation (3 consecutive cycles)
eol_dict = {}
for b, g in df.groupby('Battery_ID'):
    valid = g[~g['Is_Glitch'] & g['SOH_percent'].notna()]
    under_80 = valid[valid['SOH_percent'] <= 80.0]
    if len(under_80) > 0 and valid['SOH_percent'].iloc[0] > 80.0:
        is_sub80 = (valid['SOH_percent'] <= 80.0).values
        found_eol = None
        for idx in range(len(is_sub80) - 2):
            if is_sub80[idx] and is_sub80[idx+1] and is_sub80[idx+2]:
                found_eol = valid['Cycle_Number'].iloc[idx]
                break
        if found_eol is None:
            found_eol = under_80['Cycle_Number'].iloc[0]
        eol_dict[b] = found_eol
    else:
        eol_dict[b] = np.nan

for b, eol in eol_dict.items():
    if not np.isnan(eol):
        df.loc[df['Battery_ID'] == b, 'EOL_Cycle'] = eol
        cycles = df.loc[df['Battery_ID'] == b, 'Cycle_Number']
        rul = eol - cycles
        df.loc[df['Battery_ID'] == b, 'RUL_cycles'] = rul
        df.loc[df['Battery_ID'] == b, 'Has_RUL'] = (cycles <= eol)
    else:
        df.loc[df['Battery_ID'] == b, 'EOL_Cycle'] = np.nan
        df.loc[df['Battery_ID'] == b, 'RUL_cycles'] = np.nan
        df.loc[df['Battery_ID'] == b, 'Has_RUL'] = False

feature_cols = [
    'ambient_temperature', 'Cycle_Number', 'discharge_duration_s',
    'voltage_measured_mean', 'voltage_measured_min', 'voltage_measured_max', 'voltage_measured_std',
    'current_measured_mean', 'current_measured_min',
    'temperature_measured_mean', 'temperature_measured_max', 'temperature_measured_min',
    'voltage_load_mean', 'current_load_mean',
    'cumulative_ah', 'temp_rise', 'voltage_drop'
]

seq_len = 5
split_df = pd.read_csv('data/final/filtered_lstm_data/split_battery_assignments.csv')
train_bats = split_df[split_df['Split'] == 'Train']['Battery_ID'].tolist()
val_bats   = split_df[split_df['Split'] == 'Validation']['Battery_ID'].tolist()
test_bats  = split_df[split_df['Split'] == 'Test']['Battery_ID'].tolist()

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
            
            # SOH target normalized to [0, 1]
            if np.isnan(t_s) or t_g:
                m_s.append(0.0)
                soh.append(0.0)
            else:
                m_s.append(1.0)
                soh.append(t_s / 100.0)
                
            # RUL target normalized to [0, 1] (capped at 100 cycles)
            if np.isnan(t_r) or not t_hr or t_r < 0:
                m_r.append(0.0)
                rul.append(0.0)
            else:
                m_r.append(1.0)
                rul.append(min(t_r, 100.0) / 100.0)
                
            X.append(w_feat)
            meta.append({'Battery_ID': b, 'Cycle': b_df['Cycle_Number'].iloc[i+seq_len-1]})
            
    return np.array(X), np.array(soh), np.array(rul), np.array(m_s), np.array(m_r), pd.DataFrame(meta)

X_tr, y_s_tr, y_r_tr, m_s_tr, m_r_tr, meta_tr = build_sequences(train_bats)
X_va, y_s_va, y_r_va, m_s_va, m_r_va, meta_va = build_sequences(val_bats)
X_te, y_s_te, y_r_te, m_s_te, m_r_te, meta_te = build_sequences(test_bats)

N_tr, S, F = X_tr.shape
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr.reshape(-1, F)).reshape(N_tr, S, F)
X_va_sc = scaler.transform(X_va.reshape(-1, F)).reshape(len(X_va), S, F)
X_te_sc = scaler.transform(X_te.reshape(-1, F)).reshape(len(X_te), S, F)

os.makedirs('models/nasa_attention_lstm', exist_ok=True)
joblib.dump(scaler, 'models/nasa_attention_lstm/feature_scaler.pkl')

train_dataset = TensorDataset(
    torch.tensor(X_tr_sc, dtype=torch.float32),
    torch.tensor(y_s_tr, dtype=torch.float32).unsqueeze(1),
    torch.tensor(y_r_tr, dtype=torch.float32).unsqueeze(1),
    torch.tensor(m_s_tr, dtype=torch.float32).unsqueeze(1),
    torch.tensor(m_r_tr, dtype=torch.float32).unsqueeze(1)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define Attention Multi-Task LSTM Architecture
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, lstm_outputs):
        # lstm_outputs: (batch, seq_len, hidden_dim)
        scores = self.w(lstm_outputs) # (batch, seq_len, 1)
        weights = self.softmax(scores) # (batch, seq_len, 1)
        context = torch.sum(weights * lstm_outputs, dim=1) # (batch, hidden_dim)
        return context, weights

class AttentionMultiTaskLSTM(nn.Module):
    def __init__(self, in_features=17, h1=64, h2=32, dense1=64, dense2=32, drop=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(in_features, h1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.lstm2 = nn.LSTM(h1 * 2, h2, batch_first=True, bidirectional=True)
        self.attention = AttentionBlock(h2 * 2)
        
        self.shared_dense = nn.Sequential(
            nn.Linear(h2 * 2, dense1),
            nn.LayerNorm(dense1),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dense1, dense2),
            nn.LayerNorm(dense2),
            nn.ReLU()
        )
        self.soh_head = nn.Linear(dense2, 1)
        self.rul_head = nn.Linear(dense2, 1)
        
    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1 = self.drop(out1)
        out2, _ = self.lstm2(out1)
        context, attn_weights = self.attention(out2)
        feat = self.shared_dense(context)
        soh = self.soh_head(feat)
        rul = self.rul_head(feat)
        return soh, rul

model = AttentionMultiTaskLSTM(in_features=F).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
mse_loss = nn.MSELoss(reduction='none')
huber_loss = nn.SmoothL1Loss(reduction='none')

def compute_loss(pred_s, pred_r, true_s, true_r, mask_s, mask_r):
    l_soh = (mse_loss(pred_s, true_s) * mask_s).sum() / (mask_s.sum() + 1e-8)
    l_rul = (huber_loss(pred_r, true_r) * mask_r).sum() / (mask_r.sum() + 1e-8)
    return l_soh + 0.5 * l_rul

X_va_t = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
y_s_va_t = torch.tensor(y_s_va, dtype=torch.float32).unsqueeze(1).to(device)
y_r_va_t = torch.tensor(y_r_va, dtype=torch.float32).unsqueeze(1).to(device)
m_s_va_t = torch.tensor(m_s_va, dtype=torch.float32).unsqueeze(1).to(device)
m_r_va_t = torch.tensor(m_r_va, dtype=torch.float32).unsqueeze(1).to(device)

best_val_loss = float('inf')
best_weights = None
patience = 15
patience_counter = 0

print("Starting training...")
for epoch in range(1, 81):
    model.train()
    running_loss = 0.0
    for bx, bys, byr, bms, bmr in train_loader:
        bx, bys, byr, bms, bmr = bx.to(device), bys.to(device), byr.to(device), bms.to(device), bmr.to(device)
        optimizer.zero_grad()
        ps, pr = model(bx)
        loss = compute_loss(ps, pr, bys, byr, bms, bmr)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * len(bx)
        
    train_loss = running_loss / len(train_dataset)
    
    model.eval()
    with torch.no_grad():
        vps, vpr = model(X_va_t)
        val_loss = compute_loss(vps, vpr, y_s_va_t, y_r_va_t, m_s_va_t, m_r_va_t).item()
        
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = model.state_dict().copy()
        patience_counter = 0
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}: Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f} (New Best ⭐)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (Best Val Loss: {best_val_loss:.5f})")
            break

model.load_state_dict(best_weights)
torch.save({
    'model_state_dict': model.state_dict(),
    'best_val_loss': best_val_loss,
    'features': feature_cols
}, 'models/nasa_attention_lstm/best_attention_lstm.pth')

# Test Evaluation
model.eval()
with torch.no_grad():
    X_te_t = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
    preds_s, preds_r = model(X_te_t)
    pred_soh = (preds_s.cpu().numpy() * 100.0).flatten()
    pred_rul = (preds_r.cpu().numpy() * 100.0).flatten()

meta_te['Pred_SOH'] = pred_soh
meta_te['Pred_RUL'] = pred_rul
meta_te['True_SOH'] = y_s_te * 100.0
meta_te['True_RUL'] = y_r_te * 100.0
meta_te['Mask_SOH'] = m_s_te
meta_te['Mask_RUL'] = m_r_te

print("\n" + "="*60)
print("=== UPGRADED ATTENTION LSTM: PER-BATTERY TEST RESULTS ===")
print("="*60)

for b, g in meta_te.groupby('Battery_ID'):
    g_s = g[g['Mask_SOH'] > 0]
    g_r = g[g['Mask_RUL'] > 0]
    s_mae = mean_absolute_error(g_s['True_SOH'], g_s['Pred_SOH'])
    s_rmse = root_mean_squared_error(g_s['True_SOH'], g_s['Pred_SOH'])
    s_r2  = r2_score(g_s['True_SOH'], g_s['Pred_SOH'])
    print(f"\n🔋 {b}:")
    print(f"   SOH MAE  : {s_mae:.2f}% (RMSE: {s_rmse:.2f}%, R²: {s_r2:.4f})")
    if len(g_r) > 0:
        r_mae = mean_absolute_error(g_r['True_RUL'], g_r['Pred_RUL'])
        r_rmse = root_mean_squared_error(g_r['True_RUL'], g_r['Pred_RUL'])
        r_r2  = r2_score(g_r['True_RUL'], g_r['Pred_RUL'])
        print(f"   RUL MAE  : {r_mae:.2f} cycles (RMSE: {r_rmse:.2f}, R²: {r_r2:.4f})")
    else:
        print("   RUL      : Right-censored (no failure reached)")

v_s = meta_te[meta_te['Mask_SOH'] > 0]
v_r = meta_te[meta_te['Mask_RUL'] > 0]

print("\n" + "="*60)
print("=== OVERALL TEST EVALUATION ===")
print("="*60)
overall_soh_mae = mean_absolute_error(v_s['True_SOH'], v_s['Pred_SOH'])
overall_soh_rmse = root_mean_squared_error(v_s['True_SOH'], v_s['Pred_SOH'])
overall_soh_r2 = r2_score(v_s['True_SOH'], v_s['Pred_SOH'])

print(f"Overall SOH MAE  : {overall_soh_mae:.2f}%")
print(f"Overall SOH RMSE : {overall_soh_rmse:.2f}%")
print(f"Overall SOH R²   : {overall_soh_r2:.4f}")

if len(v_r) > 0:
    overall_rul_mae = mean_absolute_error(v_r['True_RUL'], v_r['Pred_RUL'])
    overall_rul_rmse = root_mean_squared_error(v_r['True_RUL'], v_r['Pred_RUL'])
    overall_rul_r2 = r2_score(v_r['True_RUL'], v_r['Pred_RUL'])
    print(f"Overall RUL MAE  : {overall_rul_mae:.2f} cycles")
    print(f"Overall RUL RMSE : {overall_rul_rmse:.2f} cycles")
    print(f"Overall RUL R²   : {overall_rul_r2:.4f}")

meta_te.to_csv('models/nasa_attention_lstm/upgraded_test_predictions.csv', index=False)
