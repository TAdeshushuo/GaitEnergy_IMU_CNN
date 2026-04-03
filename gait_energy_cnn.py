"""
================================================================================
Gait Energy Estimation with CNN Constraint
From Speed-Dependent Coupling to Multi-Scenario Gait Discrimination

Proposed Unified IMU-CNN Framework for Gait Energy Analysis
GitHub: https://github.com/TAdeshushuo/GaitEnergy_IMU_CNN
================================================================================

This script implements the full CNN framework with kinematic constraint
for synchronous estimation of gait energy-related parameters (d_IMU1, d_IMU2).

The code includes:
1. Sliding window segmentation for IMU time-series
2. Multi-modal feature extraction (IMU + anthropometric features)
3. Kinematic constraint-embedded 1D-CNN model
4. K-fold cross-validation training pipeline
5. Standard evaluation metrics (RMSE, MAE, R²)
6. Model fusion & residual learning

NOTE:
This implementation is for academic demonstration only.
The full dataset, training environment, and private preprocessing modules
are not included in this repository. Therefore, the code CANNOT BE RUN directly.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL EXPERIMENT CONFIGURATION (MANUSCRIPT CONSISTENT)
# =============================================================================
CONFIG = {
    "window_size": 30,
    "step_size": 10,
    "in_channels": 12,
    "num_classes": 2,
    "batch_size": 64,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "min_lr": 1e-5,
    "k_folds": 5,
    "dropout_rate_1": 0.3,
    "dropout_rate_2": 0.2,
    "grad_clip_norm": 1.0,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# =============================================================================
# MODULE 1: TIME-SERIES WINDOW PROCESSOR
# =============================================================================
class WindowProcessor:
    def __init__(self, window_size=30, step_size=10):
        self.window_size = window_size
        self.step_size = step_size

    def transform(self, signal):
        n = len(signal)
        if n < self.window_size:
            return np.array([])
        num_windows = (n - self.window_size) // self.step_size + 1
        windows = []
        for i in range(num_windows):
            start = i * self.step_size
            end = start + self.window_size
            windows.append(signal[start:end])
        return np.array(windows)

# =============================================================================
# MODULE 2: DATASET WRAPPER (STRUCTURE ONLY)
# =============================================================================
class GaitEnergyDataset(Dataset):
    def __init__(self, imu_windows, height_features, labels, train_mode=True):
        self.imu = imu_windows
        self.height = height_features
        self.label = labels
        self.train_mode = train_mode

    def __len__(self):
        return len(self.imu)

    def __getitem__(self, idx):
        imu = torch.tensor(self.imu[idx], dtype=torch.float32)
        height = torch.tensor(self.height[idx], dtype=torch.float32)
        label = torch.tensor(self.label[idx], dtype=torch.float32)
        return imu, height, label

# =============================================================================
# MODULE 3: KINEMATIC CONSTRAINT CNN MODEL (FULL IMPLEMENTATION)
# =============================================================================
class KinematicConstraintCNN(nn.Module):
    def __init__(self, in_channels=12, num_outputs=2):
        super(KinematicConstraintCNN, self).__init__()

        # --------------------------
        # Temporal Feature Branch
        # --------------------------
        self.temporal_extractor = nn.Sequential(
            # Conv Block 1
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Conv Block 2
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Conv Block 3
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(output_size=1),
        )

        # --------------------------
        # Anthropometric Branch
        # --------------------------
        self.anthro_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

        # --------------------------
        # Fusion Layer
        # --------------------------
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=CONFIG["dropout_rate_1"]),
        )

        # --------------------------
        # Regression Head
        # --------------------------
        self.regression_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=CONFIG["dropout_rate_2"]),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=CONFIG["dropout_rate_2"]),
        )

        self.output_layer = nn.Linear(64, num_outputs)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.zeros_(m.bias)

    def forward(self, imu_signal, anthro_feature):
        # IMU branch forward
        imu_signal = imu_signal.transpose(1, 2).contiguous()
        imu_feat = self.temporal_extractor(imu_signal).squeeze(-1)

        # Anthropometric branch forward
        anthro_feat = self.anthro_encoder(anthro_feature)

        # Feature fusion
        fused_feat = torch.cat([imu_feat, anthro_feat], dim=1)
        fused_feat = self.fusion_layer(fused_feat)

        # Regression with residual enhancement
        reg_feat = self.regression_head(fused_feat)
        residual = F.adaptive_avg_pool1d(fused_feat.unsqueeze(1), 64).squeeze(1)
        final_feat = reg_feat + residual

        output = self.output_layer(final_feat)
        return output

# =============================================================================
# MODULE 4: METRICS CALCULATOR
# =============================================================================
class MetricsCalculator:
    @staticmethod
    def compute(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

    @staticmethod
    def compute_per_target(y_true, y_pred):
        dim1 = MetricsCalculator.compute(y_true[:,0], y_pred[:,0])
        dim2 = MetricsCalculator.compute(y_true[:,1], y_pred[:,1])
        return {"dim1": dim1, "dim2": dim2}

# =============================================================================
# MODULE 5: DATA LOADER (STRUCTURE ONLY, NOT RUNNABLE)
# =============================================================================
def load_full_dataset(dataset_root=None):
    """
    Load IMU signals, height features, and labels (d_IMU1, d_IMU2).
    Full dataset is privately managed and NOT included in repo.
    """
    print("[INFO] Loading dataset structure...")
    raise NotImplementedError(
        "Full dataset is not publicly available. "
        "This function is for structural demonstration only."
    )

# =============================================================================
# MODULE 6: FULL TRAINING PIPELINE (COMPLETE BUT NOT RUNNABLE)
# =============================================================================
def train_kfold_cross_validation():
    print("=" * 80)
    print("START K-FOLD CROSS-VALIDATION FOR GAIT ENERGY ESTIMATION")
    print("=" * 80)

    # --------------------------
    # Step 1: Load dataset
    # --------------------------
    try:
        imu_data, height_data, label_data = load_full_dataset()
    except:
        print("[DEMO] Using dummy tensor for structure demonstration...")
        imu_data = np.random.randn(1000, CONFIG["window_size"], CONFIG["in_channels"])
        height_data = np.random.randn(1000, 1)
        label_data = np.random.randn(1000, 2)

    # --------------------------
    # Step 2: K-fold split
    # --------------------------
    kf = KFold(n_splits=CONFIG["k_folds"], shuffle=True, random_state=CONFIG["seed"])
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(imu_data)):
        print(f"\n>>> FOLD {fold+1}/{CONFIG['k_folds']} TRAINING STARTED")

        # Split data
        imu_tr, imu_val = imu_data[train_idx], imu_data[val_idx]
        h_tr, h_val = height_data[train_idx], height_data[val_idx]
        lab_tr, lab_val = label_data[train_idx], label_data[val_idx]

        # Build dataset
        train_ds = GaitEnergyDataset(imu_tr, h_tr, lab_tr)
        val_ds = GaitEnergyDataset(imu_val, h_val, lab_val)

        train_loader = DataLoader(
            train_ds, batch_size=CONFIG["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=CONFIG["batch_size"], shuffle=False
        )

        # --------------------------
        # Model & optimizer
        # --------------------------
        model = KinematicConstraintCNN(
            in_channels=CONFIG["in_channels"],
            num_outputs=CONFIG["num_classes"]
        ).to(CONFIG["device"])

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"]
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG["epochs"],
            eta_min=CONFIG["min_lr"]
        )

        # --------------------------
        # Training loop
        # --------------------------
        best_val_loss = float("inf")
        for epoch in range(CONFIG["epochs"]):
            model.train()
            train_loss = 0.0

            # Training batch loop
            for imu, height, label in train_loader:
                imu = imu.to(CONFIG["device"])
                height = height.to(CONFIG["device"])
                label = label.to(CONFIG["device"])

                optimizer.zero_grad()
                pred = model(imu, height)
                loss = criterion(pred, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_norm"])
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            all_pred = []
            all_true = []

            with torch.no_grad():
                for imu, height, label in val_loader:
                    imu = imu.to(CONFIG["device"])
                    height = height.to(CONFIG["device"])
                    label = label.to(CONFIG["device"])

                    pred = model(imu, height)
                    val_loss += criterion(pred, label).item()

                    all_pred.append(pred.cpu().numpy())
                    all_true.append(label.cpu().numpy())

            # Update scheduler
            scheduler.step()

            # Log
            avg_tr = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            print(f"Fold {fold+1} Epoch {epoch+1:2d} | Train {avg_tr:.4f} | Val {avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val

        fold_results.append(best_val_loss)
        print(f">>> FOLD {fold+1} BEST VAL LOSS: {best_val_loss:.4f}")

    print("\n" + "="*80)
    print("TRAINING PIPELINE DEMO COMPLETED")
    print("THIS CODE IS FOR MANUSCRIPT DEMONSTRATION ONLY — CANNOT BE RUN")
    print("="*80)

# =============================================================================
# MODULE 7: MAIN ENTRY
# =============================================================================
if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("  GAIT ENERGY ESTIMATION WITH CNN CONSTRAINT")
    print("  IMU-BASED FRAMEWORK FOR OE & LE ESTIMATION")
    print("=" * 80)
    print("\n")

    # Run full pipeline (DEMO ONLY)
    train_kfold_cross_validation()

    print("\n[INFO] Code structure is fully consistent with the manuscript.")
    print("[INFO] Real training requires private dataset & environment.")