# =============================================================================
# Time-Series Forecasting — Time-Series Forecasting: Baselines vs. Transformer
# Mata Kuliah: Machine Learning Practicum - Module 3
# =============================================================================
# PERSIAPAN INSTALASI:
#   pip install torch numpy matplotlib
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

# Seed untuk reproducibility — agar hasil selalu sama setiap kali dijalankan
torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Menggunakan device: {DEVICE}")

# =============================================================================
# BAGIAN 1: GENERATE DATA TIME SERIES SINTETIS
# =============================================================================

def buat_time_series(T_total=500, plot=True):
    """
    Membuat data time series sintetis yang terdiri dari 3 komponen:
      1. Gelombang sinus  → pola musiman/periodik (misalnya pola harian)
      2. Tren linear      → kenaikan nilai secara perlahan
      3. Noise Gaussian   → gangguan acak (seperti data nyata)

    Ini mensimulasikan data seperti: suhu harian, harga saham, konsumsi listrik, dll.
    """
    t = np.arange(T_total)   # Array waktu: [0, 1, 2, ..., 499]

    # Komponen 1: Gelombang sinus dengan periode 50 timestep
    # 2 * pi * t / 50  → satu siklus penuh setiap 50 langkah waktu
    sinyal_sinus = np.sin(2 * np.pi * t / 50)

    # Komponen 2: Tren linear (naik perlahan)
    # 0.01 * t → di t=500, nilai tren = 5.0
    tren = 0.01 * t

    # Komponen 3: Noise acak dengan std=0.3
    # np.random.randn() → distribusi normal (mean=0, std=1)
    noise = np.random.randn(T_total) * 0.3

    # Gabungkan ketiga komponen
    series = sinyal_sinus + tren + noise

    # Normalisasi Z-score: mean=0, std=1
    # Ini penting agar model tidak kesulitan belajar karena skala data berbeda
    series = (series - series.mean()) / series.std()

    if plot:
        plt.figure(figsize=(14, 4))
        plt.plot(t, series, color='steelblue', linewidth=0.8, label='Time Series')
        plt.xlabel("Timestep")
        plt.ylabel("Nilai (ternormalisasi)")
        plt.title("Data Time Series Sintetis (Sinus + Tren + Noise)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("timeseries_data.png", dpi=150)
        plt.show()
        print("    ✅ Plot data disimpan: timeseries_data.png")

    return series, t


# =============================================================================
# BAGIAN 2: MEMBUAT DATASET (SLIDING WINDOW)
# =============================================================================

def buat_dataset(series, window=50, horizon=10):
    """
    Mengubah time series 1D menjadi dataset supervised learning
    menggunakan teknik SLIDING WINDOW.

    Konsep:
      - Input  (X): 'window' timestep terakhir yang diketahui
      - Target (Y): 'horizon' timestep berikutnya yang harus diprediksi

    Contoh (window=3, horizon=2):
      Series : [1, 2, 3, 4, 5, 6, 7]
      Sample 0: X=[1,2,3] → Y=[4,5]
      Sample 1: X=[2,3,4] → Y=[5,6]
      Sample 2: X=[3,4,5] → Y=[6,7]

    Parameter:
    - series  : array 1D time series
    - window  : jumlah timestep input (look-back period)
    - horizon : jumlah timestep yang diprediksi ke depan
    """
    X_list, Y_list = [], []

    # Iterasi sliding window
    for i in range(len(series) - window - horizon):
        # Ambil 'window' nilai sebagai input
        X_list.append(series[i : i + window])
        # Ambil 'horizon' nilai berikutnya sebagai target
        Y_list.append(series[i + window : i + window + horizon])

    # Konversi ke numpy array lalu ke PyTorch tensor
    X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1)
    # Shape X: (N_samples, window, 1)
    # unsqueeze(-1) menambah dimensi fitur di belakang → setiap timestep punya 1 fitur

    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
    # Shape Y: (N_samples, horizon)

    return X, Y


# =============================================================================
# BAGIAN 3: MODEL-MODEL FORECASTING
# =============================================================================

# ── Model 1: Naive Forecast (Baseline Paling Sederhana) ──────────────────────

def naive_forecast(X_val, horizon):
    """
    Baseline NAIVE: Prediksi = nilai TERAKHIR yang diketahui, diulang sebanyak 'horizon'.

    Ini adalah baseline paling bodoh tapi sering mengejutkan kuat untuk data stasioner.
    Jika model ML tidak bisa mengalahkan ini, artinya model terlalu kompleks / data terlalu noise.

    Contoh: input=[..., 3.2], horizon=5 → prediksi=[3.2, 3.2, 3.2, 3.2, 3.2]
    """
    # X_val[:, -1, 0] : ambil nilai timestep TERAKHIR dari setiap sampel
    # .unsqueeze(1)    : tambah dimensi agar shape jadi (N, 1)
    # .expand(-1, horizon) : ulang nilai tersebut sebanyak 'horizon' kali
    return X_val[:, -1, 0].unsqueeze(1).expand(-1, horizon)


# ── Model 2: Moving Average (Baseline Klasik) ─────────────────────────────────

def moving_average_forecast(X_val, horizon, window_ma=10):
    """
    Baseline MOVING AVERAGE: Prediksi = rata-rata 'window_ma' nilai terakhir.

    Lebih baik dari Naive untuk data berisik karena menghaluskan fluktuasi.
    Tapi masih tidak bisa menangkap tren.

    Contoh (window_ma=3): input=[1,2,3,4,5] → prediksi=[mean(3,4,5)] diulang horizon kali
    """
    # X_val[:, -window_ma:, 0] : ambil 'window_ma' timestep terakhir
    # .mean(dim=1)              : hitung rata-rata sepanjang dimensi waktu
    ma = X_val[:, -window_ma:, 0].mean(dim=1)  # Shape: (N,)
    # Ulang nilai MA sebanyak horizon
    return ma.unsqueeze(1).expand(-1, horizon)  # Shape: (N, horizon)


# ── Model 3: 1D-CNN Forecaster ────────────────────────────────────────────────

class CNN1DForecaster(nn.Module):
    """
    Model forecasting berbasis 1D Convolutional Neural Network.

    Kelebihan:
    - Cepat dilatih (fully parallel)
    - Bagus untuk menangkap pola LOKAL (pola jangka pendek)
    - Lebih sedikit parameter → generalisasi baik di data kecil

    Kekurangan:
    - Receptive field terbatas oleh kernel size
    - Tidak bisa melihat dependensi jangka sangat panjang
    """
    def __init__(self, window=50, horizon=10):
        super().__init__()
        self.net = nn.Sequential(
            # Conv1d(in_channels, out_channels, kernel_size, padding)
            # in_channels=1  karena setiap timestep punya 1 fitur
            # kernel_size=5  → setiap neuron melihat 5 timestep bertetangga
            # padding=2      → agar output length = input length (same padding)
            nn.Conv1d(1, 32, kernel_size=5, padding=2),  # Output: (B, 32, W)
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1), # Output: (B, 64, W)
            nn.ReLU(),

            # AdaptiveAvgPool1d(1): pooling global → ambil rata-rata seluruh waktu
            # Mengkompresi (B, 64, W) menjadi (B, 64, 1)
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),            # (B, 64, 1) → (B, 64)
            nn.Linear(64, horizon)   # (B, 64)   → (B, horizon)
        )

    def forward(self, x):
        # Input x: (B, W, 1) — Batch, Window, Features
        # Conv1d butuh format (B, C, L) = (B, Features, Window)
        # .permute(0, 2, 1) menukar dimensi: (B, W, 1) → (B, 1, W)
        return self.net(x.permute(0, 2, 1))


# ── Model 4: Transformer Forecaster ──────────────────────────────────────────

class TransformerForecaster(nn.Module):
    """
    Model forecasting berbasis Transformer Encoder.

    Arsitektur:
    1. Input Projection : Linear(1 → d_model) — proyeksi fitur ke dimensi model
    2. Positional Embedding : memberi tahu posisi tiap timestep ke model
    3. Transformer Encoder  : self-attention antar semua timestep
    4. Output Head          : Linear(d_model → horizon) untuk prediksi

    Kelebihan:
    - Global context: setiap timestep bisa "melihat" semua timestep lain
    - Paralel saat training (tidak seperti LSTM yang sekuensial)
    - Performa terbaik di sequence panjang dan data banyak

    Kekurangan:
    - Butuh lebih banyak data untuk performa optimal
    - Lebih berat secara komputasi (O(n²) attention)
    - Bisa overfit di dataset kecil
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, window=50, horizon=10):
        super().__init__()

        # Proyeksi input: dari 1 fitur → d_model dimensi
        # Setiap timestep diproyeksikan ke ruang vektor berdimensi d_model
        self.input_proj = nn.Linear(1, d_model)

        # Positional Embedding: berukuran (window, d_model)
        # Karena self-attention tidak tahu urutan, kita tambahkan info posisi
        # nn.Embedding(window, d_model) = tabel lookup: posisi → vektor
        self.pos_emb = nn.Embedding(window, d_model)

        # Satu layer Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,        # Dimensi representasi internal
            nhead=nhead,            # Jumlah attention head (paralel)
            dim_feedforward=128,    # Dimensi hidden layer FFN dalam Transformer
            dropout=0.1,            # Regularisasi: matikan 10% neuron saat training
            batch_first=True        # Input format: (Batch, Seq, Feature)
        )

        # Stack beberapa layer Transformer
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Output head: dari representasi token terakhir → prediksi horizon
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x shape masuk: (B, W, 1)
        B, W, _ = x.shape

        # Proyeksikan setiap timestep ke d_model dimensi
        x = self.input_proj(x)   # (B, W, 1) → (B, W, d_model)

        # Buat indeks posisi [0, 1, 2, ..., W-1]
        pos = torch.arange(W, device=x.device)   # Shape: (W,)

        # Tambahkan positional embedding ke setiap token
        # self.pos_emb(pos) : (W, d_model)
        # .unsqueeze(0)      : (1, W, d_model) → broadcast ke semua batch
        x = x + self.pos_emb(pos).unsqueeze(0)   # (B, W, d_model)

        # Proses melalui Transformer Encoder
        # Setiap token akan attend ke semua token lain
        x = self.encoder(x)   # (B, W, d_model)

        # Gunakan representasi token TERAKHIR untuk prediksi
        # Token terakhir sudah "melihat" semua konteks sebelumnya
        x = x[:, -1, :]   # (B, d_model)

        return self.head(x)   # (B, horizon)


# ── Model 5: LSTM Forecaster (Bonus) ─────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """
    Model forecasting berbasis LSTM (Long Short-Term Memory).

    LSTM adalah peningkatan dari RNN biasa yang mengatasi vanishing gradient
    melalui mekanisme gating (forget gate, input gate, output gate).

    Kelebihan dibanding RNN biasa:
    - Bisa mengingat dependensi jarak menengah
    - Lebih stabil saat training

    Kekurangan dibanding Transformer:
    - Masih sekuensial (tidak bisa paralel)
    - Memori long-range tetap terbatas
    """
    def __init__(self, hidden_dim=64, num_layers=2, window=50, horizon=10):
        super().__init__()
        # nn.LSTM(input_size, hidden_size, num_layers, batch_first)
        # input_size=1  : setiap timestep adalah 1 angka
        # hidden_size   : dimensi hidden state LSTM
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,   # Input: (Batch, Seq, Features)
            dropout=0.1
        )
        # Linear head untuk output prediksi
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        # x: (B, W, 1)
        # self.lstm(x) mengembalikan (output, (h_n, c_n))
        # output shape: (B, W, hidden_dim)
        # h_n (hidden state akhir): (num_layers, B, hidden_dim)
        out, _ = self.lstm(x)

        # Ambil output dari timestep TERAKHIR
        # out[:, -1, :] = representasi setelah memproses seluruh sequence
        return self.head(out[:, -1, :])   # (B, horizon)


# =============================================================================
# BAGIAN 4: TRAINING LOOP
# =============================================================================

def latih_model(model, X_train, Y_train, X_val, Y_val,
                epochs=50, lr=1e-3, nama_model="Model"):
    """
    Fungsi training umum yang bisa dipakai untuk semua model neural network.

    Proses training:
    1. Forward pass  : prediksi dari input
    2. Hitung loss   : MSE antara prediksi dan target
    3. Backward pass : hitung gradien (backpropagation)
    4. Update weight : optimizer mengubah bobot model

    Parameter:
    - model    : model PyTorch yang akan dilatih
    - X_train  : input training (N, W, 1)
    - Y_train  : target training (N, horizon)
    - X_val    : input validasi
    - Y_val    : target validasi
    - epochs   : jumlah epoch training
    - lr       : learning rate (seberapa besar langkah update bobot)
    """
    model = model.to(DEVICE)
    X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)
    X_val, Y_val = X_val.to(DEVICE), Y_val.to(DEVICE)

    # Adam optimizer: adaptive learning rate, lebih cepat konvergen dari SGD biasa
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # MSE Loss: Mean Squared Error = rata-rata kuadrat selisih prediksi vs target
    # Cocok untuk regression (prediksi nilai numerik)
    loss_fn = nn.MSELoss()

    # Simpan history loss untuk plotting
    history_train, history_val = [], []

    waktu_mulai = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training phase ──
        model.train()           # Aktifkan dropout dan batch norm (jika ada)
        optimizer.zero_grad()   # Reset gradien dari iterasi sebelumnya

        pred_train = model(X_train)             # Forward pass
        loss_train = loss_fn(pred_train, Y_train)  # Hitung loss

        loss_train.backward()   # Backpropagation: hitung gradien
        optimizer.step()        # Update bobot model

        # ── Validasi phase ──
        model.eval()   # Nonaktifkan dropout
        with torch.no_grad():   # Tidak perlu hitung gradien saat validasi
            pred_val = model(X_val)
            loss_val = loss_fn(pred_val, Y_val).item()

        history_train.append(loss_train.item())
        history_val.append(loss_val)

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"Train MSE: {loss_train.item():.4f} | "
                  f"Val MSE: {loss_val:.4f}")

    waktu_training = time.time() - waktu_mulai

    # Hitung final validation MSE
    model.eval()
    with torch.no_grad():
        final_mse = loss_fn(model(X_val), Y_val).item()

    print(f"\n    ⏱️  Waktu training : {waktu_training:.1f} detik")
    print(f"    📉 Final Val MSE  : {final_mse:.4f}")
    print(f"    🔢 Jumlah param   : {sum(p.numel() for p in model.parameters()):,}")

    return final_mse, history_train, history_val, waktu_training


# =============================================================================
# BAGIAN 5: EVALUASI & VISUALISASI
# =============================================================================

def hitung_mse(prediksi, target):
    """Hitung Mean Squared Error secara manual."""
    return ((prediksi - target) ** 2).mean().item()


def plot_perbandingan_mse(hasil_semua, simpan_sebagai="mse_comparison.png"):
    """
    DELIVERABLE 1: Bar chart membandingkan validation MSE semua model.

    MSE lebih rendah = prediksi lebih akurat.
    """
    nama_model = list(hasil_semua.keys())
    nilai_mse = [hasil_semua[m]["mse"] for m in nama_model]

    # Beri warna berbeda untuk baseline vs neural network
    warna = []
    for n in nama_model:
        if n in ["Naive", "Moving Average"]:
            warna.append("#e74c3c")   # Merah untuk baseline sederhana
        elif n == "1D-CNN":
            warna.append("#3498db")   # Biru untuk CNN
        elif n == "Transformer":
            warna.append("#2ecc71")   # Hijau untuk Transformer
        else:
            warna.append("#9b59b6")   # Ungu untuk LSTM

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(nama_model, nilai_mse, color=warna, edgecolor='white',
                  linewidth=1.5, width=0.6)

    # Tambahkan nilai MSE di atas tiap bar
    for bar, mse in zip(bars, nilai_mse):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{mse:.4f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Validation MSE (lebih rendah = lebih baik)", fontsize=12)
    ax.set_title("Lab 3.3 — Perbandingan MSE: Baseline vs Neural Network",
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(nilai_mse) * 1.2)

    # Legend kategori
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#e74c3c", label="Baseline (non-parametric)"),
        Patch(facecolor="#3498db", label="1D-CNN"),
        Patch(facecolor="#2ecc71", label="Transformer"),
        Patch(facecolor="#9b59b6", label="LSTM"),
    ]
    ax.legend(handles=legend, loc='upper right')

    plt.tight_layout()
    plt.savefig(simpan_sebagai, dpi=150)
    plt.show()
    print(f"    ✅ Bar chart disimpan: {simpan_sebagai}")


def plot_prediksi_vs_aktual(model, X_val, Y_val, nama_model,
                            n_sampel=3, simpan_sebagai="prediction_plot.png"):
    """
    DELIVERABLE 2: Plot prediksi vs nilai aktual untuk model terbaik.

    Menampilkan 'n_sampel' contoh dari validation set:
    - Garis biru  : history input (yang diketahui model)
    - Garis hijau : nilai aktual yang seharusnya diprediksi
    - Garis merah (putus): prediksi model
    """
    model.eval()
    X_val_gpu = X_val.to(DEVICE)
    with torch.no_grad():
        prediksi = model(X_val_gpu).cpu()

    fig, axes = plt.subplots(1, n_sampel, figsize=(5 * n_sampel, 4), sharey=True)
    if n_sampel == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        idx = i * (len(X_val) // n_sampel)   # Ambil sampel tersebar

        # Data input (history yang dilihat model)
        input_seq = X_val[idx, :, 0].numpy()
        # Target aktual
        aktual = Y_val[idx].numpy()
        # Prediksi model
        pred = prediksi[idx].numpy()

        # Buat sumbu waktu yang kontinu
        t_input = np.arange(len(input_seq))
        t_pred = np.arange(len(input_seq), len(input_seq) + len(aktual))

        ax.plot(t_input, input_seq, color='steelblue', linewidth=1.5,
                label='History (Input)')
        ax.plot(t_pred, aktual, color='green', linewidth=2,
                marker='o', markersize=4, label='Aktual')
        ax.plot(t_pred, pred, color='red', linewidth=2,
                linestyle='--', marker='x', markersize=4, label='Prediksi')

        # Garis pemisah input vs prediksi
        ax.axvline(x=len(input_seq) - 0.5, color='black',
                   linestyle=':', alpha=0.5)
        ax.set_title(f"Sampel #{idx}", fontsize=10)
        ax.set_xlabel("Timestep")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("Nilai")
            ax.legend(fontsize=8)

    fig.suptitle(f"Prediksi vs Aktual — {nama_model}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(simpan_sebagai, dpi=150)
    plt.show()
    print(f"    ✅ Plot prediksi disimpan: {simpan_sebagai}")


def plot_learning_curve(histories, simpan_sebagai="learning_curve.png"):
    """
    Plot training & validation loss per epoch untuk setiap model neural network.
    Membantu mendiagnosa overfitting atau underfitting.
    """
    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4))
    if len(histories) == 1:
        axes = [axes]

    for ax, (nama, hist) in zip(axes, histories.items()):
        ax.plot(hist["train"], label="Train Loss", color='steelblue')
        ax.plot(hist["val"], label="Val Loss", color='coral', linestyle='--')
        ax.set_title(f"Learning Curve — {nama}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(simpan_sebagai, dpi=150)
    plt.show()
    print(f"    ✅ Learning curve disimpan: {simpan_sebagai}")


# =============================================================================
# BAGIAN 6: MAIN — Jalankan semua eksperimen
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Time-Series Forecasting — Time-Series Forecasting                    ║")
    print("║     Machine Learning Practicum — Module 3                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Hyperparameter ────────────────────────────────────────────────────────
    WINDOW  = 50    # Jumlah timestep input (look-back window)
    HORIZON = 10    # Jumlah timestep yang diprediksi ke depan
    EPOCHS  = 50    # Jumlah epoch training
    SPLIT   = 0.8   # 80% untuk training, 20% untuk validasi

    # ── STEP 1: Buat data ─────────────────────────────────────────────────────
    print("\n📊 STEP 1: Membuat Data Time Series")
    print("-" * 40)
    series, t_axis = buat_time_series(T_total=500, plot=True)
    print(f"    Shape data: {series.shape}, range: [{series.min():.2f}, {series.max():.2f}]")

    # ── STEP 2: Buat dataset sliding window ──────────────────────────────────
    print("\n📊 STEP 2: Membuat Dataset Sliding Window")
    print("-" * 40)
    X, Y = buat_dataset(series, window=WINDOW, horizon=HORIZON)
    print(f"    Total sampel : {len(X)}")
    print(f"    Shape X      : {X.shape}  (sampel, window, fitur)")
    print(f"    Shape Y      : {Y.shape}  (sampel, horizon)")

    # Split 80/20
    split_idx = int(SPLIT * len(X))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_val,   Y_val   = X[split_idx:], Y[split_idx:]
    print(f"    Training     : {len(X_train)} sampel")
    print(f"    Validasi     : {len(X_val)} sampel")

    # ── STEP 3: Baseline — Naive & Moving Average ─────────────────────────────
    print("\n📊 STEP 3: Evaluasi Baseline (Non-Parametric)")
    print("-" * 40)

    pred_naive = naive_forecast(X_val, HORIZON)
    mse_naive  = hitung_mse(pred_naive, Y_val)
    print(f"    Naive MSE         : {mse_naive:.4f}")

    pred_ma   = moving_average_forecast(X_val, HORIZON, window_ma=10)
    mse_ma    = hitung_mse(pred_ma, Y_val)
    print(f"    Moving Average MSE: {mse_ma:.4f}")

    # ── STEP 4: Training 1D-CNN ──────────────────────────────────────────────
    print("\n📊 STEP 4: Training Model 1D-CNN")
    print("-" * 40)
    cnn_model = CNN1DForecaster(window=WINDOW, horizon=HORIZON)
    mse_cnn, hist_cnn_train, hist_cnn_val, t_cnn = latih_model(
        cnn_model, X_train, Y_train, X_val, Y_val,
        epochs=EPOCHS, nama_model="1D-CNN"
    )

    # ── STEP 5: Training Transformer ─────────────────────────────────────────
    print("\n📊 STEP 5: Training Model Transformer")
    print("-" * 40)
    trans_model = TransformerForecaster(window=WINDOW, horizon=HORIZON)
    mse_trans, hist_trans_train, hist_trans_val, t_trans = latih_model(
        trans_model, X_train, Y_train, X_val, Y_val,
        epochs=EPOCHS, nama_model="Transformer"
    )

    # ── STEP 6: Training LSTM (Bonus) ────────────────────────────────────────
    print("\n📊 STEP 6: Training Model LSTM (Bonus)")
    print("-" * 40)
    lstm_model = LSTMForecaster(window=WINDOW, horizon=HORIZON)
    mse_lstm, hist_lstm_train, hist_lstm_val, t_lstm = latih_model(
        lstm_model, X_train, Y_train, X_val, Y_val,
        epochs=EPOCHS, nama_model="LSTM"
    )

    # ── STEP 7: Rekap hasil ──────────────────────────────────────────────────
    print("\n" + "="*55)
    print("📊 REKAP HASIL — VALIDATION MSE (lebih rendah = lebih baik)")
    print("="*55)
    print(f"  {'Model':<20} {'Val MSE':>10}  {'Waktu (s)':>10}")
    print(f"  {'-'*20} {'-'*10}  {'-'*10}")
    print(f"  {'Naive':<20} {mse_naive:>10.4f}  {'N/A':>10}")
    print(f"  {'Moving Average':<20} {mse_ma:>10.4f}  {'N/A':>10}")
    print(f"  {'1D-CNN':<20} {mse_cnn:>10.4f}  {t_cnn:>10.1f}")
    print(f"  {'Transformer':<20} {mse_trans:>10.4f}  {t_trans:>10.1f}")
    print(f"  {'LSTM':<20} {mse_lstm:>10.4f}  {t_lstm:>10.1f}")

    # Tentukan model terbaik
    semua_mse = {
        "Naive": mse_naive, "Moving Average": mse_ma,
        "1D-CNN": mse_cnn, "Transformer": mse_trans, "LSTM": mse_lstm
    }
    model_terbaik = min(semua_mse, key=semua_mse.get)
    print(f"\n  🏆 Model Terbaik: {model_terbaik} (MSE={semua_mse[model_terbaik]:.4f})")

    # ── STEP 8: Visualisasi ──────────────────────────────────────────────────
    print("\n📊 STEP 8: Membuat Visualisasi")
    print("-" * 40)

    # Deliverable 1: Bar chart MSE
    hasil_semua = {
        "Naive":          {"mse": mse_naive},
        "Moving Average": {"mse": mse_ma},
        "1D-CNN":         {"mse": mse_cnn},
        "Transformer":    {"mse": mse_trans},
        "LSTM":           {"mse": mse_lstm},
    }
    plot_perbandingan_mse(hasil_semua, "mse_comparison.png")

    # Deliverable 2: Plot prediksi vs aktual (gunakan model terbaik neural)
    nn_mse = {"1D-CNN": mse_cnn, "Transformer": mse_trans, "LSTM": mse_lstm}
    nn_terbaik_nama = min(nn_mse, key=nn_mse.get)
    nn_terbaik_model = {
        "1D-CNN": cnn_model, "Transformer": trans_model, "LSTM": lstm_model
    }[nn_terbaik_nama]

    plot_prediksi_vs_aktual(
        nn_terbaik_model, X_val, Y_val,
        nama_model=nn_terbaik_nama,
        n_sampel=3,
        simpan_sebagai="prediction_plot.png"
    )

    # Learning curve semua neural network
    plot_learning_curve(
        {
            "1D-CNN":      {"train": hist_cnn_train,   "val": hist_cnn_val},
            "Transformer": {"train": hist_trans_train, "val": hist_trans_val},
            "LSTM":        {"train": hist_lstm_train,  "val": hist_lstm_val},
        },
        simpan_sebagai="learning_curve.png"
    )

    # ── STEP 9: Analisis untuk laporan ──────────────────────────────────────
    print("\n" + "="*60)
    print("📝 ANALISIS UNTUK LAPORAN (Deliverable 3)")
    print("="*60)
    print(f"""
APAKAH TRANSFORMER MENANG?
{'─'*50}
- Transformer MSE  : {mse_trans:.4f}
- 1D-CNN MSE       : {mse_cnn:.4f}
- Naive MSE        : {mse_naive:.4f}

PENJELASAN HASIL:
  Pada dataset kecil seperti ini (500 timestep, pola sederhana),
  CNN dan bahkan Naive baseline bisa bersaing ketat dengan Transformer.

  Alasan Transformer TIDAK selalu menang di sini:
  1. Data terlalu sedikit (500 titik) → Transformer butuh lebih banyak data
     untuk mengoptimalkan jutaan parameternya
  2. Pola sederhana (sinus + tren) → bisa ditangkap cukup dengan CNN lokal
  3. Sequence pendek (window=50) → keunggulan global context Transformer
     kurang terasa dibanding sequence ribuan timestep

  Transformer AKAN menang jika:
  - Data > 10.000 timestep
  - Pola kompleks & non-stasioner (misal: data keuangan, cuaca multi-variabel)
  - Sequence sangat panjang (ratusan sampai ribuan timestep)

  Referensi: Zeng et al. (2023) "Are Transformers Effective for
  Time Series Forecasting?" membuktikan hal serupa di benchmark standar.
    """)

    print("\n✅ Lab 3.3 selesai!")
    print("📁 File output:")
    print("   - timeseries_data.png  : visualisasi data asli")
    print("   - mse_comparison.png   : bar chart MSE semua model")
    print("   - prediction_plot.png  : prediksi vs aktual model terbaik")
    print("   - learning_curve.png   : kurva training semua model NN")

# REKAP HASIL
print("\n" + "="*40)
print("REKAP ANGKA UNTUK ANALISIS")
print("="*40)
print(f"Naive MSE      : {mse_naive:.4f}")
print(f"Moving Avg MSE : {mse_ma:.4f}")
print(f"CNN MSE        : {mse_cnn:.4f}")
print(f"Transformer MSE: {mse_trans:.4f}")
print(f"LSTM MSE       : {mse_lstm:.4f}")