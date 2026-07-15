# =============================================================================
# Multimodal Fusion — Simple Multimodal Fusion: Image + Metadata
# Mata Kuliah: Machine Learning Practicum - Module 3
# =============================================================================
# PERSIAPAN INSTALASI:
#   pip install torch torchvision numpy matplotlib
# =============================================================================

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Menggunakan device: {DEVICE}")

# =============================================================================
# BAGIAN 1: DATASET — CIFAR-10 + METADATA SINTETIS
# =============================================================================

class CIFAR10WithMeta(Dataset):
    """
    Custom Dataset yang membungkus CIFAR-10 dan menambahkan metadata sintetis.

    CIFAR-10 berisi 60.000 gambar 32×32 dari 10 kelas:
    [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]

    Metadata yang kita tambahkan:
    1. META SINTETIS   : class_index/9 + noise kecil
       → Mensimulasikan sensor yang berkorelasi dengan kelas (misal: GPS zone)
    2. META BRIGHTNESS : rata-rata kecerahan piksel gambar (nilai nyata dari gambar)
       → Berguna untuk eksperimen Deliverable 2

    Di dunia nyata, metadata bisa berupa:
    - Koordinat GPS (foto diambil di mana)
    - Timestamp (kapan diambil)
    - Metadata kamera (ISO, shutter speed)
    - Rating pengguna, tag, dll.
    """
    def __init__(self, root, train=True, transform=None, meta_type="synthetic"):
        """
        Parameter:
        - root       : folder untuk menyimpan/mengambil dataset CIFAR-10
        - train      : True = training set, False = test set
        - transform  : transformasi preprocessing gambar
        - meta_type  : "synthetic" (korelasi dengan label) atau
                       "brightness" (kecerahan pixel nyata)
        """
        # Load CIFAR-10 dari torchvision (otomatis download jika belum ada)
        self.base = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
        self.meta_type = meta_type

        # Nama 10 kelas CIFAR-10 (urutan sesuai indeks 0-9)
        self.nama_kelas = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        if meta_type == "synthetic":
            # ── Metadata Sintetis ──────────────────────────────────────────
            # Dibuat berdasarkan label kelas + noise kecil
            # Tujuan: mensimulasikan sensor yang punya korelasi dengan kelas
            # Contoh nyata: sensor suhu yang lebih tinggi di lingkungan outdoor
            # (kelas outdoor: airplane, ship) vs indoor (cat, dog)
            np.random.seed(0)
            self.meta = np.array([
                # label / 9.0    : normalisasi ke rentang [0, 1]
                # + np.random.randn() * 0.1 : tambahkan noise ±10%
                self.base.targets[i] / 9.0 + np.random.randn() * 0.1
                for i in range(len(self.base))
            ], dtype=np.float32)

        elif meta_type == "brightness":
            # ── Metadata Brightness (Kecerahan Nyata) ─────────────────────
            # Hitung rata-rata kecerahan tiap gambar dari raw pixel
            # Ini adalah fitur nyata yang bisa diekstrak dari gambar itu sendiri
            # Berguna untuk eksperimen: apakah kecerahan membantu klasifikasi?
            print("    🔆 Menghitung brightness semua gambar...")
            raw_ds = datasets.CIFAR10(root=root, train=train, download=False)

            # raw_ds.data shape: (N, 32, 32, 3) — format numpy HWC uint8
            # .mean(axis=(1,2,3)) : rata-rata seluruh pixel (H×W×C)
            # / 255.0             : normalisasi ke [0, 1]
            self.meta = (raw_ds.data.mean(axis=(1, 2, 3)) / 255.0).astype(np.float32)
            print(f"    ✅ Brightness dihitung. Range: [{self.meta.min():.3f}, {self.meta.max():.3f}]")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        """
        Mengembalikan (gambar, metadata, label) untuk satu sampel.
        Dipanggil otomatis oleh DataLoader saat iterasi.
        """
        img, label = self.base[idx]
        # torch.tensor([...]) : bungkus scalar metadata menjadi tensor 1D ukuran (1,)
        meta = torch.tensor([self.meta[idx]])
        return img, meta, label


# =============================================================================
# BAGIAN 2: MODEL-MODEL
# =============================================================================

# ── Model A: Image-Only Baseline ──────────────────────────────────────────────

class ImageOnlyModel(nn.Module):
    """
    Baseline: Hanya menggunakan gambar, TANPA metadata.
    Ini adalah titik perbandingan untuk mengukur seberapa besar
    metadata membantu.

    Arsitektur: 3-layer CNN → Global Average Pooling → Linear
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.cnn = nn.Sequential(
            # Layer 1: Conv2d(in_channels=3, out_channels=32, kernel=3, padding=1)
            # Input RGB 3 channel, output 32 feature maps
            # padding=1 → ukuran spasial tetap sama: 32×32 → 32×32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),     # Normalisasi per-batch → training lebih stabil
            nn.ReLU(),
            # MaxPool2d(2): downsampling 2× → 32×32 → 16×16
            nn.MaxPool2d(2),

            # Layer 2: 32 → 64 channel, 16×16 → 16×16 → 8×8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3: 64 → 128 channel, 8×8 → 8×8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Global Average Pooling: (B, 128, 8, 8) → (B, 128, 1, 1)
            # Lebih baik dari Flatten untuk menghindari overfitting
            nn.AdaptiveAvgPool2d((1, 1)),

            # Flatten: (B, 128, 1, 1) → (B, 128)
            nn.Flatten()
        )
        # Dropout untuk regularisasi (matikan 30% neuron saat training)
        self.dropout = nn.Dropout(0.3)
        # Classifier akhir: 128 → 10 kelas
        self.head = nn.Linear(128, num_classes)

    def forward(self, img, meta=None):
        # meta diabaikan — model ini hanya pakai gambar
        feat = self.cnn(img)            # (B, 128)
        feat = self.dropout(feat)
        return self.head(feat)          # (B, 10)


# ── Model B: Multimodal (Late Fusion) ─────────────────────────────────────────

class MultimodalLateFusion(nn.Module):
    """
    Strategi LATE FUSION: Encode setiap modalitas secara TERPISAH,
    lalu GABUNGKAN hasilnya di akhir (sebelum classifier).

    Pipeline:
    Gambar  → CNN Encoder  → vektor 128-dim ──┐
                                               ├─ Concat → Fusion Head → kelas
    Meta    → MLP Encoder  → vektor  16-dim ──┘

    Kelebihan Late Fusion:
    + Setiap encoder bisa dilatih/di-pretrain secara independen
    + Mudah menambah/menghapus modalitas
    + Robust jika satu modalitas hilang (bisa pakai encoder yang lain saja)

    Kekurangan:
    - Interaksi antar modalitas hanya terjadi di lapisan akhir
    - Kurang efektif untuk kasus yang butuh interaksi mendalam
    """
    def __init__(self, meta_dim=1, num_classes=10):
        super().__init__()

        # ── Image Encoder (sama seperti ImageOnlyModel) ──────────────────
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )   # Output: (B, 128)

        # ── Metadata Encoder (MLP kecil) ──────────────────────────────────
        # Input: scalar metadata (1 angka per sampel)
        # MLP kecil → ekspansi ke representasi 16-dim
        # Kenapa 16? Cukup untuk representasi 1 fitur numerik, tidak berlebihan
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, 16),    # 1 → 16
            nn.ReLU(),
            nn.Linear(16, 16)           # 16 → 16
        )   # Output: (B, 16)

        # ── Fusion Head ────────────────────────────────────────────────────
        # Input: concat(img_feat, meta_feat) = 128 + 16 = 144 dim
        self.fusion_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 + 16, 64),    # 144 → 64
            nn.ReLU(),
            nn.Linear(64, num_classes)  # 64 → 10 kelas
        )

    def forward(self, img, meta):
        img_feat  = self.img_encoder(img)    # (B, 128)
        meta_feat = self.meta_encoder(meta)  # (B, 16)

        # torch.cat: menggabungkan dua tensor sepanjang dimensi fitur
        # dim=1 karena dimensi 0 adalah batch
        fused = torch.cat([img_feat, meta_feat], dim=1)  # (B, 144)

        return self.fusion_head(fused)   # (B, 10)


# ── Model C: Multimodal (Early Fusion) ────────────────────────────────────────

class MultimodalEarlyFusion(nn.Module):
    """
    Strategi EARLY FUSION: Gabungkan modalitas di AWAL,
    sebelum masuk ke encoder utama.

    Di sini metadata di-expand dan di-concat dengan feature map CNN
    setelah beberapa layer awal — bukan benar-benar "awal" tapi
    lebih awal dari Late Fusion.

    Implementasi sederhana: metadata di-broadcast dan digabung
    dengan vektor gambar sebelum fusion head.

    Kelebihan Early Fusion:
    + Interaksi antar modalitas terjadi lebih dalam
    + Bisa menangkap pola gabungan yang kompleks

    Kekurangan:
    - Lebih sensitif terhadap perbedaan skala/distribusi antar modalitas
    - Encoder tidak bisa dilatih independen
    """
    def __init__(self, meta_dim=1, num_classes=10):
        super().__init__()

        # CNN encoder yang menerima gambar
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )   # Output: (B, 128)

        # Proyeksikan metadata ke dimensi yang sama agar bisa digabung lebih awal
        self.meta_proj = nn.Linear(meta_dim, 128)   # 1 → 128

        # Setelah early concat (128 + 128 = 256), proses gabungan
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, meta):
        img_feat  = self.img_encoder(img)   # (B, 128)
        meta_feat = self.meta_proj(meta)    # (B, 128) — metadata diproyeksikan ke 128-dim

        # Gabungkan di "early" stage (sebelum classifier final)
        fused = torch.cat([img_feat, meta_feat], dim=1)  # (B, 256)
        return self.fusion(fused)   # (B, 10)


# =============================================================================
# BAGIAN 3: TRAINING & EVALUASI
# =============================================================================

def latih_model(model, train_loader, val_loader,
                epochs=10, lr=1e-3, nama="Model"):
    """
    Training loop untuk model multimodal.
    DataLoader mengembalikan (img, meta, label) — tiga elemen.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=1e-4)  # L2 regularisasi
    # LR Scheduler: kurangi LR 50% setiap 5 epoch jika tidak ada perbaikan
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    history = {"train_acc": [], "val_acc": [], "train_loss": []}
    waktu_mulai = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training ──
        model.train()
        total_loss, benar, total = 0, 0, 0

        for img, meta, label in train_loader:
            img   = img.to(DEVICE)
            meta  = meta.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            output = model(img, meta)               # Forward pass
            loss = criterion(output, label)         # Hitung loss
            loss.backward()                         # Backprop
            optimizer.step()                        # Update bobot

            total_loss += loss.item()
            # .argmax(1) : ambil indeks kelas dengan probabilitas tertinggi
            benar += (output.argmax(1) == label).sum().item()
            total += label.size(0)

        scheduler.step()   # Update learning rate

        train_acc = benar / total

        # ── Validasi ──
        val_acc = evaluasi(model, val_loader)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(total_loss / len(train_loader))

        if epoch % 2 == 0 or epoch == 1:
            print(f"    Epoch {epoch:2d}/{epochs} | "
                  f"Loss: {total_loss/len(train_loader):.3f} | "
                  f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    waktu_total = time.time() - waktu_mulai
    val_acc_akhir = evaluasi(model, val_loader)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n    ⏱️  Waktu training  : {waktu_total:.1f} detik")
    print(f"    ✅ Val Accuracy     : {val_acc_akhir:.4f} ({val_acc_akhir*100:.2f}%)")
    print(f"    🔢 Jumlah parameter : {n_params:,}")

    return val_acc_akhir, history, n_params


def evaluasi(model, loader):
    """Hitung akurasi model pada satu DataLoader."""
    model.eval()
    benar, total = 0, 0
    with torch.no_grad():
        for img, meta, label in loader:
            img, meta, label = img.to(DEVICE), meta.to(DEVICE), label.to(DEVICE)
            benar += (model(img, meta).argmax(1) == label).sum().item()
            total += label.size(0)
    return benar / total


def evaluasi_per_kelas(model, loader, nama_kelas):
    """
    Hitung akurasi per kelas untuk analisis detail.
    Berguna untuk melihat kelas mana yang paling diuntungkan oleh metadata.
    """
    model.eval()
    # Inisialisasi counter per kelas
    benar_per_kelas = {k: 0 for k in nama_kelas}
    total_per_kelas = {k: 0 for k in nama_kelas}

    with torch.no_grad():
        for img, meta, label in loader:
            img, meta, label = img.to(DEVICE), meta.to(DEVICE), label.to(DEVICE)
            pred = model(img, meta).argmax(1)

            for i in range(len(label)):
                kelas = nama_kelas[label[i].item()]
                total_per_kelas[kelas] += 1
                if pred[i] == label[i]:
                    benar_per_kelas[kelas] += 1

    # Hitung akurasi per kelas
    akurasi = {k: benar_per_kelas[k] / total_per_kelas[k]
               for k in nama_kelas}
    return akurasi


# =============================================================================
# BAGIAN 4: VISUALISASI
# =============================================================================

def plot_perbandingan_akurasi(hasil_semua, simpan_sebagai="accuracy_comparison.png"):
    """
    DELIVERABLE 1: Bar chart membandingkan akurasi semua model.
    """
    nama  = list(hasil_semua.keys())
    akurasi = [hasil_semua[m]["acc"] * 100 for m in nama]

    warna = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][:len(nama)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(nama, akurasi, color=warna, edgecolor='white',
                  linewidth=1.5, width=0.5)

    for bar, acc in zip(bars, akurasi):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{acc:.2f}%",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Lab 3.4 — Perbandingan Akurasi: Image-Only vs Multimodal",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(akurasi) + 10)
    ax.grid(axis='y', alpha=0.3)

    # Tambahkan garis referensi akurasi model baseline
    baseline_acc = hasil_semua.get("Image Only", {}).get("acc", 0) * 100
    if baseline_acc:
        ax.axhline(y=baseline_acc, color='red', linestyle='--',
                   alpha=0.5, label=f'Baseline: {baseline_acc:.2f}%')
        ax.legend()

    plt.tight_layout()
    plt.savefig(simpan_sebagai, dpi=150)
    plt.show()
    print(f"    ✅ Bar chart disimpan: {simpan_sebagai}")


def plot_akurasi_per_kelas(akurasi_baseline, akurasi_multimodal,
                           nama_kelas, simpan_sebagai="per_class_accuracy.png"):
    """
    DELIVERABLE 2 (Bagian AA2): Membandingkan akurasi per kelas
    antara model baseline (image only) dan multimodal.
    Membantu mengidentifikasi kelas mana yang paling diuntungkan metadata.
    """
    x = np.arange(len(nama_kelas))
    lebar = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - lebar/2,
                   [akurasi_baseline[k] * 100 for k in nama_kelas],
                   lebar, label='Image Only', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + lebar/2,
                   [akurasi_multimodal[k] * 100 for k in nama_kelas],
                   lebar, label='Multimodal (Late Fusion)', color='#3498db', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(nama_kelas, rotation=30, ha='right')
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Akurasi Per Kelas: Image-Only vs Multimodal Late Fusion")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(simpan_sebagai, dpi=150)
    plt.show()
    print(f"    ✅ Per-class chart disimpan: {simpan_sebagai}")


def plot_learning_curve(histories, simpan_sebagai="learning_curves.png"):
    """Plot kurva training accuracy untuk semua model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    warna_map = {
        "Image Only":        "#e74c3c",
        "Multimodal (Late)": "#3498db",
        "Multimodal (Early)":"#2ecc71",
        "Multimodal (Brightness)": "#f39c12"
    }

    for nama, hist in histories.items():
        warna = warna_map.get(nama, "gray")
        axes[0].plot(hist["train_acc"], label=nama, color=warna)
        axes[1].plot(hist["val_acc"],   label=nama, color=warna)

    for ax, judul in zip(axes, ["Training Accuracy", "Validation Accuracy"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(judul)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Learning Curves — Lab 3.4", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(simpan_sebagai, dpi=150)
    plt.show()
    print(f"    ✅ Learning curves disimpan: {simpan_sebagai}")


def visualisasi_sampel_dataset(dataset, n=8, simpan_sebagai="sample_images.png"):
    """
    Tampilkan beberapa contoh gambar dari dataset beserta nilai metadata-nya.
    Berguna untuk memahami data sebelum training.
    """
    fig, axes = plt.subplots(2, n//2, figsize=(14, 6))
    axes = axes.flatten()

    nama_kelas = dataset.nama_kelas

    for i in range(n):
        img, meta, label = dataset[i * 600]   # Ambil sampel tersebar

        # Konversi tensor ke numpy untuk ditampilkan
        # img shape: (3, 32, 32) → permute ke (32, 32, 3)
        img_np = img.permute(1, 2, 0).numpy()

        # De-normalisasi: kembalikan dari [-1,1] ke [0,1]
        img_np = img_np * 0.5 + 0.5
        img_np = np.clip(img_np, 0, 1)

        axes[i].imshow(img_np)
        axes[i].set_title(
            f"{nama_kelas[label]}\nmeta={meta.item():.2f}",
            fontsize=9
        )
        axes[i].axis('off')

    plt.suptitle("Contoh Sampel CIFAR-10 dengan Nilai Metadata", fontsize=12)
    plt.tight_layout()
    plt.savefig(simpan_sebagai, dpi=150)
    plt.show()
    print(f"    ✅ Sample images disimpan: {simpan_sebagai}")


# =============================================================================
# BAGIAN 5: MAIN
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Multimodal Fusion — Multimodal Fusion: Image + Metadata        ║")
    print("║     Machine Learning Practicum — Module 3                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Hyperparameter ────────────────────────────────────────────────────────
    BATCH_SIZE = 128
    EPOCHS     = 10   # Naikkan ke 20 untuk akurasi lebih baik (butuh waktu lebih)
    LR         = 1e-3

    # ── STEP 1: Persiapan Dataset ─────────────────────────────────────────────
    print("\n📦 STEP 1: Memuat Dataset CIFAR-10 (otomatis download ~170MB)")
    print("-" * 50)

    # Transform standar untuk CIFAR-10
    transform_train = transforms.Compose([
        # RandomHorizontalFlip: augmentasi — flip gambar secara acak
        # Membuat model lebih robust terhadap orientasi
        transforms.RandomHorizontalFlip(),
        # RandomCrop: crop acak dengan padding 4px — augmentasi posisi
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # Normalisasi ke mean=0.5, std=0.5 → rentang [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset dengan metadata SINTETIS (default)
    train_ds_syn = CIFAR10WithMeta("./data", train=True,
                                   transform=transform_train, meta_type="synthetic")
    val_ds_syn   = CIFAR10WithMeta("./data", train=False,
                                   transform=transform_val,   meta_type="synthetic")

    train_loader_syn = DataLoader(train_ds_syn, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=0)
    val_loader_syn   = DataLoader(val_ds_syn,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0)

    print(f"    Training samples : {len(train_ds_syn):,}")
    print(f"    Validation samples: {len(val_ds_syn):,}")

    # Tampilkan sampel dataset
    print("\n    Menampilkan contoh sampel...")
    visualisasi_sampel_dataset(val_ds_syn, n=8, simpan_sebagai="sample_images.png")

    # ── STEP 2: Training Image-Only Baseline ──────────────────────────────────
    print("\n📊 STEP 2: Training Image-Only Baseline")
    print("-" * 50)
    model_img = ImageOnlyModel()
    acc_img, hist_img, _ = latih_model(
        model_img, train_loader_syn, val_loader_syn,
        epochs=EPOCHS, nama="Image Only"
    )

    # ── STEP 3: Training Multimodal Late Fusion (Metadata Sintetis) ───────────
    print("\n📊 STEP 3: Training Multimodal Late Fusion (Metadata Sintetis)")
    print("-" * 50)
    model_late = MultimodalLateFusion()
    acc_late, hist_late, _ = latih_model(
        model_late, train_loader_syn, val_loader_syn,
        epochs=EPOCHS, nama="Multimodal Late Fusion"
    )

    # ── STEP 4: Training Multimodal Early Fusion ──────────────────────────────
    print("\n📊 STEP 4: Training Multimodal Early Fusion (Metadata Sintetis)")
    print("-" * 50)
    model_early = MultimodalEarlyFusion()
    acc_early, hist_early, _ = latih_model(
        model_early, train_loader_syn, val_loader_syn,
        epochs=EPOCHS, nama="Multimodal Early Fusion"
    )

    # ── STEP 5: Eksperimen Brightness (Deliverable 2) ─────────────────────────
    print("\n📊 STEP 5: Eksperimen dengan Metadata Brightness (Nilai Nyata)")
    print("-" * 50)
    print("    Membuat dataset dengan metadata brightness...")
    train_ds_br = CIFAR10WithMeta("./data", train=True,
                                  transform=transform_train, meta_type="brightness")
    val_ds_br   = CIFAR10WithMeta("./data", train=False,
                                  transform=transform_val,   meta_type="brightness")
    train_loader_br = DataLoader(train_ds_br, batch_size=BATCH_SIZE,
                                 shuffle=True,  num_workers=0)
    val_loader_br   = DataLoader(val_ds_br,   batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0)

    model_bright = MultimodalLateFusion()
    acc_bright, hist_bright, _ = latih_model(
        model_bright, train_loader_br, val_loader_br,
        epochs=EPOCHS, nama="Multimodal (Brightness)"
    )

    # ── STEP 6: Rekap & Visualisasi ───────────────────────────────────────────
    print("\n" + "="*55)
    print("📊 REKAP HASIL")
    print("="*55)
    print(f"  {'Model':<30} {'Val Accuracy':>12}")
    print(f"  {'-'*30} {'-'*12}")
    print(f"  {'Image Only':<30} {acc_img*100:>11.2f}%")
    print(f"  {'Multimodal Late Fusion':<30} {acc_late*100:>11.2f}%")
    print(f"  {'Multimodal Early Fusion':<30} {acc_early*100:>11.2f}%")
    print(f"  {'Multimodal (Brightness)':<30} {acc_bright*100:>11.2f}%")

    improve_syn = (acc_late - acc_img) * 100
    improve_br  = (acc_bright - acc_img) * 100
    print(f"\n  📈 Peningkatan dari metadata SINTETIS  : {improve_syn:+.2f}%")
    print(f"  📈 Peningkatan dari metadata BRIGHTNESS: {improve_br:+.2f}%")

    hasil_semua = {
        "Image Only":            {"acc": acc_img},
        "Multimodal (Late)":     {"acc": acc_late},
        "Multimodal (Early)":    {"acc": acc_early},
        "Multimodal (Brightness)":{"acc": acc_bright},
    }

    print("\n📊 Membuat visualisasi...")
    plot_perbandingan_akurasi(hasil_semua, "accuracy_comparison.png")
    plot_learning_curve(
        {
            "Image Only":          hist_img,
            "Multimodal (Late)":   hist_late,
            "Multimodal (Early)":  hist_early,
            "Multimodal (Brightness)": hist_bright,
        },
        simpan_sebagai="learning_curves.png"
    )

    # ── STEP 7: Analisis per kelas ────────────────────────────────────────────
    print("\n📊 Analisis per kelas...")
    nama_kelas = val_ds_syn.nama_kelas
    acc_per_kelas_img  = evaluasi_per_kelas(model_img,  val_loader_syn, nama_kelas)
    acc_per_kelas_late = evaluasi_per_kelas(model_late, val_loader_syn, nama_kelas)
    plot_akurasi_per_kelas(acc_per_kelas_img, acc_per_kelas_late,
                           nama_kelas, "per_class_accuracy.png")

    # ── STEP 8: Analisis untuk laporan ───────────────────────────────────────
    print("\n" + "="*60)
    print("📝 ANALISIS UNTUK LAPORAN")
    print("="*60)
    print(f"""
DELIVERABLE 1 — PERBANDINGAN AKURASI:
  Image Only              : {acc_img*100:.2f}%
  Multimodal Late Fusion  : {acc_late*100:.2f}%
  Multimodal Early Fusion : {acc_early*100:.2f}%
  Multimodal (Brightness) : {acc_bright*100:.2f}%

DELIVERABLE 2 — METADATA SINTETIS vs BRIGHTNESS:
  Metadata sintetis BERKORELASI LANGSUNG dengan label (class/9 + noise).
  Karena itu ia memberikan "cheat code" ke model → peningkatan besar.

  Metadata brightness adalah fitur NYATA dari gambar.
  Korelasinya dengan label lebih lemah (beberapa kelas bisa terang/gelap).
  Peningkatan dari brightness biasanya lebih kecil.

  Insight: Kualitas metadata sangat menentukan manfaat multimodal fusion.
  Metadata acak/tidak relevan bahkan bisa MENURUNKAN akurasi (noise injection).

DELIVERABLE 3 — REFLEKSI:
  Arsitektur ini paling berguna di skenario nyata seperti:
  - E-commerce: gambar produk + harga/kategori/rating teks
  - Kesehatan  : foto lesi kulit + usia/jenis kelamin pasien
  - Pertanian  : foto tanaman + data cuaca/tanah sensor
  Metadata memberikan konteks yang tidak bisa dilihat dari gambar saja.
    """)

    print("\n✅ Lab 3.4 selesai!")
    print("📁 File output:")
    print("   - sample_images.png       : contoh gambar dataset")
    print("   - accuracy_comparison.png : bar chart akurasi semua model")
    print("   - learning_curves.png     : kurva training semua model")
    print("   - per_class_accuracy.png  : akurasi per kelas image vs multimodal")

    # REKAP HASIL
print("\n" + "="*40)
print("REKAP ANGKA UNTUK ANALISIS")
print("="*40)
print(f"Image Only Acc      : {acc_img:.4f}")
print(f"Late Fusion Acc     : {acc_late:.4f}")
print(f"Early Fusion Acc    : {acc_early:.4f}")
print(f"Brightness Meta Acc : {acc_bright:.4f}")
print(f"Improvement Sintetis: {(acc_late-acc_img)*100:+.2f}%")
print(f"Improvement Brightness: {(acc_bright-acc_img)*100:+.2f}%")
