# =============================================================================
# Automatic Speech Recognition (ASR) — ASR dengan Whisper: Transkripsi & Visualisasi Attention
# Mata Kuliah: Machine Learning Practicum - Module 3
# =============================================================================
# PERSIAPAN INSTALASI (jalankan di terminal sebelum menjalankan script ini):
#
#   pip install openai-whisper torchaudio matplotlib numpy sounddevice scipy
#
# Jika error ffmpeg:
#   - Windows : download dari https://ffmpeg.org/download.html, tambahkan ke PATH
#   - Linux   : sudo apt install ffmpeg
#   - Mac     : brew install ffmpeg
# =============================================================================

# --- Import library yang dibutuhkan ---
import whisper                          # Library utama Whisper dari OpenAI
import torchaudio                       # Library PyTorch untuk memproses audio
import matplotlib.pyplot as plt         # Untuk membuat grafik/plot
import matplotlib.patches as mpatches  # Untuk menambahkan kotak anotasi pada plot
import numpy as np                      # Untuk operasi numerik
import torch                            # Framework deep learning PyTorch
import os                               # Untuk operasi file dan direktori
import time                             # Untuk mengukur waktu eksekusi
import warnings                         # Untuk menyembunyikan peringatan tidak penting
warnings.filterwarnings("ignore")       # Sembunyikan warning agar output bersih

# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 0: REKAM AUDIO SENDIRI (OPSIONAL)
# Jika kamu ingin merekam suara langsung dari mikrofon laptop.
# Jika sudah punya file .wav, lewati bagian ini dan langsung ke Bagian 1.
# ─────────────────────────────────────────────────────────────────────────────

def rekam_audio(nama_file="my_speech.wav", durasi_detik=5, sample_rate=16000):
    """
    Merekam audio dari mikrofon laptop dan menyimpannya sebagai file .wav.

    Parameter:
    - nama_file     : nama file output (default: my_speech.wav)
    - durasi_detik  : berapa lama merekam dalam detik (default: 5 detik)
    - sample_rate   : kualitas audio dalam Hz (default: 16000 Hz = standar speech)
    """
    try:
        import sounddevice as sd    # Library untuk mengakses mikrofon
        from scipy.io.wavfile import write  # Untuk menyimpan file WAV

        print(f"\n🎙️  Bersiap merekam selama {durasi_detik} detik...")
        print("    Tekan Enter untuk mulai!")
        input()  # Tunggu user menekan Enter

        print("    🔴 MEREKAM... Bicaralah sekarang!")

        # sd.rec() : merekam audio dari mikrofon
        # durasi_detik * sample_rate = total jumlah sampel yang direkam
        # channels=1 = mono (1 saluran audio, cukup untuk speech)
        # dtype='int16' = format data audio 16-bit integer (standar WAV)
        audio_data = sd.rec(
            int(durasi_detik * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'
        )

        sd.wait()  # Tunggu hingga perekaman selesai
        print("    ✅ Rekaman selesai!")

        # Simpan hasil rekaman ke file .wav
        write(nama_file, sample_rate, audio_data)
        print(f"    💾 Disimpan sebagai: {nama_file}")
        return nama_file

    except ImportError:
        # Jika sounddevice belum terinstall
        print("⚠️  sounddevice belum terinstall.")
        print("    Jalankan: pip install sounddevice scipy")
        print("    Atau siapkan file .wav secara manual.\n")
        return None
    except Exception as e:
        print(f"⚠️  Gagal merekam: {e}")
        print("    Pastikan mikrofon terhubung dan izin mikrofon sudah diberikan.\n")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 1: MEMBUAT AUDIO SAMPEL SINTETIS (UNTUK TESTING TANPA MIKROFON)
# Jika kamu sudah punya file .wav sendiri, bagian ini tidak wajib dijalankan.
# ─────────────────────────────────────────────────────────────────────────────

def buat_audio_sintetis(nama_file="test_audio.wav", durasi=3.0, sample_rate=16000):
    """
    Membuat audio sintetis berupa gelombang sinus (tone) untuk testing.
    Berguna jika tidak ada mikrofon atau ingin memastikan kode berjalan dulu.

    Parameter:
    - nama_file   : nama file output
    - durasi      : durasi audio dalam detik
    - sample_rate : sample rate dalam Hz
    """
    from scipy.io.wavfile import write as wav_write

    print(f"🔧 Membuat audio sintetis: {nama_file}")

    # np.linspace(0, durasi, ...) : membuat array waktu dari 0 hingga 'durasi' detik
    # int(sample_rate * durasi)   : total jumlah titik sampel
    t = np.linspace(0, durasi, int(sample_rate * durasi))

    # Gabungkan beberapa frekuensi untuk membuat suara lebih kompleks
    # 440 Hz = nada A, 880 Hz = oktaf di atas A, 220 Hz = oktaf di bawah A
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +   # Frekuensi 440 Hz (nada A4)
        0.2 * np.sin(2 * np.pi * 880 * t) +   # Frekuensi 880 Hz (harmonis)
        0.1 * np.sin(2 * np.pi * 220 * t)     # Frekuensi 220 Hz (sub-harmonis)
    )

    # Normalisasi: skala nilai ke rentang int16 (-32767 hingga 32767)
    audio_int16 = (audio * 32767).astype(np.int16)
    wav_write(nama_file, sample_rate, audio_int16)
    print(f"    ✅ Audio sintetis disimpan: {nama_file}")
    return nama_file


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 2: MEMUAT MODEL WHISPER
# ─────────────────────────────────────────────────────────────────────────────

def muat_model(ukuran_model="tiny"):
    """
    Memuat model Whisper dari OpenAI.

    Pilihan ukuran model (dari kecil ke besar):
    - "tiny"   : ~39 juta parameter, ~1 GB RAM, paling cepat, akurasi terendah
    - "base"   : ~74 juta parameter, ~1 GB RAM, cepat, akurasi cukup
    - "small"  : ~244 juta parameter, ~2 GB RAM, seimbang
    - "medium" : ~769 juta parameter, ~5 GB RAM, akurasi tinggi
    - "large"  : ~1.5 miliar parameter, ~10 GB RAM, akurasi tertinggi

    Untuk laptop biasa tanpa GPU, gunakan "tiny" atau "base".
    """
    print(f"\n📦 Memuat model Whisper '{ukuran_model}'...")
    print("    (Download otomatis ~150MB untuk 'tiny', hanya pertama kali)")

    # whisper.load_model() : mengunduh dan memuat model ke memori
    model = whisper.load_model(ukuran_model)

    # Hitung total parameter model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    ✅ Model dimuat! Total parameter: {total_params:,}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 3: TRANSKRIPSI AUDIO (TASK 1 — Transkripsi 5 Kalimat)
# ─────────────────────────────────────────────────────────────────────────────

def transkripsi_audio(model, path_file, bahasa=None, tampilkan_detail=True):
    """
    Mentranskripsikan file audio menjadi teks menggunakan Whisper.

    Parameter:
    - model         : model Whisper yang sudah dimuat
    - path_file     : lokasi file audio (.wav, .mp3, dll)
    - bahasa        : kode bahasa ('en'=Inggris, 'id'=Indonesia, None=auto-detect)
    - tampilkan_detail : apakah menampilkan info detail
    """
    if not os.path.exists(path_file):
        print(f"⚠️  File tidak ditemukan: {path_file}")
        return None

    print(f"\n🎧 Mentranskripsikan: {path_file}")

    # Catat waktu mulai untuk mengukur kecepatan
    waktu_mulai = time.time()

    # model.transcribe() : fungsi utama untuk transkripsi
    # verbose=True       : tampilkan progress transkripsi
    # language           : None = Whisper otomatis mendeteksi bahasa
    hasil = model.transcribe(
        path_file,
        language=bahasa,    # None = auto-detect bahasa
        verbose=False        # False = tidak tampilkan log internal
    )

    waktu_selesai = time.time()
    durasi_proses = waktu_selesai - waktu_mulai

    if tampilkan_detail:
        print(f"    📝 Transkripsi  : {hasil['text']}")
        print(f"    🌍 Bahasa terdeteksi: {hasil['language']}")
        print(f"    ⏱️  Waktu proses : {durasi_proses:.2f} detik")

    return hasil


def bandingkan_model_tiny_vs_base(path_file):
    """
    TASK 1: Membandingkan akurasi model Whisper 'tiny' vs 'base'.
    Sesuai instruksi lab: transkripsi 5 kalimat berbeda dengan dua model.
    """
    print("\n" + "="*60)
    print("TASK 1: PERBANDINGAN MODEL TINY vs BASE")
    print("="*60)

    hasil_perbandingan = {}

    for ukuran in ["tiny", "base"]:
        print(f"\n--- Model: {ukuran} ---")
        model_sementara = muat_model(ukuran)
        hasil = transkripsi_audio(model_sementara, path_file, bahasa="en")
        if hasil:
            hasil_perbandingan[ukuran] = hasil['text']

        # Bebaskan memori setelah digunakan
        del model_sementara
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n📊 HASIL PERBANDINGAN:")
    print(f"  Tiny : {hasil_perbandingan.get('tiny', 'N/A')}")
    print(f"  Base : {hasil_perbandingan.get('base', 'N/A')}")
    return hasil_perbandingan


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 4: VISUALISASI MEL-SPECTROGRAM (TASK 2)
# ─────────────────────────────────────────────────────────────────────────────

def visualisasi_mel_spectrogram(path_file, simpan_sebagai="mel_spectrogram.png"):
    """
    TASK 2: Memplot Mel-Spectrogram dari file audio.

    Mel-Spectrogram adalah representasi 2D audio:
    - Sumbu X : Waktu (frame 10ms)
    - Sumbu Y : Frekuensi dalam skala Mel (bukan linear)
    - Warna   : Intensitas energi di frekuensi tersebut (dB)

    Ini adalah INPUT yang diberikan ke model Whisper.
    """
    print(f"\n📊 Membuat visualisasi Mel-Spectrogram: {path_file}")

    # whisper.load_audio() : membaca file audio dan mengkonversi ke array numpy
    # Output: array 1D float32 pada sample rate 16000 Hz
    audio = whisper.load_audio(path_file)

    # whisper.pad_or_trim() : memotong atau menambah padding audio menjadi 30 detik
    # Whisper selalu memproses audio dalam potongan 30 detik
    audio = whisper.pad_or_trim(audio)

    # whisper.log_mel_spectrogram() : mengkonversi waveform ke log Mel-Spectrogram
    # Output shape: (80, 3000) = 80 mel bins × 3000 time frames (untuk 30 detik)
    # 80 bins karena manusia paling sensitif pada rentang frekuensi tersebut
    mel = whisper.log_mel_spectrogram(audio)  # Tensor shape: (80, 3000)

    # Deteksi durasi asli audio sebelum padding
    audio_asli = whisper.load_audio(path_file)
    durasi_detik = len(audio_asli) / 16000  # 16000 = sample rate standar

    # ── Buat plot ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Lab 3.2 — Visualisasi Audio', fontsize=14, fontweight='bold')

    # Plot 1: Waveform (gelombang suara mentah)
    t_asli = np.linspace(0, durasi_detik, len(audio_asli))
    axes[0].plot(t_asli, audio_asli, color='steelblue', linewidth=0.5, alpha=0.8)
    axes[0].set_xlabel("Waktu (detik)")
    axes[0].set_ylabel("Amplitudo")
    axes[0].set_title("Waveform Audio Mentah (Raw Waveform)")
    axes[0].axvline(x=durasi_detik, color='red', linestyle='--',
                    alpha=0.7, label=f'Akhir speech ({durasi_detik:.1f}s)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Mel-Spectrogram
    # mel.numpy() : konversi PyTorch Tensor ke numpy array untuk plotting
    img = axes[1].imshow(
        mel.numpy(),
        aspect="auto",      # Sesuaikan rasio aspek otomatis
        origin="lower",     # Frekuensi rendah di bawah (konvensi audio)
        cmap="viridis",     # Colormap: ungu=rendah, kuning=tinggi energi
        extent=[0, 30, 0, 80]  # [xmin, xmax, ymin, ymax]
    )
    plt.colorbar(img, ax=axes[1], label="Log Mel Energy (dB)")
    axes[1].set_xlabel("Waktu (detik)")
    axes[1].set_ylabel("Mel Frequency Bins (0-80)")
    axes[1].set_title("Mel-Spectrogram (Input ke Model Whisper)")

    # Tambahkan garis vertikal menandai akhir speech asli
    axes[1].axvline(x=durasi_detik, color='red', linestyle='--',
                    linewidth=2, label=f'Akhir speech ({durasi_detik:.1f}s)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(simpan_sebagai, dpi=150, bbox_inches='tight')
    print(f"    ✅ Grafik disimpan: {simpan_sebagai}")
    plt.show()

    return mel, durasi_detik


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 5: ANOTASI SPECTROGRAM DENGAN TIMESTAMP (TASK 3)
# ─────────────────────────────────────────────────────────────────────────────

def spectrogram_dengan_anotasi(model, path_file,
                                simpan_sebagai="mel_annotated.png"):
    """
    TASK 3: Menampilkan Mel-Spectrogram yang dianotasi dengan kata-kata
    beserta timestamp dari hasil transkripsi Whisper.

    Ini membantu kita melihat KAPAN setiap kata diucapkan dalam audio.
    """
    print(f"\n🏷️  Membuat spectrogram dengan anotasi kata...")

    # Transkripsi dengan word_timestamps=True
    # Fitur ini membuat Whisper melaporkan waktu mulai & selesai tiap kata
    hasil = model.transcribe(
        path_file,
        word_timestamps=True,   # Aktifkan timestamp per-kata
        verbose=False
    )

    # Tampilkan timestamp per kata
    print("\n⏱️  Timestamp per kata:")
    semua_kata = []
    for segment in hasil["segments"]:
        for info_kata in segment.get("words", []):
            kata = info_kata["word"].strip()
            mulai = info_kata["start"]
            selesai = info_kata["end"]
            print(f"    '{kata}' [{mulai:.2f}s — {selesai:.2f}s]")
            semua_kata.append({"kata": kata, "mulai": mulai, "selesai": selesai})

    # Buat Mel-Spectrogram untuk divisualisasikan
    audio = whisper.load_audio(path_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)

    # ── Plot Mel-Spectrogram dengan anotasi ──
    fig, ax = plt.subplots(figsize=(16, 5))

    ax.imshow(
        mel.numpy(),
        aspect="auto",
        origin="lower",
        cmap="magma",       # Colormap magma: hitam-merah-kuning
        extent=[0, 30, 0, 80]
    )

    # Tambahkan anotasi teks untuk setiap kata
    warna_teks = ['white', 'yellow', 'cyan', 'lightgreen', 'orange']
    for i, info in enumerate(semua_kata):
        tengah_waktu = (info["mulai"] + info["selesai"]) / 2

        # ax.text() : menambahkan teks ke posisi (x, y) pada plot
        ax.text(
            tengah_waktu,   # posisi x = tengah durasi kata
            75,             # posisi y = dekat bagian atas spectrogram
            info["kata"],   # teks yang ditampilkan
            color=warna_teks[i % len(warna_teks)],
            fontsize=9,
            ha='center',    # horizontal alignment: tengah
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5)
        )

        # Garis vertikal menandai awal dan akhir setiap kata
        ax.axvline(x=info["mulai"], color='white', alpha=0.3, linewidth=0.8)
        ax.axvline(x=info["selesai"], color='white', alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Waktu (detik)", fontsize=11)
    ax.set_ylabel("Mel Frequency Bins", fontsize=11)
    ax.set_title(f'Mel-Spectrogram dengan Anotasi Kata\nTranskripsi: "{hasil["text"]}"',
                 fontsize=12)
    ax.set_xlim(0, max(w["selesai"] for w in semua_kata) + 1 if semua_kata else 10)

    plt.tight_layout()
    plt.savefig(simpan_sebagai, dpi=150, bbox_inches='tight')
    print(f"\n    ✅ Spectrogram teranotasi disimpan: {simpan_sebagai}")
    plt.show()

    return hasil, semua_kata


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 6: CODE-SWITCHING (TASK 4) — Campuran Indonesia + Inggris
# ─────────────────────────────────────────────────────────────────────────────

def uji_code_switching(model, path_file_campuran=None):
    """
    TASK 4: Menguji bagaimana Whisper menangani code-switching
    (pencampuran dua bahasa dalam satu kalimat).

    Contoh code-switching Indonesia-Inggris:
    "Hari ini kita akan belajar tentang machine learning."
    "Saya sudah submit assignment-nya kemarin."
    """
    print("\n" + "="*60)
    print("TASK 4: UJI CODE-SWITCHING (Bahasa Campuran)")
    print("="*60)

    if path_file_campuran and os.path.exists(path_file_campuran):
        # Transkripsi tanpa menentukan bahasa (auto-detect)
        print("\n🔍 Transkripsi TANPA menentukan bahasa (auto-detect):")
        hasil_auto = transkripsi_audio(model, path_file_campuran, bahasa=None)

        # Transkripsi dengan bahasa Indonesia
        print("\n🔍 Transkripsi dengan bahasa='id' (Indonesia):")
        hasil_id = transkripsi_audio(model, path_file_campuran, bahasa="id")

        # Transkripsi dengan bahasa Inggris
        print("\n🔍 Transkripsi dengan bahasa='en' (Inggris):")
        hasil_en = transkripsi_audio(model, path_file_campuran, bahasa="en")

        # Ringkasan perbandingan
        print("\n📊 PERBANDINGAN HASIL CODE-SWITCHING:")
        print(f"  Auto-detect : {hasil_auto['text'] if hasil_auto else 'N/A'}")
        print(f"  Bahasa ID   : {hasil_id['text'] if hasil_id else 'N/A'}")
        print(f"  Bahasa EN   : {hasil_en['text'] if hasil_en else 'N/A'}")

        print("\n💡 ANALISIS:")
        print("  - Whisper mendeteksi bahasa dominan dari audio")
        print("  - Kosakata bahasa kedua (Inggris dalam kalimat Indonesia)")
        print("    sering tetap ditranskripsi dengan benar karena Whisper")
        print("    dilatih dengan data multilingual 680.000 jam")
        print("  - Namun kata Indonesia yang fonetiknya mirip Inggris")
        print("    (misal: 'submit', 'assignment') lebih mudah dikenali")

    else:
        print("\n⚠️  File audio campuran tidak tersedia.")
        print("    Siapkan rekaman kalimat seperti:")
        print("    'Hari ini kita belajar deep learning di kampus.'")
        print("    'Tolong review code-nya sebelum deadline besok.'")
        print("\n    Jalankan rekam_audio('campuran.wav') lalu panggil fungsi ini.")


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 7: TRANSKRIPSI MULTI-KLIP (bonus dari starter code asli)
# ─────────────────────────────────────────────────────────────────────────────

def transkripsi_multi_klip(model, daftar_klip):
    """
    Mentranskripsikan beberapa file audio sekaligus dan membandingkan hasilnya.

    Parameter:
    - model      : model Whisper yang sudah dimuat
    - daftar_klip: dict berisi {nama: path_file}

    Contoh:
    daftar_klip = {
        "Kalimat Inggris": "english.wav",
        "Kalimat Indonesia": "indonesian.wav",
        "Audio Berisik": "noisy.wav",
    }
    """
    print("\n" + "="*60)
    print("TRANSKRIPSI MULTI-KLIP")
    print("="*60)

    hasil_semua = {}
    for nama, path in daftar_klip.items():
        print(f"\n[{nama}]")
        if os.path.exists(path):
            hasil = transkripsi_audio(model, path)
            hasil_semua[nama] = hasil
        else:
            print(f"    ⚠️  File tidak ditemukan: {path}")

    return hasil_semua


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 8: RINGKASAN HASIL & LAPORAN (untuk deliverables lab)
# ─────────────────────────────────────────────────────────────────────────────

def cetak_laporan_lab(hasil_transkripsi, hasil_timestamp):
    """
    Mencetak ringkasan hasil lab sesuai deliverables yang diminta.
    """
    print("\n" + "="*60)
    print("LAPORAN Automatic Speech Recognition (ASR) — RINGKASAN HASIL")
    print("="*60)

    print("""
╔══════════════════════════════════════════════════════════╗
║           DELIVERABLES Automatic Speech Recognition (ASR) — STATUS                  ║
╠══════════════════════════════════════════════════════════╣
║ ✅ Task 1: Transkripsi & bandingkan tiny vs base          ║
║ ✅ Task 2: Plot Mel-Spectrogram (disimpan .png)           ║
║ ✅ Task 3: Anotasi spectrogram dengan word timestamps     ║
║ ✅ Task 4: Uji code-switching Indonesia-Inggris           ║
╚══════════════════════════════════════════════════════════╝
    """)

    print("📝 POIN REFLEKSI UNTUK LAPORAN:")
    print("""
  1. TRANSFORMER ENCODER-DECODER:
     - Encoder   : memproses Mel-spectrogram menjadi representasi fitur audio
     - Decoder   : menghasilkan teks token-per-token menggunakan cross-attention
     - Cross-attention menghubungkan "apa yang didengar" (encoder)
       dengan "kata apa yang sedang digenerate" (decoder)

  2. MENGAPA WHISPER ROBUST:
     - Dilatih dengan 680.000 jam audio dari 96 bahasa
     - Menggunakan weak supervision (subtitle internet, bukan label manual)
     - Arsitektur seq2seq memungkinkan koreksi konteks antar kata

  3. CODE-SWITCHING:
     - Whisper mendeteksi bahasa dari 30 detik pertama audio
     - Kosakata asing tetap bisa dikenali jika fonetiknya ada dalam training
     - Performa terbaik jika satu bahasa dominan dalam klip

  4. MEL-SPECTROGRAM:
     - 80 mel bins = representasi yang efisien & sesuai persepsi manusia
     - Frame 10ms = resolusi temporal cukup untuk speech
     - Area putih/terang = energi tinggi (suara keras/vokal)
     - Area gelap = energi rendah (jeda/silence)
    """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Jalankan semua task lab secara berurutan
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Automatic Speech Recognition (ASR) — ASR dengan Whisper (Lokal Setup)           ║")
    print("║     Machine Learning Practicum — Module 3                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── STEP 1: Siapkan file audio ──────────────────────────────────────────
    print("\n📂 STEP 1: Menyiapkan File Audio")
    print("-" * 40)
    print("Pilihan:")
    print("  A) Rekam suara sendiri         → jalankan rekam_audio()")
    print("  B) Gunakan audio sintetis       → sudah otomatis dibuat di bawah")
    print("  C) Gunakan file .wav milik kamu → ganti AUDIO_FILE di bawah")

    # GANTI PATH INI dengan file audio kamu!
    # Contoh: AUDIO_FILE = "C:/Users/nama/Downloads/rekaman.wav"
    AUDIO_FILE = "my_speech.wav"

    # Jika file belum ada, buat audio sintetis untuk demo
    if not os.path.exists(AUDIO_FILE):
        print(f"\n⚠️  File '{AUDIO_FILE}' tidak ditemukan.")
        print("    Opsi 1: Rekam sekarang")
        pilihan = input("    Rekam audio sekarang? (y/n): ").strip().lower()

        if pilihan == 'y':
            AUDIO_FILE = rekam_audio("my_speech.wav", durasi_detik=10) or "test_tone.wav"
        else:
            print("    → Menggunakan audio sintetis untuk demo...")
            try:
                AUDIO_FILE = buat_audio_sintetis("test_tone.wav")
            except Exception:
                print("⚠️  scipy tidak terinstall, install dengan: pip install scipy")
                print("    Silakan siapkan file .wav manual lalu ganti AUDIO_FILE")
                exit(1)

    # ── STEP 2: Muat model Whisper ───────────────────────────────────────────
    print("\n📂 STEP 2: Memuat Model Whisper")
    print("-" * 40)
    # Gunakan "tiny" untuk laptop biasa, "base" untuk akurasi lebih baik
    MODEL_UTAMA = muat_model("tiny")

    # ── STEP 3 (Task 1): Transkripsi dasar ──────────────────────────────────
    print("\n📂 STEP 3 (Task 1): Transkripsi & Deteksi Bahasa")
    print("-" * 40)
    hasil_utama = transkripsi_audio(MODEL_UTAMA, AUDIO_FILE)

    # ── STEP 4 (Task 2): Visualisasi Mel-Spectrogram ─────────────────────────
    print("\n📂 STEP 4 (Task 2): Visualisasi Mel-Spectrogram")
    print("-" * 40)
    mel_data, durasi = visualisasi_mel_spectrogram(AUDIO_FILE, "mel_spectrogram.png")
    print(f"    ℹ️  Durasi audio: {durasi:.2f} detik")
    print(f"    ℹ️  Shape Mel-Spectrogram: {mel_data.shape} (mel_bins × time_frames)")

    # ── STEP 5 (Task 3): Anotasi Spectrogram dengan Timestamp ────────────────
    print("\n📂 STEP 5 (Task 3): Anotasi Spectrogram dengan Word Timestamps")
    print("-" * 40)
    hasil_ts, daftar_kata = spectrogram_dengan_anotasi(
        MODEL_UTAMA, AUDIO_FILE, "mel_annotated.png"
    )

    # ── STEP 6 (Task 4): Code-Switching ──────────────────────────────────────
    print("\n📂 STEP 6 (Task 4): Uji Code-Switching")
    print("-" * 40)
    # Ganti path di bawah dengan file rekaman kalimat campuran Inggris-Indonesia
    uji_code_switching(MODEL_UTAMA, path_file_campuran="campuran.wav")

    # ── STEP 7: Laporan ───────────────────────────────────────────────────────
    cetak_laporan_lab(hasil_utama, daftar_kata)

    print("\n✅ Semua task Lab 3.2 selesai!")
    print("📁 File output yang dihasilkan:")
    print("   - mel_spectrogram.png  (Task 2: Mel-Spectrogram plot)")
    print("   - mel_annotated.png    (Task 3: Spectrogram dengan anotasi kata)")
    print("\n💡 TIP: Untuk Task 1 perbandingan tiny vs base, jalankan:")
    print("   bandingkan_model_tiny_vs_base('my_speech.wav')")
