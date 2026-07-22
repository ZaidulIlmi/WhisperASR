📂 Daftar File
whisper_asr.ipynb : Notebook utama berisi kode dan langkah-langkah implementasi transkripsi suara menggunakan Whisper.
requirements.txt : Daftar library dan dependency yang dibutuhkan untuk menjalankan proyek.
🚀 Cara Menjalankan
Ikuti langkah-langkah berikut untuk menjalankan proyek ini di lingkungan lokal Anda:

1. Clone Repository
Buka terminal atau command prompt, lalu jalankan perintah berikut:

git clone https://github.com/ZaidulIlmi/WhisperASR.git
2. Install Dependencies
Masuk ke folder proyek dan install seluruh library yang dibutuhkan:

bash

cd WhisperASR
pip install -r requirements.txt
3. Buka Notebook
Jalankan Jupyter Notebook melalui terminal:

bash

jupyter notebook
Lalu, buka file whisper_asr.ipynb di browser Anda.

4. Langkah-Langkah di Dalam Notebook
Jalankan sel-sel kode secara berurutan dari atas ke bawah. Terdapat 5 tahapan utama:

Tahap 1: Persiapan dan import library (seperti whisper, torch).
Tahap 2: Load model Whisper (pilih ukuran model seperti base, small, medium, dll).
Tahap 3: Persiapan file audio (format .wav, .mp3, dll) yang ingin ditranskripsi.
Tahap 4: Proses transkripsi (Speech-to-Text) dari audio menjadi teks.
Tahap 5: Menampilkan dan menyimpan hasil transkripsi.
Dibuat dengan ❤️ oleh Zaidul Ilmi
