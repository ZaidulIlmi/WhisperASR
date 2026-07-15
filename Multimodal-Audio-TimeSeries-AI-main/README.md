# Multimodal-Audio-TimeSeries-AI

Welcome to the **Multimodal-Audio-TimeSeries-AI** repository. This project showcases five advanced machine learning tasks, spanning speech and audio processing, Automatic Speech Recognition (ASR), time-series forecasting, multimodal data fusion, and lightweight Text-to-Speech (TTS) synthesis.

---

## 📂 Project Structure

```bash
├── AudioClassification/      # Task 1: Speech Commands Classification (CNN vs. Transformer)
├── WhisperASR/               # Task 2: Speech-to-Text & Spectrogram Attention Visualization
├── TimeSeriesForecasting/    # Task 3: Multi-step Time-Series Forecasting
├── MultimodalFusion/         # Task 4: Image + Tabular Metadata Fusion (Early vs. Late)
└── KokoroTTS/                # Task 5: CPU-friendly Text-to-Speech Synthesis
```

---

## 🛠️ Project Details

### 1. 🎵 Audio Classification
* **Folder**: `[AudioClassification](file:///c:/Users/venerdi/Documents/Modul%20Machine%20Learning/Modul%203%20Machine%20Learning/AudioClassification)`
* **Description**: Classifies audio spoken commands (10 classes from `SpeechCommands` dataset, e.g., *yes, no, up, down, stop, go*).
* **Models Compared**: 
  * **SmallCNN**: A 2D Convolutional Neural Network processing Mel-spectrograms.
  * **TinyAudioTransformer**: A Sequence-to-Sequence Transformer Encoder.
* **Outputs**: Training validation accuracy histories and confusion matrix.

### 2. 🎙️ Whisper ASR (Automatic Speech Recognition)
* **Folder**: `[WhisperASR](file:///c:/Users/venerdi/Documents/Modul%20Machine%20Learning/Modul%203%20Machine%20Learning/WhisperASR)`
* **Description**: Transcribes speech audio to text and maps temporal details of pronunciation.
* **Core Tasks**:
  * OpenAI **Whisper** model transcriptions (`tiny` vs. `base`).
  * Plotting raw waveforms and log Mel-spectrograms.
  * Word-level timestamp alignment and attention visualization.
  * Bilingual code-switching (mixed Indonesian-English speech) robustness analysis.

### 3. 📈 Time-Series Forecasting
* **Folder**: `[TimeSeriesForecasting](file:///c:/Users/venerdi/Documents/Modul%20Machine%20Learning/Modul%203%20Machine%20Learning/TimeSeriesForecasting)`
* **Description**: Multi-step time-series prediction comparing non-parametric baselines with deep learning models.
* **Models Compared**:
  * Baselines: **Naive Forecast**, **Moving Average**
  * Neural Networks: **1D-CNN**, **LSTM**, **Transformer Encoder**
* **Goal**: Investigating under what conditions (e.g., sequence size, dataset volume) Transformers outperform simpler architectures.

### 4. 🖼️ Multimodal Fusion
* **Folder**: `[MultimodalFusion](file:///c:/Users/venerdi/Documents/Modul%20Machine%20Learning/Modul%203%20Machine%20Learning/MultimodalFusion)`
* **Description**: Enhancing image classification (CIFAR-10) by fusing image features with tabular metadata.
* **Architectures Compared**:
  * **Image-Only**: Standard CNN baseline.
  * **Early Fusion**: Projecting and concatenating metadata with intermediate CNN feature maps.
  * **Late Fusion**: Concatenating image representations and MLP-encoded metadata representations before the classification head.

### 5. 🗣️ Kokoro Text-to-Speech (TTS)
* **Folder**: `[KokoroTTS](file:///c:/Users/venerdi/Documents/Modul%20Machine%20Learning/Modul%203%20Machine%20Learning/KokoroTTS)`
* **Description**: Running a CPU-friendly Text-to-Speech pipeline without GPU dependencies using **Kokoro-ONNX**.
* **Core Tasks**:
  * Synthesizing text using natural preset voices (male/female speakers).
  * Analyzing the effect of speed parameter on speech naturalness.
  * Comparing human recording waveforms with synthesized speech.

---

## 🚀 Setup and Installation

Follow these steps to set up the Python virtual environment and run the files locally.

### 1. Create and Activate Virtual Environment
```bash
# Navigate to the workspace root
cd "Multimodal-Audio-TimeSeries-AI"

# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Required Dependencies
```bash
pip install torch torchaudio torchvision numpy matplotlib requests soundfile scipy kokoro-onnx openai-whisper
```
*(Ensure `ffmpeg` is installed on your system for Whisper ASR audio loading).*

---

## 📈 Summary of Outputs

The models and scripts save corresponding verification figures (`.png` files) inside their respective directories:
* `TimeSeriesForecasting/mse_comparison.png` — Validation MSE bar chart.
* `MultimodalFusion/accuracy_comparison.png` — Comparison of Early vs. Late Fusion.
* `WhisperASR/mel_annotated.png` — Attention timestamp per-word visualization.
* `KokoroTTS/waveform_comparison.png` — Synthesized waveform vs. human speech comparisons.
