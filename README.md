# 🏭 VISION CORE - Sistem Peringatan Dini dan Deteksi Cacat Produk

**VISION CORE** adalah sebuah aplikasi web *dashboard* tingkat industri yang dikembangkan untuk mengotomatisasi proses *Quality Control* (QC) manufaktur. Proyek ini dibangun dengan mengimplementasikan dan mengkomparasikan dua arsitektur *Convolutional Neural Network* (CNN) tingkat tinggi, yakni **ResNet50** dan **Xception**, sebagai bentuk pemenuhan tugas *Capstone Project*.

Untuk mengatasi kelemahan kotak hitam (*black-box*) pada algoritma kecerdasan buatan, sistem ini diintegrasikan dengan **Explainable AI (Grad-CAM)**. Aplikasi ini dirancang dengan antarmuka *Dark Mode* modern yang memberikan kenyamanan visual bagi operator pabrik serta meminimalisir kelelahan mata saat melakukan pemantauan mutu.

## ✨ Fitur Utama
1. **Komparasi AI *Head-to-Head***: Mengeksekusi dan membandingkan probabilitas prediksi serta latensi kecepatan antara model ResNet50 dan Xception secara *real-time*.
2. **Explainable AI (Grad-CAM)**: Menampilkan visualisasi peta panas (*heatmap*) yang secara transparan menyoroti koordinat spasial anomali (goresan/penyok) pada produk logam.
3. **Pemrosesan Fleksibel**: Mendukung mode pemindaian citra tunggal (*Single Image*) maupun pemindaian massal (*Batch Mode*).
4. **Dashboard Analitik**: Menyajikan grafik distribusi kualitas produksi (Produk Normal, Cacat, dan Inspeksi Manual) secara seketika.
5. **Mitigasi *Bug Runtime* Asinkron**: Menerapkan skrip pembedah berkas *runtime* untuk membersihkan parameter siluman HDF5 (Keras 3) demi menjamin stabilitas peladen awan.
6. **Protokol *Human-in-the-Loop***: Secara otomatis menahan keputusan mesin dan mewajibkan intervensi manual jika tingkat keyakinan AI berada di bawah batas ambang 85%.

## 📁 Struktur Folder Proyek
    VISION_CORE_CAPSTONE/
    │
    ├── app/
    │   ├── static/
    │   │   ├── css/
    │   │   │   └── style.css          # Desain UI/UX Dashboard (Dark Mode)
    │   │   ├── images/                # Aset gambar web dan output Grad-CAM
    │   │   └── js/
    │   │       └── script.js          # Logika navigasi dan render grafik analitik
    │   └── templates/
    │       ├── base.html              # Struktur Layout Induk
    │       ├── index.html             # Halaman Beranda / Dashboard Utama
    │       └── komparasi.html         # Halaman Eksekusi Head-to-Head & Grad-CAM
    │
    ├── dataset/
    │   ├── ok_front/                  # Sampel citra produk normal
    │   └── def_front/                 # Sampel citra produk cacat
    │
    ├── grafik_evaluasi/               # Folder output Confusion Matrix, ROC, & Learning Curve
    │
    ├── models/
    │   ├── resnet50_model.h5          # Model AI ResNet50 tersimpan
    │   └── xception_model.h5          # Model AI Xception tersimpan (Otak Utama)
    │
    ├── .gitignore
    ├── app.py                         # File Utama Backend (Flask, Routing, Grad-CAM logic)
    ├── Procfile                       # Konfigurasi deployment server (Gunicorn)
    ├── README.md                      # Dokumentasi proyek
    ├── requirements.txt               # Daftar pustaka (library) Python
    └── train_models.py                # Script pelatihan ML, evaluasi metrik, & split data

## 🛠️ Persyaratan Sistem & Instalasi
Sebelum menjalankan aplikasi, pastikan Anda telah menginstal **Python 3.8+** di komputer Anda.

1. **Clone repositori ini (atau unduh zip folder proyek):**
   > git clone (https://github.com/permana191/Capstone_Project_kecerdasan_buatan)
   > cd vision-core

2. **Buat dan aktifkan Virtual Environment (Sangat Direkomendasikan):**
   > python -m venv venv
   >
   > *Untuk Windows:*
   > venv\Scripts\activate
   >
   > *Untuk Mac/Linux:*
   > source venv/bin/activate

3. **Instal seluruh pustaka yang dibutuhkan:**
   > pip install -r requirements.txt

## 🚀 Cara Menjalankan Aplikasi
Terdapat dua skrip utama dalam proyek ini yang dapat Anda eksekusi.

**A. Melatih Ulang Model AI (Opsional)**
Jika Anda ingin melatih ulang komparasi ResNet50 dan Xception, menyesuaikan parameter gambar (300x300), atau mencetak ulang *Confusion Matrix* dan kurva ROC, jalankan:
> python train_models.py

*(Skrip ini akan memakan waktu komputasi yang bervariasi tergantung spesifikasi GPU/CPU, lalu memperbarui berkas model di folder `models/` dan grafik di `grafik_evaluasi/`).*

**B. Menjalankan Web Dashboard Utama**
Untuk menghidupkan server Flask dan menggunakan antarmuka deteksi cacat secara langsung:
> python app.py

Setelah server berjalan dan terminal menampilkan pesan `Running on http://127.0.0.1:5000`, buka *browser* web Anda (Chrome/Edge/Firefox) dan akses URL tersebut.

## 📊 Metrik Evaluasi Model AI
Kedua arsitektur diuji menggunakan skenario pembagian data yang ketat. Berdasarkan hasil validasi independen:
* **Xception (Terpilih):** Mencapai konvergensi stabil. *Confusion Matrix* mencatat 453 *True Positives*, 261 *True Negatives*, dan hanya 1 *False Positive*. Meraih skor mutlak **AUC = 1.0000**.
* **ResNet50:** Mengalami *model collapse* (gagal generalisasi) dengan memprediksi seluruh sampel pengujian sebagai kelas tunggal, menghasilkan 262 *False Positives* dan skor acak **AUC = 0.5448**.

## 🔗 Tautan Deployment dan Youtube
Aplikasi ini telah berhasil di-deploy, lolos dari *bug deserialisasi* memori, dan dapat diakses secara publik menggunakan domain komersial tersertifikasi (SSL) melalui tautan berikut:
**[https://www.capstone-project-kecerdasanbuatan-sigit.my.id](https://www.capstone-project-kecerdasanbuatan-sigit.my.id)**
**[[https://youtu.be/RjMmWIbMCXU](https://youtu.be/RjMmWIbMCXU))**

---
*Dikembangkan oleh **Sigit Miraj Permana** (NIM: 301240037)*
