# BMI Analyzer Pro

**Universitas AMIKOM Yogyakarta — Data Mining Project**

---

## Cara Menjalankan

### Lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Google Colab
1. Upload semua file ke Colab
2. Jalankan notebook `BMI_Analyzer_Pro.ipynb` sel per sel
3. Di cell terakhir, aktifkan kode ngrok dan isi token dari https://dashboard.ngrok.com

### Streamlit Cloud (Deploy Gratis)
1. Push folder ini ke GitHub (repository publik atau privat)
2. Buka https://share.streamlit.io
3. Hubungkan GitHub repo → pilih `app.py` sebagai entry point → Deploy

---

## File yang Diperlukan
| File | Keterangan |
|------|------------|
| `app.py` | Aplikasi Streamlit utama |
| `bmi.csv` | Dataset (500 sampel, 4 kolom) |
| `requirements.txt` | Daftar dependencies |
| `.streamlit/config.toml` | Konfigurasi tema |
| `BMI_Analyzer_Pro.ipynb` | Notebook lengkap (EDA + Training + App) |

---

## Fitur Aplikasi

- **Tab 1 — Kalkulator**: BMI gauge interaktif, body silhouette SVG, prediksi ML realtime
- **Tab 2 — Analisis Data**: EDA lengkap dari dataset asli (distribusi, scatter, heatmap, box plot, auto-insights)
- **Tab 3 — Model ML**: Latih 6 model, bandingkan akurasi/F1/precision/recall, confusion matrix, feature importance
- **Tab 4 — Rekomendasi**: Tips personal diet, olahraga, hidrasi, tidur, lifestyle + BMI journey tracker
- **Tab 5 — Batch Prediksi**: Upload CSV, prediksi massal, download hasil

---

## Model yang Digunakan
- Random Forest (default terbaik: ~98% akurasi)
- Gradient Boosting
- SVM (RBF)
- KNN
- Logistic Regression
- XGBoost (opsional, jika terinstall)
