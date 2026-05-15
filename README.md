# 💬 Amazon Review Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.0-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

A binary sentiment classifier trained on **21,000+ real Amazon customer reviews**.  
Given any review text, the model predicts whether the sentiment is **Positive** or **Negative** — served through an interactive Streamlit web app.

---

## 🚀 Live Demo

> Run locally with:
> ```bash
> streamlit run app.py
> ```

---

## 📸 App Preview

```
┌─────────────────────────────────────────────┐
│  💬 Amazon Review Sentiment Analyser         │
│                                             │
│  Customer Review:                           │
│  ┌─────────────────────────────────────┐   │
│  │ The product quality was excellent!  │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  [ Analyse Sentiment ]                      │
│                                             │
│  ✅ Sentiment: Positive                     │
│  Model Confidence: 96.3 %                  │
└─────────────────────────────────────────────┘
```

---

## 🧠 How It Works

```
Raw Review Text
      │
      ▼
 Text Cleaning
 (lowercase + remove non-alpha chars)
      │
      ▼
 TF-IDF Vectorisation
 (top 5,000 terms)
      │
      ▼
 Logistic Regression
      │
      ▼
 Positive / Negative + Confidence Score
```

---

## 📊 Model Performance

| Metric        | Score  |
|---------------|--------|
| Accuracy      | 93.8 % |
| Macro F1      | 0.92   |
| Negative F1   | 0.96   |
| Positive F1   | 0.89   |

> Trained on ~16,000 samples · Evaluated on ~4,000 samples (80/20 stratified split)

---

## 📁 Project Structure

```
amazon-sentiment-analysis/
├── Sentiment_Analysis.ipynb   # Full training pipeline & analysis
├── app.py                     # Streamlit web app
├── requirements.txt           # Python dependencies
├── model.pkl                  # Trained Logistic Regression model
├── vectorizer.pkl             # Fitted TF-IDF vectoriser
├── amazon_reviews.csv         # Raw dataset
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the notebook** *(optional — model.pkl is already included)*
```bash
jupyter notebook Sentiment_Analysis.ipynb
```

**4. Launch the web app**
```bash
streamlit run app.py
```

---

## 🗂️ Dataset

- **Source:** Amazon customer reviews scraped from Trustpilot
- **Size:** ~21,000 reviews (2007–2024)
- **Countries:** US, GB, AU, DK and more
- **Label encoding:**
  - `1` → Positive (rating ≥ 4 stars)
  - `0` → Negative (rating ≤ 2 stars)
  - Neutral 3-star reviews excluded

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading & preprocessing |
| `scikit-learn` | TF-IDF vectorisation & Logistic Regression |
| `streamlit` | Interactive web app |
| `pickle` | Model serialisation |
| `re` | Text cleaning |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Your Name**  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_USERNAME)
