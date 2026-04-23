# 🛒 Amazon Product Review — Sentiment Classifier

A Natural Language Processing (NLP) project that classifies Amazon product reviews as **Positive** or **Negative** using TF-IDF Vectorization and Logistic Regression.

---

## 📌 Project Overview

Online platforms like Amazon receive millions of product reviews daily. Manually reading and categorizing them is impossible at scale. This project builds a machine learning pipeline that automatically detects the **sentiment** of a review — helping businesses understand customer feedback efficiently.

---

## 🎯 Objectives

- Clean and preprocess raw text review data
- Convert text into numerical features using TF-IDF
- Train a Logistic Regression classifier to predict sentiment
- Evaluate model performance using accuracy, precision, recall, and F1-score
- Visualize patterns using WordClouds, confusion matrix, and confidence distributions

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| **Source** | [Amazon Reviews — Kaggle](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) |
| **Size** | 3.6 million reviews (we use 50,000 for training) |
| **Format** | CSV (`train.csv`, `test.csv`) |
| **Labels** | 1 = Negative, 2 = Positive |
| **Columns** | `label`, `title`, `review` |

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.10` | Core language |
| `Pandas` | Data loading and manipulation |
| `NumPy` | Numerical operations |
| `Scikit-learn` | TF-IDF, Logistic Regression, Evaluation |
| `Matplotlib` | Plotting charts |
| `Seaborn` | Statistical visualizations |
| `WordCloud` | Word frequency visualization |
| `Jupyter Notebook` | Development environment |

---

## 📁 Project Structure

```
amazon-sentiment-classifier/
│
├── amazon_sentiment_classifier.ipynb   # Main notebook
├── train.csv                           # Training data (from Kaggle)
├── test.csv                            # Test data (from Kaggle)
├── README.md                           # Project documentation
│
├── class_distribution.png             # Auto-saved charts
├── review_length_distribution.png
├── wordclouds.png
├── top_words.png
├── confusion_matrix.png
├── influential_words.png
└── confidence_distribution.png
```

---

## ⚙️ How to Run

### 1. Clone or Download this Repository
```bash
git clone https://github.com/yourusername/amazon-sentiment-classifier.git
cd amazon-sentiment-classifier
```

### 2. Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud jupyter
```

### 3. Download the Dataset
- Go to 👉 https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
- Download and place `train.csv` and `test.csv` in the project folder

### 4. Launch Jupyter from the Project Folder
```bash
jupyter notebook
```

### 5. Open and Run the Notebook
- Open `amazon_sentiment_classifier.ipynb`
- Click **Kernel → Restart & Run All**

---

## 🔄 Pipeline

```
Raw CSV Data
     ↓
Data Loading & Sampling (50,000 reviews)
     ↓
Text Cleaning (lowercase, remove HTML, punctuation)
     ↓
Train-Test Split (80% / 20%)
     ↓
TF-IDF Vectorization (30,000 features, unigrams + bigrams)
     ↓
Logistic Regression Training
     ↓
Evaluation (Accuracy, Confusion Matrix, Classification Report)
     ↓
Custom Review Prediction
```

---

## 📊 Results

| Metric | Score |
|---|---|
| **Accuracy** | ~93–94% |
| **Precision (Positive)** | ~94% |
| **Recall (Positive)** | ~93% |
| **F1-Score** | ~93% |

---

## 📸 Visualizations

### Sentiment Word Clouds
> Green = Positive reviews &nbsp;&nbsp;|&nbsp;&nbsp; Red = Negative reviews

![WordClouds](wordclouds.png)

---

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

---

### Most Influential Words
![Influential Words](influential_words.png)

---

## 🔑 Key Findings

1. **High Accuracy** — TF-IDF + Logistic Regression achieves ~93% accuracy, making it a strong NLP baseline without deep learning
2. **Balanced Dataset** — Nearly equal positive and negative reviews, so the model is unbiased
3. **Review Length** — Negative reviews tend to be slightly longer; unhappy customers write more detail
4. **Predictive Words** — Words like *excellent*, *love*, *perfect* strongly signal positive sentiment; *waste*, *broken*, *terrible* signal negative
5. **Bigrams Matter** — Word pairs like *not good* or *highly recommend* capture context that single words miss
6. **High Confidence** — Most predictions are made with 85%+ confidence, showing clear decision boundaries

---

## 🚀 Possible Extensions

- Compare with `MultinomialNB`, `LinearSVC`, or `Random Forest`
- Add a `VADER` rule-based baseline for comparison
- Deploy as a simple web app using `Streamlit`
- Fine-tune a `BERT` model for even higher accuracy

---

## 🔗 Lab Concept Mapping

This project directly extends concepts from the Data Science lab:

| Lab Assignment | Concept Used Here |
|---|---|
| A6 — Spam Classifier | Logistic Regression, classification pipeline |
| A4 — Titanic EDA | Data preprocessing, handling missing values |
| A3 — Open Dataset | Aggregate statistics, grouping, EDA |
| A8 — Uber Fares | Model evaluation metrics (accuracy, precision, recall) |

---

## 👤 Author

**Your Name**  
B.Sc. / B.Tech — Data Science  
GitHub: [@yourusername](https://github.com/yourusername)

---

## 📄 License

This project is for educational purposes as part of a Data Science course portfolio.
