# 🎬 Movie Sentiment Analysis

A **Movie Sentiment Analysis** project that uses Natural Language Processing (NLP) and Machine Learning to classify movie reviews as **Positive** or **Negative**.

## 📌 Project Overview

This project analyzes movie reviews from the **IMDB Dataset** and predicts the sentiment of each review.

The model learns from previously labeled movie reviews and determines whether a new review expresses a positive or negative opinion.

### Example

**Review:**

> "This movie was amazing and I really enjoyed it."

**Prediction:**

> 😊 Positive

**Review:**

> "The movie was boring and disappointing."

**Prediction:**

> 😞 Negative

---

## 🛠️ Technologies Used

* 🐍 Python
* 🐼 Pandas
* 🔢 NumPy
* 🤖 Scikit-learn
* 📊 Matplotlib
* 📝 Natural Language Processing (NLP)
* 📚 IMDB Movie Reviews Dataset

---

## 📂 Project Structure

```text
Movie-Sentiment-Analysis/
│
├── dataset/
│   └── IMDB Dataset.csv
│
├── src/
│   └── sentiment_analysis.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

The project uses the **IMDB Dataset**, which contains movie reviews along with their sentiment labels.

The dataset contains two main columns:

| Column      | Description          |
| ----------- | -------------------- |
| `review`    | Movie review text    |
| `sentiment` | Positive or Negative |

---

## ⚙️ How It Works

The project follows these basic steps:

```text
IMDB Dataset
      ↓
Data Loading
      ↓
Data Cleaning
      ↓
Text Preprocessing
      ↓
Feature Extraction
      ↓
Machine Learning Model
      ↓
Model Training
      ↓
Prediction
      ↓
Positive / Negative
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project

```bash
cd Movie-Sentiment-Analysis
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install required libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Run the Python file:

```bash
python src/sentiment_analysis.py
```

---

## 📈 Sample Output

```text
First 5 rows:

                                              review sentiment
0  One of the other reviewers has mentioned that...  positive
1  A wonderful little production. <br /><br />The...  positive
2  I thought this was a wonderful way to spend time... positive

Dataset shape:
(50000, 2)

Sentiment count:

positive    25000
negative    25000
```

---

## 🎯 Objective

The main objective of this project is to understand how **Natural Language Processing and Machine Learning** can be used to analyze human opinions expressed through text.

---

## 🔮 Future Improvements

* Add a web interface for entering movie reviews.
* Deploy the model online.
* Improve prediction accuracy.
* Try advanced NLP techniques.
* Compare multiple machine learning algorithms.
* Add visualization of sentiment results.
* Use deep learning models such as LSTM or Transformers.

---

## 👨‍💻 Author

**Namdev**

Electronics & Computer Engineering Student

---

## ⭐ If you like this project

If you found this project useful, consider giving the repository a ⭐ on GitHub.
