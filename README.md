# Anime Recommendation System 🎌

A Content-Based Anime Recommendation System built using Python and Machine Learning techniques. This project recommends similar anime based on genre similarity using CountVectorizer and Cosine Similarity.

---

# 📌 Features

- Anime recommendation based on genres
- Content-based filtering approach
- Text vectorization using CountVectorizer
- Similarity calculation using Cosine Similarity
- Data preprocessing and cleaning
- Beginner-friendly ML project

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

---

# 📂 Dataset

This project uses:
- `anime.csv`
- `rating.csv`

Dataset contains anime information such as:
- Anime name
- Genre
- Ratings
- Anime ID

---

# ⚙️ How It Works

## 1. Load Dataset
The anime dataset is loaded using Pandas.

```python
anime = pd.read_csv("anime.csv")
```

## 2. Data Cleaning
- Remove null values
- Remove duplicate records
- Keep required columns only

## 3. Genre Processing
Genres are cleaned and converted into text format suitable for vectorization.

## 4. Vectorization
Genres are transformed into numerical vectors using CountVectorizer.

```python
cv = CountVectorizer(max_features=3000, stop_words='english')
vectors = cv.fit_transform(anime['genre'])
```

## 5. Similarity Calculation
Cosine Similarity is used to find similar anime.

```python
similarity = cosine_similarity(vectors)
```

## 6. Recommendation Function
The system recommends top similar anime based on genre similarity.

```python
recommend("Naruto")
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/Afreen-7/Anime-Recommendation-system.git
```

## Install Dependencies

```bash
pip install pandas numpy scikit-learn
```

## Run Project

```bash
python app.py
```

---

# 📸 Example Output

Input:

```python
recommend("Naruto")
```

Output:

```python
Bleach
One Piece
Fairy Tail
Dragon Ball Z
Hunter x Hunter
```

---

# 🔮 Future Improvements

- Hybrid Recommendation System
- Collaborative Filtering
- Streamlit Web App
- Personalized Recommendations
- Recommendation based on ratings and user preferences

---

# 👩‍💻 Author

Afreen Tariq

- Data Science Student
- Machine Learning Enthusiast
- Python Developer

---

# ⭐ GitHub Repository

If you like this project, give it a star ⭐ on GitHub.
