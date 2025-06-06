# sentiment-anaalysis
This project focuses on sentiment analysis 
# 🎯 Sentiment Analysis using Machine Learning in Python

## 📘 Project Overview

Sentiment Analysis aims to determine a writer’s **attitude or emotional tone** towards a topic, product, or brand. This project uses **Natural Language Processing (NLP)** techniques and **Machine Learning** algorithms to classify text data as **Positive** or **Negative**.

We analyze movie reviews from IMDb and build a machine learning pipeline to automatically predict sentiments based on the review content.

---

## 🧾 Dataset Description

- The dataset contains **40,000 IMDb movie reviews** with two columns:
  - **`text`**: Review content
  - **`label`**: Sentiment class (0 = Negative, 1 = Positive)

We limit our analysis to **10,000 records** for better processing.

```python
data.shape  # Original: (40000, 2)
data = data.iloc[:10000, :]
🧼 Data Preprocessing
Text is cleaned using the following steps:

Remove HTML tags

Extract emojis

Remove special characters, punctuation, and symbols

Lowercase conversion

Remove stopwords

Tokenization and Stemming

python
Copy
Edit
def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoji_pattern.findall(text)
    text = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]
    return " ".join(text)
📊 Data Visualization
We analyze and plot the distribution of sentiments and top frequent words:

Label distribution (Pie Chart)

Most common words in positive and negative reviews using bar graphs.

python
Copy
Edit
positivedata = data[data['label'] == 1]['text']
negdata = data[data['label'] == 0]['text']
✏️ Feature Extraction
We use TF-IDF Vectorizer to convert text data into numerical features for training the model.

python
Copy
Edit
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None,
                        use_idf=True, norm='l2', smooth_idf=True)
x = tfidf.fit_transform(data.text)
y = data.label.values
🤖 Model Training
We train a Logistic Regression model with 50% of the data for training and 50% for testing.

python
Copy
Edit
from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(cv=6, scoring='accuracy', random_state=0,
                           n_jobs=-1, verbose=3, max_iter=500).fit(X_train, y_train)
✅ Model Evaluation
python
Copy
Edit
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
Accuracy: 87.56%

🧠 Predicting Sentiment
python
Copy
Edit
def prediction(comment):
    preprocessed_comment = preprocessing(comment)
    comment_list = [preprocessed_comment]
    comment_vector = tfidf.transform(comment_list)
    prediction = clf.predict(comment_vector)[0]
    return prediction
💾 Saving the Model
We use Pickle to serialize the model and TF-IDF vectorizer.

python
Copy
Edit
import pickle
pickle.dump(clf, open('clf.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
💬 Example Predictions
python
Copy
Edit
prediction = prediction("The movie had great visuals and emotional depth.")
# Output: positive comment
📌 Key Takeaways
Logistic Regression is effective for binary sentiment classification.

Preprocessing greatly improves model performance.

The model can be used to analyze reviews, social media content, etc.

🚀 Future Improvements
Use deep learning models like LSTM, BERT.

Deploy the model with a Flask API or Web App.

Extend to multilingual sentiment analysis.

👩‍💻 Author
J. Sreeja Rayal
Reg No: 23BCE7013
