# Fake News Detection 📰

## About the Dataset 📊

This dataset consists of news articles labeled as real or fake. It contains the following attributes:

1. **id**: Unique ID for a news article
2. **title**: Title of the news article
3. **author**: Author of the news article
4. **text**: Full text of the article (may be incomplete)
5. **label**: Indicates whether the news article is real or fake
   - `1`: Fake news
   - `0`: Real news

---

## Importing the Dependencies 🛠️

```python
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

# Printing the stopwords in English
print(stopwords.words('english'))
```

---

## Data Pre-processing 🧹

```python
# Loading the dataset into a Pandas DataFrame
news_dataset = pd.read_csv('/content/train.csv')

# Checking the shape of the dataset
news_dataset.shape

# Displaying the first 5 rows
news_dataset.head()

# Checking for missing values
news_dataset.isnull().sum()

# Replacing null values with empty strings
news_dataset = news_dataset.fillna('')

# Merging the author name and news title
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

print(news_dataset['content'])

# Separating features and labels
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

print(X)
print(Y)
```

---

## Stemming (Text Normalization) ✂️

Stemming is the process of reducing words to their root form. Example:

**actor, actress, acting → act**

```python
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Applying stemming to the content column
news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset['content'])

# Updating features and labels
X = news_dataset['content'].values
Y = news_dataset['label'].values

print(X)
print(Y)

# Checking label shape
Y.shape
```

---

## Converting Text to Numerical Data 🔢

```python
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

print(X)
```

---

## Splitting the Dataset (Training & Testing) 📚

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

---

## Training the Model: Logistic Regression 🤖

```python
model = LogisticRegression()

# Training the model
model.fit(X_train, Y_train)
```

---

## Model Evaluation 🏆

### Accuracy Score

```python
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data:', training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data:', test_data_accuracy)
```

---

## Making a Predictive System 🧠

```python
X_new = X_test[3]
prediction = model.predict(X_new)
print(prediction)

if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')

print(Y_test[3])
```

---

## 🔹 Future Scope

This model can be further improved and extended to detect fake news across various platforms, including social media and digital publications. Potential enhancements include:

- Using deep learning techniques (LSTMs, Transformers)
- Expanding the dataset with more diverse sources
- Improving preprocessing techniques for better accuracy

💡 **Feel free to contribute and improve this project!** 🚀

