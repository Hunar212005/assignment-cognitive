import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string

nltk.download('punkt')
nltk.download('stopwords')

paragraph = """Artificial Intelligence is transforming the world of technology. It powers virtual assistants, recommendation systems, and autonomous vehicles. AI helps in medical diagnoses, financial predictions, and language translation. This technology is constantly evolving. The future will be shaped significantly by AI advancements."""

# 1. Lowercase & remove punctuation
cleaned_text = re.sub(r'[^\w\s]', '', paragraph.lower())

# 2. Tokenization
sentences = sent_tokenize(paragraph)
words_split = cleaned_text.split()
words_token = word_tokenize(cleaned_text)

# 3. Compare split() and word_tokenize()
print("Split:", words_split)
print("word_tokenize:", words_token)

# 4. Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words_token if word not in stop_words]

# 5. Frequency distribution
freq_dist = Counter(filtered_words)
print("Word Frequencies (without stopwords):", freq_dist)


from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Only alphabetic words
alpha_words = re.findall(r'\b[a-zA-Z]+\b', cleaned_text)

# 2. Remove stopwords
filtered_alpha = [word for word in alpha_words if word not in stop_words]

# 3. Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered_alpha]

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_alpha]

# 5. Compare
print("Stemmed:", stemmed)
print("Lemmatized:", lemmatized)


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

texts = [
    "Apple releases new iPhone with AI features.",
    "Samsung unveils foldable phone innovation.",
    "Google launches updated Android version."
]

# 1. Bag of Words
cv = CountVectorizer()
bow = cv.fit_transform(texts)
print("Bag of Words:\n", bow.toarray())

# 2. TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(texts)

# 3. Top 3 keywords per text
feature_names = np.array(tfidf.get_feature_names_out())
for i, row in enumerate(tfidf_matrix.toarray()):
    top_indices = row.argsort()[-3:][::-1]
    print(f"Text {i+1} Top 3 Keywords:", feature_names[top_indices])


from sklearn.metrics.pairwise import cosine_similarity

text1 = "Artificial Intelligence is changing how we interact with machines. It enables smart systems."
text2 = "Blockchain provides a secure and decentralized way of handling data and transactions."

# Preprocessing
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return set(word_tokenize(text))

set1, set2 = preprocess(text1), preprocess(text2)

# a. Jaccard Similarity
jaccard = len(set1 & set2) / len(set1 | set2)
print("Jaccard Similarity:", jaccard)

# b. Cosine Similarity
vec = TfidfVectorizer()
tfidf_vectors = vec.fit_transform([text1, text2])
cos_sim = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])[0][0]
print("Cosine Similarity:", cos_sim)


from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

reviews = [
    "The phone is fantastic and has great battery life.",
    "Terrible customer service and buggy software.",
    "An average experience, nothing too good or bad."
]

# Sentiment Analysis
for review in reviews:
    blob = TextBlob(review)
    print(f"Review: {review}")
    print("Polarity:", blob.polarity, "Subjectivity:", blob.subjectivity)
    sentiment = "Positive" if blob.polarity > 0 else "Negative" if blob.polarity < 0 else "Neutral"
    print("Sentiment:", sentiment)

# Word Cloud for Positive Reviews
positive_text = " ".join([r for r in reviews if TextBlob(r).sentiment.polarity > 0])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import numpy as np

text = """Technology is evolving rapidly. Artificial Intelligence and Machine Learning are revolutionizing industries. Future innovations will redefine our daily lives and work patterns."""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Generate sequences
input_sequences = []
for line in text.lower().split('.'):
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        input_sequences.append(tokens[:i+1])

# Padding
max_seq_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))
X, y = input_sequences[:,:-1], input_sequences[:,-1]

# Model
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_seq_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0)

# Text Generation
seed_text = "artificial"
for _ in range(5):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0).argmax(axis=-1)[0]
    output_word = tokenizer.index_word[predicted]
    seed_text += " " + output_word
print("Generated Text:", seed_text)
