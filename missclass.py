import streamlit as st
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "mode1.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("Sentiment Analysis")

user_input = st.text_input("Enter your text here")

if st.button("Analyze sentiment"):
    vec_input = vectorizer.transform([user_input])

    prediction = model.predict(vec_input)

    if prediction[0] == 1:
        st.success("A Positive Sentiment")
    else:
        st.warning("A Negative Sentiment")
































# import re
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score ,confusion_matrix, classification_report
# from sklearn.naive_bayes import MultinomialNB
#
#
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
#
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
#
# def preprocess(text):
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z0-9]", " ", text)
#     tokens = word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(word)for word in tokens if word not in stop_words]
#
# st.set_page_config(page_title="Missed Class", layout="wide")
# st.title("Missed Class")
# st.write("Binary class using Naive Bayes")
#
# @st.cache(allow_output_mutation=True)
# def load_data():
#     df = pd.read_csv("missed_class.csv")
#     df = df.loc[df["rating","review"]]
#     df["rating"] = (df["rating"] != 3 ).copy()
#     df["rating"] = (df["rating"] > 3).astype()
#
# df = load_data()
# st.subheader("Review")
# st.dataframe(df())
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
#
#
# tfidf = TfidfVectorizer(max_features= 5000,ngram_range=(1, 2) )
#
# X_train_tfidf = tfidf.fit_transform(X_train)
#
# X_test_tfidf = tfidf.transform(X_test)
#
#
#
# model1 = MultinomialNB()
# model1.fit(X_train_tfidf, y_train)
#
# y_pred= model1.predict(X_test_tfidf)
# accuracy_score = accuracy_score(y_test, y_pred)
# st.subheader("Accuracy")
# st.write(f"Accuracy:{accuracy:2f}")