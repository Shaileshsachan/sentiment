import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from joblib import dump

data = pd.read_csv('twitter_sentiments.csv')
print(data.head())

train, test = train_test_split(data, test_size = 0.2, stratify=data['label'], random_state=21)
print(train.shape, test.shape)

tfidf_V = TfidfVectorizer(lowercase=True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)
tfidf_V.fit(train.tweet)

train_idf = tfidf_V.transform(train.tweet)
test_idf = tfidf_V.transform(test.tweet)

LR_model = LogisticRegression()

LR_model.fit(train_idf, train.label)
train_predict = LR_model.predict(train_idf)
test_predict = LR_model.predict(test_idf)

score_train = f1_score(y_true=train.label, y_pred=train_predict)
print(score_train)

score_test = f1_score(y_true=test.label, y_pred=test_predict)
print(score_test)

pipeline = Pipeline(steps=[('tfidf',TfidfVectorizer(lowercase=True,
                                                    max_features=1000,
                                                    stop_words=ENGLISH_STOP_WORDS)),
                           ('model', LogisticRegression())])

pipeline.fit(train.tweet, train.label)

text = ["Your are racist"]
print(pipeline.predict(text))

dump(pipeline, filename="Classification.joblib")
