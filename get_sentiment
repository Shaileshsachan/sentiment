import tweepy
import time
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from joblib import load

pd.set_option('display.max_colwidth', 1000)
api_key = 'QQElOU88MLKvJNihWjmgCA7uS'
api_secret_key = 'cHzQIRRwScrxtFub22TJxM6bR2DQOaKargDnhovR1CifZiaiak'

access_token = '1300702866034667520-bhlj531VqYkQbqbOWjK5tI61qEZvPP'
access_token_secret = 'If4gu9MdBVHZH5Pwp9mtCVtATUStB3JPVLKR5Fy3waQxB'

authentication = tweepy.OAuthHandler(api_key, api_secret_key)
authentication.set_access_token(access_token, access_token_secret)

api = tweepy.API(authentication, wait_on_rate_limit=True)

def get_related_tweets(text_query):
    tweets_list = []
    count = 50
    try:
        for tweet in api.search(q=text_query, count=count):
            print(tweet.text)
            tweets_list.append({'created_at' : tweet.created_at,
                                'tweet_id' : tweet.id,
                                'tweet_text' : tweet.text})
        return pd.DataFrame.from_dict(tweets_list)
    except BaseException as e:
        print('failed on status,', str(e))
        time.sleep(3)

pipeline = load('Classification.joblib')


def requestResults(name):
    tweets = get_related_tweets(name)
    tweets['prediction'] = pipeline.predict(tweets['tweet_text'])
    data = str(tweets.prediction.value_counts()) + '\n\n'
    return data + str(tweets)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success', name=user))


@app.route('/success/<name>')
def success(name):
    return "<xmp>" + str(requestResults(name)) + " </xmp> "


if __name__ == '__main__':
    app.run(debug=True)
