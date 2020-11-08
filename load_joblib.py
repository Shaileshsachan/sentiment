from joblib import load

text = ['I love coding in Pyhon']

pipeline = load("Classification.joblib")

print(pipeline.predict(text))
