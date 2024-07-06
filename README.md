#PRODIGY_DS_4-Social-Media-Sentiment-Patterns

import pandas as pd
from transformers import pipeline

file_path = 'path_to_your_downloaded_file/twitter_entity_sentiment.csv'
df = pd.read_csv(file_path)


df = df.dropna()
df = df.drop_duplicates()


classifier = pipeline('sentiment-analysis')

def get_transformers_sentiment(text):
    result = classifier(text)[0]
    if result['label'] == 'POSITIVE':
        return 'Positive'
    elif result['label'] == 'NEGATIVE':
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['tweet'].apply(get_transformers_sentiment)


sentiment_counts = df['Sentiment'].value_counts()
import matplotlib.pyplot as plt
y=['Neutral' , 'Negative' , 'Positive']
plt.pie(sentiment_counts , labels=y, autopct='%0.1f%%' )
circle=plt.Circle((0,0),0.4, facecolor='white')
plt.gca().add_patch(circle)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()  