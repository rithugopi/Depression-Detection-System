from django.test import TestCase

# Create your tests here.
#Libraries
#...........................................snscraper...............................
import pandas as pd
import numpy as np
#...........................................preprocessing............................
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
#..........................................Polarization.............................
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

#..........................................Saved k-mean model....................
import pickle
from collections import Counter
#İmport dataset
df = pd.read_csv("/content/testing_dataset.csv")
df
# Get user input for the username
#username = input("Enter the username to search: ")
username = input("Enter the username to search: ")
# Filter the dataset based on the username
#tweets_df = df[df['username'] == username]
tweets_df = df[df['username'] == username]
if tweets_df.empty:
    print("Username has not been found.")
elif tweets_df.shape[0] < 10:
    print("Your data is not enough for analysis.")
else:
#step 2: Preprocessing

    def preprocess_text(df, column_name):
        # Convert to lowercase]
        df[column_name] = df[column_name].apply(lambda x: " ".join(x.lower() for x in x.split()))
        # Removing numerical values
        df[column_name] = df[column_name].str.replace("\d", "")
        # Removing punctuations
        df[column_name] = df[column_name].str.replace("[^\w\s]", "")
        df[column_name] = df[column_name].str.replace(r"(\x23.* )+", "")
        df[column_name] = df[column_name].str.replace('_', '')
        df[column_name] = df[column_name].str.replace('__', '')
        # Removing double space
        df[column_name] = df[column_name].str.replace("\s+", " ")
        # Removing user
        df[column_name] = df[column_name].str.replace('(@[A-Za-z]+[A-Za-z0-9-_]+)', '') # remove twitted at
        # Removing links
        df[column_name] = df[column_name].str.replace('http\S+', '')
        # Removing small words which are less than given condition
        df[column_name] = df[column_name].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
        # STOPWORDS
        sw = stopwords.words("english")
        df[column_name] = df[column_name].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
        # Lemmatization (forgot converted into forget)
        df[column_name] = df[column_name].apply(lambda x: " ".join([Word(x).lemmatize()]))
        df[column_name+"_tokens"] = df[column_name].apply(lambda x: TextBlob(x).words)
        # Frequency Analysis
        df[column_name+"_frequency"] = df[column_name].apply(lambda x: len(str(x).split(" ")))

        return df
    # apply pre-processing function on'text'
    tweets_df = preprocess_text(tweets_df,'text')
#......................................................................................................................................
#step 3: Polarization
    positive_words_tweet = []
    negative_words_tweet = []
    neutral_words_tweet = []

    positive_sentiment_score = 0
    negative_sentiment_score = 0
    neutral_sentiment_score = 0

    N = []
    P = []



    for index, row in tweets_df.iterrows():
        tweet = row['text']


        # Sentiment analysis for tweet_text column
        scores_tweet = sia.polarity_scores(tweet)
        tweets_df.at[index, 'tweet_positive_score'] = scores_tweet['pos']
        tweets_df.at[index, 'tweet_negative_score'] = scores_tweet['neg']
        tweets_df.at[index, 'tweet_neutral_score'] = scores_tweet['neu']

        if scores_tweet['compound'] > 0.1:
            tweets_df.at[index, 'tweet_sentiment'] = 'positive'
            positive_sentiment_score += 1
            positive_words_tweet.extend(tweet.split())
            P.append(scores_tweet['pos'])  # Store positive score in P

        elif scores_tweet['compound'] < -0.1:
            tweets_df.at[index, 'tweet_sentiment'] = 'negative'
            negative_sentiment_score += 1
            negative_words_tweet.extend(tweet.split())
            N.append(scores_tweet['neg'])  # Store negative score in N

        else:
            tweets_df.at[index, 'tweet_sentiment'] = 'neutral'
            neutral_sentiment_score += 1
            neutral_words_tweet.extend(tweet.split())
    tweets_df.head(10)

    # Filter the tweets_df dataframe to contain only the negative tweets and positive tweets
    negative_df = tweets_df.loc[tweets_df['tweet_sentiment'] == 'negative', ['text', 'tweet_negative_score']]
    positive_df = tweets_df.loc[tweets_df['tweet_sentiment'] == 'positive', ['text', 'tweet_positive_score']]
    #Step 4: apply saved model
    # Check if there are any negative tweets

    neg_result = 0  # Initialize pos_result to 0
    if len(N) > 0:
        # Load the saved K-means labels
        with open('kmeans_negative_labels.pkl', 'rb') as f:
            labels = pickle.load(f)


        # Assuming you have new testing data stored in a variable called `new_data`

        # Apply the labels to the new testing data
        new_labels = labels[:tweets_df.shape[0]]  # Get labels for the same number of tweets as new data

        # Filter out the cluster labels for the negative tweets
        negative_labels = new_labels[tweets_df['tweet_sentiment'] == 'negative']

        # Print the cluster labels for the negative tweets
        for label in negative_labels:
            print("Cluster Label:", label)
        # Assign descriptive names to the cluster labels
        cluster_names = {
            0: "Lightly Depressed",
            1: "Highly Depressed",
            2: "Slightly Depressed"
        }
        # Create a list of descriptive labels for negative tweets
        negative_label_names = [cluster_names[label] for label in negative_labels]

       # Add a new column 'label' to the DataFrame for negative tweets with descriptive names
        tweets_df.loc[tweets_df['tweet_sentiment'] == 'negative', 'label'] = negative_label_names

        # Count the number of tweets in each cluster
        cluster_counts = Counter(negative_labels)

        # Get the cluster label with the maximum number of tweets
        max_cluster_label = max(cluster_counts, key=cluster_counts.get)

        # Print the cluster labels and their corresponding counts
        for label, count in cluster_counts.items():
            cluster_name = cluster_names.get(label, "Unknown")
            print(f"Cluster {label} ({cluster_name}) has {count} tweets.")

        # Print the cluster with the highest number of tweets
        max_cluster_name = cluster_names.get(max_cluster_label, "Unknown")
        print(f"The cluster with the highest number of tweets is Cluster {max_cluster_label} ({max_cluster_name}).")
        # Store the maximum cluster label in the 'result' variable
        neg_result = max_cluster_label
        print(neg_result)
        # Update the 'result' variable based on the value of 'max_cluster_label'
        if max_cluster_label == 0:
            neg_result = -1
        elif max_cluster_label == 1:
            neg_result = -3
        elif max_cluster_label == 2:
            neg_result = -2

        print(neg_result)



  #/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    pos_result = 0  # Initialize pos_result to 0
    if len(P) > 0:

        # Load the saved K-means labels
        with open('kmeans_positive_labels.pkl', 'rb') as f:
            labels = pickle.load(f)

        # Assuming you have new testing data stored in a variable called `new_data`

        # Apply the labels to the new testing data
        new_labels = labels[:tweets_df.shape[0]]  # Get labels for the same number of tweets as new data

        # Filter out the cluster labels for the negative tweets
        positive_labels = new_labels[tweets_df['tweet_sentiment'] == 'positive']

        # Print the cluster labels for the negative tweets
        for label in positive_labels:
            print("Cluster Label:", label)
        # Assign descriptive names to the cluster labels
        cluster_names = {
            0: "lightly Positive",
            1: "Highly Positive",
            2: "SLightly Positive"
        }

        # Create a list of descriptive labels for negative tweets
        positive_label_names = [cluster_names[label] for label in positive_labels]

       # Add a new column 'label' to the DataFrame for negative tweets with descriptive names
        tweets_df.loc[tweets_df['tweet_sentiment'] == 'positive', 'label'] = positive_label_names
        # Count the number of tweets in each cluster
        cluster_counts = Counter(positive_labels)

        # Get the cluster label with the maximum number of tweets
        max_cluster_label = max(cluster_counts, key=cluster_counts.get)

        # Print the cluster labels and their corresponding counts
        for label, count in cluster_counts.items():
            cluster_name = cluster_names.get(label, "Unknown")
            print(f"Cluster {label} ({cluster_name}) has {count} tweets.")

        # Print the cluster with the highest number of tweets
        max_cluster_name = cluster_names.get(max_cluster_label, "Unknown")
        print(f"The cluster with the highest number of tweets is Cluster {max_cluster_label} ({max_cluster_name}).")
        # Store the maximum cluster label in the 'result' variable
        pos_result = max_cluster_label
        print(pos_result)
        # Update the 'result' variable based on the value of 'max_cluster_label'
        if max_cluster_label == 0:
            pos_result = 1
        elif max_cluster_label == 1:
            pos_result = 3
        elif max_cluster_label == 2:
            pos_result = 2

        print(pos_result)

#/////////////////////////////////////////////////////////////////////////////////////////////////////




    # Count the total number of tweets
    total_tweets = len(tweets_df)

    # Count the number of negative tweets
    negative_tweets = len(tweets_df[tweets_df['tweet_sentiment'] == 'negative'])

    # Count the number of positive tweets
    positive_tweets = len(tweets_df[tweets_df['tweet_sentiment'] == 'positive'])

    # Calculate the weights based on the counts
    negative_weight = negative_tweets / total_tweets
    positive_weight = positive_tweets / total_tweets

    # Calculate the weighted average
    weighted_average = (negative_weight * neg_result) + (positive_weight * pos_result)
    print("Weighted Average:", weighted_average)


    # Threshold ranges for negative sentiments
    negative_thresholds = {
        "Lightly Depressed": [-1, 0],
        "Slightly Depressed": [-1.5, -1],
        "Highly Depressed": [-float('inf'), -1.5]
    }

    # Threshold ranges for positive sentiments
    positive_thresholds = {
        "Lightly Positive": [0, 1],
        "Slightly Positive": [1, 2],
        "Highly Positive": [2, float('inf')]
    }

    # Categorize the weighted average for negative sentiments
    negative_category = None
    for category, threshold in negative_thresholds.items():
        if threshold[0] <= weighted_average <= threshold[1]:
            negative_category = category
            break

    if weighted_average == 0.0:
      negative_category = "Lightly Depressed"

    # Categorize the weighted average for positive sentiments
    positive_category = None
    for category, threshold in positive_thresholds.items():
        if threshold[0] <= weighted_average <= threshold[1]:
            positive_category = category
            break

    # Print the categorized weighted average
    if negative_category is not None:
        print("Weighted Average (Negative):", weighted_average, "Category:", negative_category)

    if positive_category is not None and weighted_average != 0.0:
        print("Weighted Average (Positive):", weighted_average, "Category:", positive_category)


tweets_df
if tweets_df.empty:
    print("Username has not been found.")
elif tweets_df.shape[0] < 10:
    print("Your data is not enough for analysis.")
else:
#step 2: Preprocessing

    def preprocess_text(df, column_name):
        # Convert to lowercase]
        df[column_name] = df[column_name].apply(lambda x: " ".join(x.lower() for x in x.split()))
        # Removing numerical values
        df[column_name] = df[column_name].str.replace("\d", "")
        # Removing punctuations
        df[column_name] = df[column_name].str.replace("[^\w\s]", "")
        df[column_name] = df[column_name].str.replace(r"(\x23.* )+", "")
        df[column_name] = df[column_name].str.replace('_', '')
        df[column_name] = df[column_name].str.replace('__', '')
        # Removing double space
        df[column_name] = df[column_name].str.replace("\s+", " ")
        # Removing user
        df[column_name] = df[column_name].str.replace('(@[A-Za-z]+[A-Za-z0-9-_]+)', '') # remove twitted at
        # Removing links
        df[column_name] = df[column_name].str.replace('http\S+', '')
        # Removing small words which are less than given condition
        df[column_name] = df[column_name].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
        # STOPWORDS
        sw = stopwords.words("english")
        df[column_name] = df[column_name].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
        # Lemmatization (forgot converted into forget)
        df[column_name] = df[column_name].apply(lambda x: " ".join([Word(x).lemmatize()]))
        df[column_name+"_tokens"] = df[column_name].apply(lambda x: TextBlob(x).words)
        # Frequency Analysis
        df[column_name+"_frequency"] = df[column_name].apply(lambda x: len(str(x).split(" ")))

        return df
    # apply pre-processing function on'text'
    tweets_df = preprocess_text(tweets_df,'text')
#......................................................................................................................................
#step 3: Polarization
    positive_words_tweet = []
    negative_words_tweet = []
    neutral_words_tweet = []

    positive_sentiment_score = 0
    negative_sentiment_score = 0
    neutral_sentiment_score = 0

    N = []
    P = []



    for index, row in tweets_df.iterrows():
        tweet = row['text']


        # Sentiment analysis for tweet_text column
        scores_tweet = sia.polarity_scores(tweet)
        tweets_df.at[index, 'tweet_positive_score'] = scores_tweet['pos']
        tweets_df.at[index, 'tweet_negative_score'] = scores_tweet['neg']
        tweets_df.at[index, 'tweet_neutral_score'] = scores_tweet['neu']

        if scores_tweet['compound'] > 0.1:
            tweets_df.at[index, 'tweet_sentiment'] = 'positive'
            positive_sentiment_score += 1
            positive_words_tweet.extend(tweet.split())
            P.append(scores_tweet['pos'])  # Store positive score in P

        elif scores_tweet['compound'] < -0.1:
            tweets_df.at[index, 'tweet_sentiment'] = 'negative'
            negative_sentiment_score += 1
            negative_words_tweet.extend(tweet.split())
            N.append(scores_tweet['neg'])  # Store negative score in N

        else:
            tweets_df.at[index, 'tweet_sentiment'] = 'neutral'
            neutral_sentiment_score += 1
            neutral_words_tweet.extend(tweet.split())
    tweets_df.head(10)

    # Filter the tweets_df dataframe to contain only the negative tweets and positive tweets
    negative_df = tweets_df.loc[tweets_df['tweet_sentiment'] == 'negative', ['text', 'tweet_negative_score']]
    positive_df = tweets_df.loc[tweets_df['tweet_sentiment'] == 'positive', ['text', 'tweet_positive_score']]
    #Step 4: apply saved model
    # Check if there are any negative tweets

    neg_result = 0  # Initialize pos_result to 0
    if len(N) > 0:
        # Load the saved K-means labels
        with open('kmeans_negative_labels.pkl', 'rb') as f:
            labels = pickle.load(f)


        # Assuming you have new testing data stored in a variable called `new_data`

        # Apply the labels to the new testing data
        new_labels = labels[:tweets_df.shape[0]]  # Get labels for the same number of tweets as new data

        # Filter out the cluster labels for the negative tweets
        negative_labels = new_labels[tweets_df['tweet_sentiment'] == 'negative']

        # Print the cluster labels for the negative tweets
        for label in negative_labels:
            print("Cluster Label:", label)
        # Assign descriptive names to the cluster labels
        cluster_names = {
            0: "Lightly Depressed",
            1: "Highly Depressed",
            2: "Slightly Depressed"
        }
        # Create a list of descriptive labels for negative tweets
        negative_label_names = [cluster_names[label] for label in negative_labels]

       # Add a new column 'label' to the DataFrame for negative tweets with descriptive names
        tweets_df.loc[tweets_df['tweet_sentiment'] == 'negative', 'label'] = negative_label_names

        # Count the number of tweets in each cluster
        cluster_counts = Counter(negative_labels)

        # Get the cluster label with the maximum number of tweets
        max_cluster_label = max(cluster_counts, key=cluster_counts.get)

        # Print the cluster labels and their corresponding counts
        for label, count in cluster_counts.items():
            cluster_name = cluster_names.get(label, "Unknown")
            print(f"Cluster {label} ({cluster_name}) has {count} tweets.")

        # Print the cluster with the highest number of tweets
        max_cluster_name = cluster_names.get(max_cluster_label, "Unknown")
        print(f"The cluster with the highest number of tweets is Cluster {max_cluster_label} ({max_cluster_name}).")
        # Store the maximum cluster label in the 'result' variable
        neg_result = max_cluster_label
        print(neg_result)
        # Update the 'result' variable based on the value of 'max_cluster_label'
        if max_cluster_label == 0:
            neg_result = -1
        elif max_cluster_label == 1:
            neg_result = -3
        elif max_cluster_label == 2:
            neg_result = -2

        print(neg_result)



  #/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    pos_result = 0  # Initialize pos_result to 0
    if len(P) > 0:

        # Load the saved K-means labels
        with open('kmeans_positive_labels.pkl', 'rb') as f:
            labels = pickle.load(f)

        # Assuming you have new testing data stored in a variable called `new_data`

        # Apply the labels to the new testing data
        new_labels = labels[:tweets_df.shape[0]]  # Get labels for the same number of tweets as new data

        # Filter out the cluster labels for the negative tweets
        positive_labels = new_labels[tweets_df['tweet_sentiment'] == 'positive']

        # Print the cluster labels for the negative tweets
        for label in positive_labels:
            print("Cluster Label:", label)
        # Assign descriptive names to the cluster labels
        cluster_names = {
            0: "lightly Positive",
            1: "Highly Positive",
            2: "SLightly Positive"
        }

        # Create a list of descriptive labels for negative tweets
        positive_label_names = [cluster_names[label] for label in positive_labels]

       # Add a new column 'label' to the DataFrame for negative tweets with descriptive names
        tweets_df.loc[tweets_df['tweet_sentiment'] == 'positive', 'label'] = positive_label_names
        # Count the number of tweets in each cluster
        cluster_counts = Counter(positive_labels)

        # Get the cluster label with the maximum number of tweets
        max_cluster_label = max(cluster_counts, key=cluster_counts.get)

        # Print the cluster labels and their corresponding counts
        for label, count in cluster_counts.items():
            cluster_name = cluster_names.get(label, "Unknown")
            print(f"Cluster {label} ({cluster_name}) has {count} tweets.")

        # Print the cluster with the highest number of tweets
        max_cluster_name = cluster_names.get(max_cluster_label, "Unknown")
        print(f"The cluster with the highest number of tweets is Cluster {max_cluster_label} ({max_cluster_name}).")
        # Store the maximum cluster label in the 'result' variable
        pos_result = max_cluster_label
        print(pos_result)
        # Update the 'result' variable based on the value of 'max_cluster_label'
        if max_cluster_label == 0:
            pos_result = 1
        elif max_cluster_label == 1:
            pos_result = 3
        elif max_cluster_label == 2:
            pos_result = 2

        print(pos_result)

#/////////////////////////////////////////////////////////////////////////////////////////////////////




    # Count the total number of tweets
    total_tweets = len(tweets_df)

    # Count the number of negative tweets
    negative_tweets = len(tweets_df[tweets_df['tweet_sentiment'] == 'negative'])

    # Count the number of positive tweets
    positive_tweets = len(tweets_df[tweets_df['tweet_sentiment'] == 'positive'])

    # Calculate the weights based on the counts
    negative_weight = negative_tweets / total_tweets
    positive_weight = positive_tweets / total_tweets

    # Calculate the weighted average
    weighted_average = (negative_weight * neg_result) + (positive_weight * pos_result)
    print("Weighted Average:", weighted_average)


    # Threshold ranges for negative sentiments
    negative_thresholds = {
        "Lightly Depressed": [-1, 0],
        "Slightly Depressed": [-1.5, -1],
        "Highly Depressed": [-float('inf'), -1.5]
    }

    # Threshold ranges for positive sentiments
    positive_thresholds = {
        "Lightly Positive": [0, 1],
        "Slightly Positive": [1, 2],
        "Highly Positive": [2, float('inf')]
    }

    # Categorize the weighted average for negative sentiments
    negative_category = None
    for category, threshold in negative_thresholds.items():
        if threshold[0] <= weighted_average <= threshold[1]:
            negative_category = category
            break

    if weighted_average == 0.0:
      negative_category = "Lightly Depressed"

    # Categorize the weighted average for positive sentiments
    positive_category = None
    for category, threshold in positive_thresholds.items():
        if threshold[0] <= weighted_average <= threshold[1]:
            positive_category = category
            break

    # Print the categorized weighted average
    if negative_category is not None:
        print("Weighted Average (Negative):", weighted_average, "Category:", negative_category)

    if positive_category is not None and weighted_average != 0.0:
        print("Weighted Average (Positive):", weighted_average, "Category:", positive_category)

tweets_df