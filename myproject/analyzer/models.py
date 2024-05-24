from django.db import models

# Create your models here.


#code of if, else in positive negative
if tweets_df.empty:
    print("Username has not been found.")
elif tweets_df.shape[0] < 10:
    print("Your data is not enough for analysis.")
else:
#step 2: Preprocessing

    def preprocess_text(df, column_name):
        # Convert to lowercase
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
        print(f"The cluster with the highest number of tweets is Cluster {max_cluster_label} ({max_cluster_name}).") #}}}}}}}}}}}}}}}}}}}


        # Display negative words used in the respective cluster
        if max_cluster_label == 0:  # Lightly Depressed cluster
           negative_tweets_lightly = tweets_df[(tweets_df['tweet_sentiment'] == 'negative') & (new_labels == max_cluster_label)]
           unique_negative_words_lightly = set(negative_tweets_lightly['text'].apply(lambda tweet: tweet.split()).sum())
           print("Negative words used only in tweets of Lightly Depressed cluster:")
           print(unique_negative_words_lightly)

        elif max_cluster_label == 1:  # Highly Depressed cluster
           negative_tweets_highly = tweets_df[(tweets_df['tweet_sentiment'] == 'negative') & (new_labels == max_cluster_label)]
           unique_negative_words_highly = set(negative_tweets_highly['text'].apply(lambda tweet: tweet.split()).sum())
           print("Negative words used only in tweets of Highly Depressed cluster:")
           print(unique_negative_words_highly)
           print(unique_negative_words_highly)

        elif max_cluster_label == 2:  # Slightly Depressed cluster
           negative_tweets_slightly = tweets_df[(tweets_df['tweet_sentiment'] == 'negative') & (new_labels == max_cluster_label)]
           unique_negative_words_slightly = set(negative_tweets_slightly['text'].apply(lambda tweet: tweet.split()).sum())
           print("Negative words used only in tweets of Slightly Depressed cluster:")
           print(unique_negative_words_slightly)

    else:

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

        # Display negative words used in the respective cluster
        if max_cluster_label == 0:  # Lightly Positive cluster
           positive_tweets_lightly = tweets_df[(tweets_df['tweet_sentiment'] == 'positive') & (new_labels == max_cluster_label)]
           unique_positive_words_lightly = set(positive_tweets_lightly['text'].apply(lambda tweet: tweet.split()).sum())
           print("Positive words used only in tweets of Lightly Positive cluster:")
           print(unique_positive_words_lightly)

        elif max_cluster_label == 1:  # Highly Positive cluster
           positive_tweets_highly = tweets_df[(tweets_df['tweet_sentiment'] == 'positive') & (new_labels == max_cluster_label)]
           unique_positive_words_highly = set(positive_tweets_highly['text'].apply(lambda tweet: tweet.split()).sum())
           print("Positive words used only in tweets of Highly Positive cluster:")
           print(unique_positive_words_highly)

        elif max_cluster_label == 2:  # Slightly Positive cluster
           positive_tweets_slightly = tweets_df[(tweets_df['tweet_sentiment'] == 'positive') & (new_labels == max_cluster_label)]
           unique_positive_words_slightly = set(positive_tweets_slightly['text'].apply(lambda tweet: tweet.split()).sum())
           print("Positive words used only in tweets of Slightly Depressed cluster:")
           print(unique_positive_words_slightly)

  