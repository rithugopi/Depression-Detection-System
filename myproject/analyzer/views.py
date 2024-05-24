from django.shortcuts import render
import pandas as pd
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
nltk.download('stopwords')
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.downloader.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re



def preprocess_text(data):
    
    data = data.lower()
    
    data = re.sub(r'\d+', '', data)
    
    data = re.sub(r'[^\w\s]', '', data)
    
    data = data.replace('__', '')
    
    data = data.replace('_', '')
   
    data = re.sub(r'\s+', ' ', data)
    
    data = re.sub(r'@[A-Za-z0-9_]+', '', data)
    
    data = re.sub(r'http\S+', '', data)
    
    data = ' '.join([w for w in data.split() if len(w) > 3])
    
    stop_words = set(stopwords.words('english'))
    data = ' '.join([word for word in data.split() if word not in stop_words])
   
    data = ' '.join([Word(word).lemmatize() for word in data.split()])
    
    return data

# def detect_depression_text(data):
#     if isinstance(data, str):  # If input is text
#         preprocessed_text = preprocess_text(data)
#         sentiment_score = sia.polarity_scores(preprocessed_text)
#         if sentiment_score['compound'] > 0.1:  # Negative sentiment
#             return "No depression detected."
#         elif sentiment_score['compound'] < -0.1:
#             return "Depression is detected."
#         else:
#             return "Slight depression detected."


def detect_depression_text(text):
    result = " "  
    preprocessed_text = preprocess_text(text)

    sia = SentimentIntensityAnalyzer()

    # Sentiment analysis
    scores = sia.polarity_scores(preprocessed_text)

    print("Sentiment Scores:", scores)  

    if scores['compound'] >= 0.1:
        sentiment = 'positive'
    elif scores['compound'] <= -0.1:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'


    print("Sentiment Scores:", sentiment)


    # Load and apply saved K-means labels for sentiment clusters
    if sentiment == 'positive':
        with open('kmeans_positive_labels.pkl', 'rb') as f:
            labels = pickle.load(f)
            
    elif sentiment == 'negative':
        with open('kmeans_negative_labels.pkl', 'rb') as f:
            labels = pickle.load(f)    
    else:
        return "The user is slightly depressed"

    # Count the number of texts in each cluster
    cluster_counts = Counter(labels)
    # Get the cluster label with the maximum number of texts
    max_cluster_label = max(cluster_counts, key=cluster_counts.get)

    # Store the maximum cluster label in the 'result' variable
    result = max_cluster_label

    # Update the 'result' variable based on the value of 'max_cluster_label'
    if max_cluster_label == 0:
        result = "The user is depressed"
    elif max_cluster_label == 1:
        result = "The user is slightly depressed"
    elif max_cluster_label == 2:
        result = "The user is normal"

    # Print the result
    return result

    
def detect_depression_username(username):
    analysis_result = " "  # Initialize with a default value
    N = []  # Initialize N as an empty list
    P = []  # Initialize P as an empty list
    sia = SentimentIntensityAnalyzer()
    # Load the dataset
    df = pd.read_csv("testing_dataset.csv")
    # Filter the dataset based on the username
    tweets_df = df[df['username'] == username]

    if tweets_df.empty:
            analysis_result = "Username has not been found."
            return analysis_result
   

            
    else:
            # Preprocessing
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

            # Apply preprocessing function to tweets_df
            tweets_df = preprocess_text(tweets_df, 'text')


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

            
#Step 4: apply saved model
# Check if there are any negative tweets


    neg_result = 0  
    if len(N) > 0:
        # Load the saved K-means labels
        with open('kmeans_negative_labels.pkl', 'rb') as f:
            labels = pickle.load(f)


        # Assuming you have new testing data stored in a variable called `new_data`

        # Apply the labels to the new testing data
        new_labels = labels[:tweets_df.shape[0]]  # Get labels for the same number of tweets as new data

        # Filter out the cluster labels for the negative tweets
        negative_labels = new_labels[tweets_df['tweet_sentiment'] == 'negative']

        
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

        
        
        # Store the maximum cluster label in the 'result' variable
        neg_result = max_cluster_label
        
        # Update the 'result' variable based on the value of 'max_cluster_label'
        if max_cluster_label == 0:
            neg_result = -1
        elif max_cluster_label == 1:
            neg_result = -3
        elif max_cluster_label == 2:
            neg_result = -2

    pos_result = 0  

    if len(P) > 0:

        # Load the saved K-means labels
        with open('kmeans_positive_labels.pkl', 'rb') as f:
            labels = pickle.load(f)

        # Assuming you have new testing data stored in a variable called `new_data`

        # Apply the labels to the new testing data
        new_labels = labels[:tweets_df.shape[0]]  # Get labels for the same number of tweets as new data

        # Filter out the cluster labels for the negative tweets
        positive_labels = new_labels[tweets_df['tweet_sentiment'] == 'positive']

        
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
        # Store the maximum cluster label in the 'result' variable
        pos_result = max_cluster_label
       
        # Update the 'result' variable based on the value of 'max_cluster_label'
        if max_cluster_label == 0:
            pos_result = 1
        elif max_cluster_label == 1:
            pos_result = 3
        elif max_cluster_label == 2:
            pos_result = 2



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
    analysis_result={}

    # Print the categorized weighted average
    if negative_category is not None:
        
        analysis_result["The user is"] = negative_category

    if positive_category is not None and weighted_average != 0.0:
        
        analysis_result["The user is"] = positive_category

# Print the analysis_result dictionary
    print("Analysis Result:", analysis_result)
    
    return analysis_result
    


def index(request):
    analysis_result = " "  
    if request.method == 'POST':
        username = request.POST.get('username')
        text = request.POST.get('text')
        
        if username:  
        
            analysis_result = detect_depression_username(username)
        
        elif text:  
            
            analysis_result = detect_depression_text(text)
            
        else:
            analysis_result = "Please provide either a username or text for depression detection."

        return render(request, 'index.html', {'result': analysis_result})

    return render(request, 'index.html', {'result': analysis_result})




