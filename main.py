import pandas as pd
import random

with open(r"C:\Users\owner\Desktop\Projects\Movie recommender\full_movies.csv",'rb') as f:
    df = pd.read_csv(f)

def title():
    return df["title"].to_list()
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df['overview'] = df['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['new_title']).drop_duplicates()
#import regex as re
def preprocess_title(title):
    # Convert to lowercase and remove special characters
    title = title.lower().replace("\'","")
    return title

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    new_title = title
    title = preprocess_title(title)
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    random.shuffle(sim_scores)
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    result = df[['title', 'overview', 'release_date']].iloc[movie_indices]

    return result,df["overview"][idx],new_title
