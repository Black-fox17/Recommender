import streamlit as st
import time

import pandas as pd
import random
def main():
    with open("full_movies.csv",'rb') as f:
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
    # Predefined list of suggestions
    suggestions = title()
    def response_generator(text):
        for word in text.split():
            yield word + " "
            time.sleep(0.05)
    
    # Function to filter suggestions based on user input
    def filter_suggestions(user_input):
        return [suggestion for suggestion in suggestions if user_input.lower() in suggestion.lower()]
    
    st.title("🎬 Movie Recommender")
    
    # Introduction
    st.subheader("Welcome to our Movie Recommender!")
    response = st.write_stream(response_generator("This recommender suggests similar movies based on your search."))
    
    # # Streamlit app layout
    # st.title("Search Box with Suggestions")
    st.warning("Please enter as it shows under the suggestions and scroll down to See the Search button.")
    # Search box component
    search_input = st.text_input("Search for a Movie:", "")
    # Filtering suggestions based on user input
    #filtered_suggestions = filter_suggestions(search_input)
    
    main_placeholder = st.empty()
    # Displaying filtered suggestions
    # if filtered_suggestions:
    #     st.write("Suggestions:")
    #     for suggestion in filtered_suggestions:
    #         st.write(suggestion)
    search_button = st.button("Search")
    if search_button:
        try:
            result,overview,new_title = get_recommendations(search_input)
    
            st.header("🍿 Recommendations:")
            st.subheader("Double tap on overview to expand")
            st.write_stream(response_generator(f"Here are some recommendations based on {new_title}:\n{overview}"))
            st.dataframe(result[['title', 'overview', 'release_date']])
        except:
            st.warning("Sorry we do not have data based on your provided information\nPlease Try again with another Movie.")

if __name__ == "__main__":
    main()

