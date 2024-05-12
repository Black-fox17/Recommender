from main import get_recommendations,title
import streamlit as st
import time
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
filtered_suggestions = filter_suggestions(search_input)

main_placeholder = st.empty()
# Displaying filtered suggestions
if filtered_suggestions:
    st.write("Suggestions:")
    for suggestion in filtered_suggestions:
        st.write(suggestion)
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