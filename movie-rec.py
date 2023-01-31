import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


### Data Preprocessing

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
#print(movies.head())
#print(credits.head())

# merge two datasets
movies = movies.merge(credits,on='title')
movies.head(1)
print(movies.head())

# Remove unnecessary columns
movies = movies[['id','title','overview','genres','keywords','cast','crew']]
#movies.head()

# Check Missing data
print(movies.describe())
print(movies.info())

# Remove Missing data
movies.dropna(inplace=True)

# Check for duplicated data
movies.duplicated().sum()

## Preprocess ‘genres’ column
#print(movies.iloc[0].genres) # the genere column is a list of dictionaries

# convert 'generes' into the list of names
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
#print(movies['genres'])

## Preprocess 'keywords' column
movies['keywords'] = movies['keywords'].apply(convert)
#print(movies['keywords'])

## Preprocess 'cast' column
#print(movies['cast'])
# convers to the list of names and get the top three casts
def convert_top3(obj):
    L = []
    c=0
    for i in ast.literal_eval(obj):
        if(c!=3):
            L.append(i['name'])
            c+=1
        else:
            break
    return L
movies['cast'] = movies['cast'].apply(convert_top3)
#print(movies['cast'])

## Preprocess 'crew' column
'''‘crew’ contains a list of dictionaries with details of 
the job of crew members. We need the ‘director’ name from 
this data, hence we need to extract the job ‘Director’.'''
def fetchDirector(obj):
    L = []
    for i in ast.literal_eval(obj):
        if(i['job']=='Director'):
            L.append(i['name'])
            break
    return L
movies['crew'] = movies['crew'].apply(fetchDirector)
#print(movies['crew'])

## Preprocess 'overview' column
'''‘overview’ is actually a string and all other columns are
 lists. Hence, we will convert the string to the list.'''
movies['overview'] = movies['overview'].apply(lambda x:x.split())
print(movies['overview'])

'''the columns ‘keywords’, ‘cast’, and ‘crew’, the content is 
separated by white spaces and we don’t want them to scatter 
while we concatenate and build our model otherwise it will lead 
to low efficiency.'''
# Replace the ” ”(white space) with “” for every column

columns = ['genres', 'keywords', 'cast', 'crew']
for col in columns:
    movies[col] = movies[col].apply(lambda x:[i.replace(" ","") for i in x])

#print(movies.head())

# Concatenating the columns into one ‘tags’
movies['tags'] = movies['overview'] + movies['genres'] + movies['cast'] + movies['crew']
movies.head()

# New dataframe will contain 3 columns: ‘id’, ‘title’, ‘tags’
new_df = movies[['id','title','tags']]
new_df.head()

'''the ‘tags’ should be a paragraph, i.e. a string to make it 
understood by our model.'''
# convert 'tags' to a paragraph
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
# convert all words in 'tags' into lower case alphabets
new_df['tags'] =  new_df['tags'].apply(lambda x:x.lower())

#print(new_df['tags'][0])

### Building the Model

# remove commoner morphological and inflexional endings from words
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

# vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
#print(cv.get_feature_names_out())

# find similarity between vectors(movies)
similarity = cosine_similarity(vectors)
print(similarity[0])

'''Recommender System Algo: First, we will find the 
similarity vector of the movie provided in the input. 
Then we’ll sort these numbers in increasing order and 
display the top 5 movies out of corresponding similarity scores.'''

def recommend(movie):

    #find the index of the movies
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    #to fetch movies from indeces
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


print(recommend('Avatar'))

# convert any kind of python object (list, dict, etc.) into byte streams (0s and 1s) is called pickling or serialization or flattening or marshaling
pickle.dump(new_df.to_dict(),open('movies.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))
