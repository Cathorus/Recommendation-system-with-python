import pandas as pd 
import os
print(os.listdir("../input"))
# your data must have " ['movie.csv', 'tag.csv', 'genome_tags.csv', 'genome_scores.csv', 'link.csv', 'rating.csv']"

# import movie data and look at columns
movie = pd.read_csv("../input/movie.csv")
movie.columns

#we need movie id and title
movie = movie.loc[:,["movieId","title"]]
movie.head(10)

# import rating data and look at columns
rating = pd.read_csv("../input/rating.csv")
rating.columns

# we need user id, movie id and rating
rating = rating.loc[:,["userId","movieId","rating"]]
rating.head(10)

#merge 2 datas
data = pd.merge(movie,rating)

#we can take only first 1m for easier run
data = data.iloc[:1000000,:]

#make a pivot table in order to make rows are users and columns are movies. And values are rating
pivot_table = data.pivot_table(index = ["userId"],columns = ["title"],values = "rating")
pivot_table.head(10)

#select movie that you want to analyse and compare it with other movies
movie_watched = pivot_table["Movie name"]
similarity = pivot_table.corrwith(movie_watched)  # find correlation between "Bad Boys (1995)" and other movies
similarity = similarity.sort_values(ascending=False)
similarity.head()
