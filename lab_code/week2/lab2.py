import pods
import zipfile
import sys
import pandas as pd
import numpy as np

pods.util.download_url("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
zip_console = zipfile.ZipFile('ml-latest-small.zip', 'r')
for name in zip_console.namelist():
    zip_console.extract(name, './')
    YourStudentID = 20  # Include here the last three digits of your UCard number
    nUsersInExample = 10  # The maximum number of Users we're going to analyse at one time

    ratings = pd.read_csv("./ml-latest-small/ratings.csv")
    """
    ratings is a DataFrame with four columns: userId, movieId, rating and tags. We
    first want to identify how many unique users there are. We can use the unique 
    method in pandas
    """
    # unique 去重
    indexes_unique_users = ratings['userId'].unique()
    # 用户总数
    n_users = indexes_unique_users.shape[0]
    """ 
    We randomly select 'nUsers' users with their ratings. We first fix the seed
    of the random generator to make sure that we always get the same 'nUsers'
    """
    np.random.seed(YourStudentID)
    # premutation 重新排列
    indexes_users = np.random.permutation(n_users)
    # 随便取出nUsersInExample个
    my_batch_users = indexes_users[0:nUsersInExample]
    """
    We will use now the list of 'my_batch_users' to create a matrix Y. 
    """
    # We need to make a list of the movies that these users have watched
    list_movies_each_user = [[] for _ in range(nUsersInExample)]
    list_ratings_each_user = [[] for _ in range(nUsersInExample)]

    # Movies
    list_movies = ratings['movieId'][ratings['userId'] == my_batch_users[0]].values
    list_movies_each_user[0] = list_movies
    # Ratings
    list_ratings = ratings['rating'][ratings['userId'] == my_batch_users[0]].values
    list_ratings_each_user[0] = list_ratings
    # Users
    n_each_user = list_movies.shape[0]
    list_users = my_batch_users[0] * np.ones((1, n_each_user))

    for i in range(1, nUsersInExample):
        # Movies
        local_list_per_user_movies = ratings['movieId'][ratings['userId'] == my_batch_users[i]].values
        list_movies_each_user[i] = local_list_per_user_movies
        list_movies = np.append(list_movies, local_list_per_user_movies)
        # Ratings
        local_list_per_user_ratings = ratings['rating'][ratings['userId'] == my_batch_users[i]].values
        list_ratings_each_user[i] = local_list_per_user_ratings
        list_ratings = np.append(list_ratings, local_list_per_user_ratings)
        # Users
        n_each_user = local_list_per_user_movies.shape[0]
        local_rep_user = my_batch_users[i] * np.ones((1, n_each_user))
        list_users = np.append(list_users, local_rep_user)

    # Let us first see how many unique movies have been rated # 600
    indexes_unique_movies = np.unique(list_movies)
    n_movies = indexes_unique_movies.shape[0]

    # As it is expected no all users have rated all movies. We will build a matrix Y
    # with NaN inputs and fill according to the data for each user
    temp = np.empty((n_movies, nUsersInExample,))
    temp[:] = np.nan

    Y_with_NaNs = pd.DataFrame(temp)

    for i in range(nUsersInExample):
        local_movies = list_movies_each_user[i]
        # Test whether each element of a 1-D array is also present in a second array.
        # Returns a boolean array the same length as ar1 that is True
        # where an element of ar1 is in ar2 and False otherwise.
        # We recommend using isin instead of in1d for new code.
        ixs = np.in1d(indexes_unique_movies, local_movies)
        # Access a group of rows and columns by label(s) or a boolean array.
        Y_with_NaNs.loc[ixs, i] = list_ratings_each_user[i]

    Y_with_NaNs.index = indexes_unique_movies.tolist()
    Y_with_NaNs.columns = my_batch_users.tolist()
