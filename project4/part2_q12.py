import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import math
import code

def lr(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
    

def rmse(preds, labels):
    return np.sqrt(np.mean((np.array(preds) - np.array(labels)) ** 2))


def train(actorid_pagerank, movieid_actorids, movieid_rating):# list of list, list

    features = []
    ratings = []
    for movieid in range(len(movieid_rating)):
        # print('movieid: ', movieid)
        # label
        rating = movieid_rating[movieid]
        if rating == -1:
            continue

        # feature
        feature_num = 6
        pageranks = []
        actorids = movieid_actorids[movieid]
        for actorid in actorids:
            if actorid < len(actorid_pagerank):
                pageranks.append(actorid_pagerank[actorid])

        # sort pageranks
        pageranks.sort(reverse=True)

        feature = []
        for pagerank in pageranks[: feature_num]:
            feature.append(pagerank)

        if len(feature) == feature_num:
            features.append(feature)
            ratings.append(rating)
        
    # code.interact(local=locals())
    # data preparing
    X = np.array(features)
    y = ratings

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 29)

    model = lr(X_train, y_train)
        
    y_test_pred = model.predict(X_test)
   

    rmse_result = rmse(y_test_pred, y_test)
    print('rmse: ', rmse_result)

    code.interact(local=locals())

    X_target = genfromtxt('target_feature.txt', delimiter=',')
    y_pred = model.predict(X_target)
    print('y_pred: ', y_pred)


if __name__ == '__main__':

    # get actor_pagerank
    actorid_pagerank = [float(line.strip()) for line in open("actor_pagerank.txt", 'r')]
    movieid_actorids_tmp = genfromtxt('movie_actors.txt', delimiter=',', dtype=int)
    movieid_rating_tmp = genfromtxt('movie_ratings.txt', delimiter=',')

    movie_max_id = int(max(np.max(movieid_rating_tmp[:, 0]), np.max(movieid_actorids_tmp[:, 0])))

    movieid_rating =  -1 * np.ones((movie_max_id + 1))

    for row in range(movieid_rating_tmp.shape[0]):
        movieid = int(movieid_rating_tmp[row, 0])
        rating = 0
        if math.isnan(movieid_rating_tmp[row, 1]):
            rating = -1
        else:
            rating = int(movieid_rating_tmp[row, 1])
        movieid_rating[movieid] = rating


    movieid_actorids =  [[] for i in range(movie_max_id + 1)]
    for row in range(movieid_actorids_tmp.shape[0]):
        movieid = int(movieid_actorids_tmp[row, 0])
        actorid = int(movieid_actorids_tmp[row, 1])
        if movieid_rating[movieid] != -1:
            movieid_actorids[movieid].append(actorid)

    train(actorid_pagerank, movieid_actorids, movieid_rating)


