"""
Generate network from map.txt

Copyright: Tianyi Liu
"""


import numpy as np
import csv

if __name__ == "__main__":


    with open("part1_data_v1/actorid_movieids_map.txt") as f:
        reader = csv.reader(f, delimiter="\t")
        actorid_movieids_map_orig = list(reader)

    # create movie-actor relation matrix
    max_movie_id = 0
    for i in range(0, len(actorid_movieids_map_orig)):
        for j in range(1, len(actorid_movieids_map_orig[i])):
            if (actorid_movieids_map_orig[i][j] != ''):
                max_movie_id = max(max_movie_id, int(actorid_movieids_map_orig[i][j]))

    movie_map = []
    k = 0
    for i in range(0, max_movie_id + 1):
        movie_map.append([])
    for i in range(0, len(actorid_movieids_map_orig)):
        for j in range(1, len(actorid_movieids_map_orig[i])):
            if (actorid_movieids_map_orig[i][j] != ''):
                movie_map[int(actorid_movieids_map_orig[i][j])].append(i)

    # create network
    f_edge = open("movie_graph_edge_list.txt", 'w')

    for movieid1 in range(0, len(movie_map)/50 - 1):
        for movieid2 in range(movieid1 + 1, len(movie_map)/50):
            
            actors1 = movie_map[movieid1]
            actors2 = movie_map[movieid2]
            if(len(actors1) < 5 or len(actors2) < 5):
                continue
            inter = len(set(actors1).intersection(set(actors2)))
            union = len(set(actors1).union(set(actors2)))
            if inter == 0:
                continue
            weight = inter * 1.0 / union
            f_edge.write(str(movieid1) + u"\t" + str(movieid2) + u"\t" + str(weight) + u"\n")

    f_edge.close()