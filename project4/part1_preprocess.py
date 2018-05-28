from io import open


# def parse_line: parse each line, with preprocessing and data clean:
# ret: actor_movies: [string, list of string];
# (1) removing the actor/actress who has acted in less than 10 movies
# (2) cleaning the merged text file (avoid inconsistency)
def parse_line(line):
    if line is None or line == "":
        return None

    ret_actor_movies = []
    tokens = line[: -1].split("\t\t")
    movie_num = len(tokens) - 1
    
    # (1) removing the actor/actress who has acted in less than 10 movies
    if movie_num < 10:
        return None

    # actor
    actor_name = tokens[0].strip(" ").strip("\t")
    ret_actor_movies.append(actor_name)

    # movies
    movies_name_list = tokens[1: ]
    cleaned_movies_name_list = []
    for movie_name in movies_name_list:
        # (2) cleaning the merged text file (avoid inconsistency)
        pos = movie_name.find(")")
        movie_name = movie_name[: pos + 1].strip(" ").strip("\t")
        cleaned_movies_name_list.append(movie_name)
    ret_actor_movies.append(cleaned_movies_name_list)

    return ret_actor_movies



def intersection(list1, list2):
    intersection = set(list1).intersection(set(list2))
    return intersection




# main

# input file
f1name = 'actor_movies.txt'
f2name = 'actress_movies.txt'
fsname = [f1name, f2name]

# output file
actorid_movieids_map_fname = "actorid_movieids_map.txt"
edges_fname = "graph_edge_list.txt"
actor_idname_map_fname = "actor_idname_map.txt"
movie_idname_map_fname = "movie_idname_map.txt"

f_actorid_movieids_map = open(actorid_movieids_map_fname, 'w', encoding='latin-1')
f_edge = open(edges_fname, 'w', encoding='latin-1')
f_actor_idname_map = open(actor_idname_map_fname, 'w', encoding='latin-1')
f_movie_idname_map = open(movie_idname_map_fname, 'w', encoding='latin-1')

# preprocessing
actor_idname_map = [] # map actorid -> actor's name (to avoid actors/actresses having the same name)

movie_idname_map = [] # map movieid -> movie's name (to speed up)
movie_nameid_map = {} # map movie's name -> movieid

actorid_movieids_map = []  # map actorid to his/her movies' ids
unique_movies = set()

actor_id = 0
movie_id = 0
for fname in fsname:
    f = open(fname, 'r', encoding='latin-1')
    for line in f.readlines():
        actor_movies = parse_line(line)
        if actor_movies is None: # filter out unsatisfied actors/actresses
            continue

        # assigned each actor/actress with a unique id
        actor_idname_map.append(actor_movies[0])
        f_actor_idname_map.write(str(actor_id) + u"\t\t" + actor_movies[0] + u"\n")
        f_actorid_movieids_map.write(str(actor_id) + u"\t\t")
        actor_id += 1

        # assigned each movie with a unique id; 
        # update actorid_movieids_map
        movie_ids = []
        for movie in actor_movies[1]:
            if movie not in unique_movies: 
                movie_nameid_map[movie] = movie_id
                f_movie_idname_map.write(str(movie_id) + u"\t\t" + movie + u"\n")
                unique_movies.add(movie)
                movie_id += 1
            movie_ids.append(movie_nameid_map[movie])
            f_actorid_movieids_map.write(str(movie_nameid_map[movie]) + u"\t\t")
        actorid_movieids_map.append(movie_ids)        
        f_actorid_movieids_map.write(u"\n")
            

f_actor_idname_map.close()
f_movie_idname_map.close()
f_actorid_movieids_map.close()

# ---------------------Q1-----------------------
print("total number of actors and actresses after data clean: ", len(actor_idname_map))
print("total number of unique movies: ", len(unique_movies))
# ----------------------------------------------


# --------Q2-save-graph-edge-to-file------------
# get edge data and save to file for further graph creation
edge_weight_dict = {} # {actorX: [(actorY, weightXY), (actorZ, weightXZ)]}
for actorid1 in range(0, len(actorid_movieids_map) - 1):
    print('actorid1: ', actorid1)
    for actorid2 in range(actorid1 + 1, len(actorid_movieids_map)):
        # calc: actorid1 -> actorid2: weight

        movieids1 = actorid_movieids_map[actorid1]
        movieids2 = actorid_movieids_map[actorid2]
        common_movies = intersection(movieids1, movieids2)
        if len(common_movies) == 0:
            continue
        weight_1to2 = len(common_movies) * 1.0 / len(actorid_movieids_map[actorid1])
        weight_2to1 = len(common_movies) * 1.0 / len(actorid_movieids_map[actorid2])
        f_edge.write(str(actorid1) + u"\t\t" + str(actorid2) + u"\t\t" + str(weight_1to2) + u"\n")
        f_edge.write(str(actorid2) + u"\t\t" + str(actorid1) + u"\t\t" + str(weight_2to1) + u"\n")

        if actorid1 not in edge_weight_dict:
            edge_weight_dict[actorid1] = []
        if actorid2 not in edge_weight_dict:
            edge_weight_dict[actorid2] = []
        edge_weight_dict[actorid1].append([actorid2, weight_1to2])
        edge_weight_dict[actorid2].append([actorid1, weight_2to1])

f_edge.close()
# ----------------------------------------------


# ---------------------Q3-----------------------
actorname_list = ["Cruise, Tom", "Watson, Emma (II)", "Clooney, George", "Hanks, Tom", "Johnson, Dwayne (I)", "Depp, Johnny", "Smith, Will (I)", "Streep, Meryl", "DiCaprio, Leonardo", "Pitt, Brad"]
actorid_list = [14503, 111298, 12812, 27258, 32389, 16878, 62774, 107832, 17285, 53248]

pair_result = {} # [actor1: [actorX, weight1X], actor2: [actorY, weight2Y] ...]

for actorid1 in actorid_list:
    max_weight = 0

    condidate = [-1, 0]
    for actorid2, weight in edge_weight_dict[actorid1]:
        if weight > max_weight:
            max_weight = weight
            condidate = [actorid2, weight]
    pair_result[actorid1] = condidate

print('pair_result: ', pair_result)
