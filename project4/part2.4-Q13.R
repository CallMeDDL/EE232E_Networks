filepath = "part1_data_original/movie_graph_edge_list.txt"
movie_network<-read.graph(filepath, format = "ncol",directed = FALSE)

fpath1 <- "part1_data_original/actorid_movieid_map.txt"
ori_actor_movie <- read.table(fpath1, header = FALSE, sep = '\t', col.names = c('actor_id','movie_id'))
valid_actor_movie <- subset(ori_actor_movie, movie_id %in% names(V(movie_network)))

fpath2 <- "part1_data_original/movie_idrating_map.txt"
ori_movie_rating <- read.table(fpath2, header = FALSE, sep = '\t', col.names = c('movie_id','rating'))

valid_movie_rating <- subset(ori_movie_rating, movie_id %in% names(V(movie_network)))
valid_movie_rating <- subset(valid_movie_rating, rating != "NaN") # remove "NaN"
#object movieID: 12596,48391,100856
actor_movie_bipartite_graph <- graph_from_data_frame(valid_actor_movie, directed = FALSE, vertices = NULL)

unique_actors <- unique(valid_actor_movie$actor_id)

ratings <- vector()
avg_ratings <- vector()

actor_rating <- data.frame(actor_id = integer(0), avg_rating = double(0))

# write by row

for (act_id in unique_actors){
  mvs <- valid_actor_movie$movie_id[valid_actor_movie$actor_id == act_id]
  ratings <- valid_movie_rating$rating[valid_movie_rating$movie_id %in% mvs]
  avg_rating <- mean(ratings)
  #avg_ratings <- append(avg_ratings, avg_rating)
  row<- c(act_id, avg_rating)
  actor_rating <- rbind(actor_rating, row)
}
#actor_rating <- data.frame(actor_id = unique_actors, avg_rating = avg_ratings)
colnames(actor_rating) <- c('actor_id','score')
#remove NaN -- some of the actors' score is NaN, since all the movies they involved in are non-rated
actor_rating <- subset(actor_rating, actor_rating$score != 'NaN')
# write to file
f_output <- "part1_data_original/actor_score.csv"
write.csv(actor_rating, f_output)

prediction <- function(mv_id){
  inv_actors <- valid_actor_movie$actor_id[valid_actor_movie$movie_id == mv_id]
  #print(inv_actors)
 # print(actor_rating$score[actor_rating$actor_id %in% inv_actors])
  pre_ratings <- mean(actor_rating$score[actor_rating$actor_id %in% inv_actors])
  #print(pred_ratings)
  return(pre_ratings)
}


gt_ratings <- valid_movie_rating$rating
pred_ratings <- vector()
for(mv_id in valid_movie_rating$movie_id){
  pred_ratings <- append(pred_ratings, prediction(mv_id))
}
cat("RMSE:", sqrt(mean((gt_ratings - pred_ratings)^2)))

# predict for three movies
obj_movies <- c(12596, 48391, 100856)
obj_mv_names <- c("Batman v Superman: Dawn of Justice (2016)", "Mission: Impossible - Rogue Nation (2015)","Minions (2015)")
for (index in 1:3) {
  cat(obj_mv_names[index], '\n')
  cat("Predict Rating is:", prediction(obj_movies[index]), '\n', '\n')
}





