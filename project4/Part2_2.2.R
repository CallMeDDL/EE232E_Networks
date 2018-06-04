library("igraph")
library("ggplot2")
library("dplyr")

filepath = "par1_sampled_data_v1/movie_graph_edge_list.txt"
movie_network<-read.graph(filepath, format = "ncol",directed = FALSE)

# load genre file
filepath2 = "par1_sampled_data_v1/movie_idgenre_map.txt"
df_movie_id_genre <- read.table(filepath2, header = FALSE, sep = '\t', col.names=c('movie_id','movie_genre'))


# get communities
comms <- fastgreedy.community(movie_network) # get communities


# # write community information to csv file (list -> csv)
# outfile <- "/Users/wendycui/Documents/Bruin Life/EE232/project4/communities.csv" ##file path
# # express each community as a dataframe and output each list to a csv file
# for(i in 1:length(sizes(comms))){
#   movies<-V(movie_network)[comms$membership == i]
#   df_temp <- data.frame(i,names(movies))  
#   names(df_temp) <- c("community_id","movie_id")
#   #print(df_temp)
#   if(i == 1){
#     write.table(df_temp, outfile, sep = ",", col.names = T, append = T, row.names = F)
#   }
#   write.table(df_temp, outfile, sep = ",", col.names = F, append = T, row.names = F)
# }



## Q7 ## plot genre distribution in 10 communities

# plot function for one communiy
plot_distribution <- function(df_movie_id_genre, movies,i){
  sel_df <- subset(df_movie_id_genre, movie_id %in% names(movies))
  #print(sel_df)
  p = ggplot(sel_df, aes(x = movie_genre, fill = movie_genre)) + 
    geom_bar() +
    geom_text(stat='count', aes(label = ..count.., color = movie_genre), vjust = -0.5, hjust = 0.5, size = 3) +
    labs(x = "Genre", y = "The Number of Movies", title = paste("Genre Distribution of Communinty",as.character(i)),show_guide = FALSE) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.text.x = element_text(size = 10, vjust = 0.5, hjust = 0.5, angle = 45))
  return(p)
}


# plot first 10 communities
for(i in 1:10) 
{ 
  #movies<-(1:vcount(movie_network))[comms$membership == i] # index of vertices in one community
  movies<-V(movie_network)[comms$membership == i]
  p = plot_distribution(df_movie_id_genre, movies,i)
  figpath = paste("/Users/wendycui/Documents/Bruin Life/EE232/project4/plots/","community_", as.character(i),".png")
  ggsave(figpath, p, device = "png", width = 10, height = 7, units = "in")
  print("save success!")
}

# Q8(a)

get_dominant_genre <- function(movies){   # input movie ids in one community and return the donimant genre.
  genres <- vector()
  for(i in 1:length(movies$name)){
    id = movies$name[i]
    #genre = V(movie_network)[id]$genre
    genre = df_movie_id_genre$movie_genre[df_movie_id_genre$movie_id == id]
    ## remove "NAN" here
    if(genre != "NAN"){
      genres <- append(genres, as.character(genre))
      # here must turn "genre" to character, otherwise it return the order number of the corresponding genre
    }
  }

  dom_genre <- names(table(genres)[table(genres) == max(table(genres))])

  return(dom_genre) # maybe more than one (here we did not remove the "NAN")
}

# get all the dominate genres for all communities
dominate_genres <- vector() # use a vector to store the dominate genre in each community(make sure to recomve the "NAN"s)
for(i in 1:length(sizes(comms))){
  # get the dominant genre in each community
  movies<-V(movie_network)[comms$membership == i]
  dom_genre <- get_dominant_genre(movie_network, movies)
  #print(paste("the dominate genre(s) of",i,"-th community:", dom_genre)) # print the genre list of each community
  print(paste("the dominate genre(s) of",i,"-th community:"))
  print(dom_genre)
  dominate_genres <- append(dominate_genres, dom_genre)
}
print("all dominate genres of among all communities:")
print(dominate_genres)
# the most frequent dominant genres(maybe multiple) amoung all communities
result <- names(table(dominate_genres)[table(dominate_genres) == max(table(dominate_genres))])

print(paste("the most dominate genre(s) among all communities:", result))


## 8(b) ##
# get genre-movie statistic value (q(i) in the Q8), using df_movie_id_genre to generate a dataframe(genre, fraction)
movie_genre_stat <- summarise(group_by(df_movie_id_genre, movie_genre), count = n()) # including the NAN

# To compute the score of each genre (cur_genre is the current genre that needs to compute score in a community)
get_genre_score <- function(cur_genre, movie_genre_stat, comm_genre_stat, movie_num){
  c_i = comm_genre_stat$count[comm_genre_stat$genre == cur_genre]
  p_i = c_i / movie_num
  q_i = movie_genre_stat$count[movie_genre_stat$movie_genre == cur_genre] / 468150 # total number of movies is 468150
  score <- log(c_i) *  p_i / q_i
  return(score)
}

for(i in 1:length(sizes(comms))){
  movies<-V(movie_network)[comms$membership == i] # movies in community i-th
  genres <- vector()
  movie_ids <- vector()
  for(j in 1:length(movies$name)){
    id = movies$name[j]
    #genre = V(movie_network)[id]$genre
    genre = df_movie_id_genre$movie_genre[df_movie_id_genre$movie_id == id]
    # remove "NAN" here
    if(genre != "NAN"){
      genres <- append(genres, as.character(genre))
      movie_ids <- append(movie_ids, id) # only add avaliable genre and movie id 
    }
  }

  df_comm_genre <- data.frame(movie_id = movie_ids, genre = genres) # movie-genre table within one community
  comm_genre_stat <- summarise(group_by(df_comm_genre, genre), count = n())
  movie_num <- length(movies)
  scores <- vector()
  genres <- unique(genres) # remove duplicates

  for(k in 1:length(genres)){ # for each genre in genres
    # call function get_genre_score(genre, movie_genre_stat, comm_genre_stat, movie_num) to get the score
    score <- get_genre_score(genres[k], movie_genre_stat, comm_genre_stat, movie_num)
    scores <- append(scores, score)
  }
  rank_genre <- order(scores, decreasing = TRUE)
  sort_score <- sort(scores, decreasing = TRUE)
  dom_genres <- vector()
  for(s in 1:length(sort_score)){
    dom_genres <- append(dom_genres, genres[rank_genre[s]])
    if(s + 1 < length(sort_score) && sort_score[s + 1] != sort_score[s]){
      break
    }
  }
  print(paste("the dominate genre(s) of",i,"-th community:"))
  print(dom_genres)
}

## 8(c) ##

# get the object community
for(i in 1:length(sizes(comms))){
  if(lengths(comms[i]) >= 10 && lengths(comms[i]) <= 20){
    obj_comm <- comms[i]
    sel_comm_id <- i
    break
  }
}

obj_comm <- unlist(obj_comm, use.names = FALSE) # turn list to vector

# load actor_id -- movie_id file  -> actorid_movieid_map.txt
filepath3 <- "par1_sampled_data_v1/actorid_movieid_map.txt"
df_actor_movie <- read.table(filepath3, header = FALSE, sep = '\t', col.names = c('actor_id','movie_id'))

obj_actors <- df_actor_movie$actor_id[df_actor_movie$movie_id %in% obj_comm]

obj_actors <- unique(obj_actors)  # remove duplicates

df_obj_actor_movie <- subset(df_actor_movie, movie_id %in% obj_comm)

# start plotting bipartite graph
actor_movie_graph <- graph_from_data_frame(df_obj_actor_movie, directed = FALSE, vertices = NULL)

V(actor_movie_graph)[V(actor_movie_graph)$name %in%unique(df_obj_actor_movie$actor_id)]$type = TRUE
V(actor_movie_graph)[V(actor_movie_graph)$name %in%unique(df_obj_actor_movie$movie_id)]$type = FALSE
V(actor_movie_graph)[V(actor_movie_graph)$name %in%unique(df_obj_actor_movie$actor_id)]$color = "red"
V(actor_movie_graph)[V(actor_movie_graph)$name %in%unique(df_obj_actor_movie$movie_id)]$color = "skyblue"

#manually set coords to avoid overlap
coords <- layout_as_bipartite(actor_movie_graph)
seq1 <- seq(0, 20000, by = 1000) # 21 movies
seq2 <- c(300,1400,2500,3600,4600,5700,6800,7900,9000,10100,11200,12300)
coords[,1] <- c(seq1, seq2)


actor_movie_graph %>%
  add_layout_(as_bipartite()) %>%
  plot(layout = coords,vertex.size = 10, vertex.label.cex = 0.5)



comm_actor_stat <- summarise(group_by(df_obj_actor_movie, actor_id), count = n()) 

sort_comm_actor_stat <- comm_actor_stat[with(comm_actor_stat, order(-count)),] # sort df by count decreasing
top3_actor <- sort_comm_actor_stat$actor_id[1:3]

# observe the relation between top3_actor and genre of this community
obj_movies <- vector()
top_genres <- vector()
for(w in 1:length(top3_actor)){
  movies_w <- df_obj_actor_movie$movie_id[df_obj_actor_movie$actor_id == top3_actor[w]]
  genres_w <- df_movie_id_genre$movie_genre[df_movie_id_genre$movie_id %in% movies_w]
  genres_w <- as.vector(genres_w)
  top_genres <- append(top_genres,genres_w)
}
top_genre <- names(table(top_genres)[table(top_genres) == max(table(top_genres))])

