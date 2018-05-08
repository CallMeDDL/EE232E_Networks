#project 2- 1.4

library(igraph)
# load whole graph
file_path = '/Users/wendycui/Documents/Bruin Life/EE232/project2/facebook_combined.txt'

fb_graph <- read.graph(file_path,format=c("edgelist"),directed=FALSE)

# get ID415 subgraph
v415_neighbors <- neighbors(fb_graph, 415) # ID - 1
fb_415 <- induced_subgraph(fb_graph, c(415, v415_neighbors))
V(fb_415)$degree <- degree(fb_415)
V(fb_415)$vid <- seq(vcount(fb_415),1)

# return the vertices' vid with degree == 24, they are "test data" in the following question
Nr = V(fb_415)[V(fb_415)$degree == 24] # vid is different from the vertices id in edgelist


# Q16: |Nr|: the number of vertices with degree == 24
num_Nr = length(Nr)

# observ = max(degree(fb_415))
# observ2 = degree(fb_graph, 415)
# vertices(fb_415)


obj_node = which(degree(fb_415) == 24) # list of nodes not graph type

# randomly picked deleted list

# function to delete friends of user i and recommend friends for him.
# the arg ori_friends should be his neighbor list(his friends)
get_gt_del_friend <- function(ori_friends){ 
  #user is the focusing node, ori_friends are his original friends
  R <- vector() # deleted friends list
  for (i in 1:length(ori_friends)){
    flag <- rbinom(1,1,0.25)
    if(flag){ # delete successfully
      #R <- c(R,ori_friends[i]$vid)
      R <- c(R,ori_friends[i])
    }
  }
  return(R) # deleted vertices
}

# first we need to get the original friend list of users in Nr
# note the i-th user's current friend list should be 'setdiff(ori_friends,get_gt_friend(ori_friends))'
# We should also find the non-neighbor list of the user

get_recommend <- function(curr_friends, non_neighbors, cur_g, rnum, method){
  # we have 3 mehtods to recommend: "CN","J","AA"
  Pi <- vector() # recommendation results
  candidate_list <- vector() # store the scores
  if(method == "CN"){
    for (user in non_neighbors){ # user is vid
      Sj <- neighbors(cur_g, user)
      #print(paste("Sj:", length(Sj)))
      candidate_list <- c(candidate_list, length(intersect(curr_friends, Sj)))
    }
  }
  else if (method == "J"){
    for (user in non_neighbors){
      Sj <- neighbors(cur_g, user)
      jaccard <- length(intersect(curr_friends, Sj)) / length(union(curr_friends, Sj))
      candidate_list <- c(candidate_list, jaccard)
    }
  }
  else if (method == "AA"){
    for (user in non_neighbors){
      Sj <- neighbors(cur_g, user)
      common_friends <- intersect(curr_friends, Sj)
      AdamicAdar <- 0
      for (k in common_friends){
        Sk <- neighbors(cur_g, k)
        AdamicAdar <- AdamicAdar + 1 / log(length(Sk))
      }
      candidate_list <- c(candidate_list, AdamicAdar)
    }
  }
  rec_index <- sort(candidate_list, decreasing = TRUE, index.return = TRUE)$ix[1:rnum]
  Pi <- non_neighbors[rec_index]
  return(Pi)
}

get_accuracy <- function(Pi, Ri){ # Pi is the recommendation result and Ri is the ground truth
  return(length(intersect(Pi,Ri)) / length(Ri))
}
main_func <- function(ori_g, obj_user){
  CN_acc <- vector()
  J_acc <- vector()
  AA_acc <- vector()
  for (i in 1:length(obj_user)){ 
    # for each objective user, we run each recommendation method 10 times and got three average acurracy.
    iter <- 1
    # three elements in this list
    acc1 <- 0
    acc2 <- 0
    acc3 <- 0
    
    user = obj_user[i] 
    ori_friends <- neighbors(ori_g, user) 
    
    while(iter <= 10){

      gt_del_friends <- get_gt_del_friend(ori_friends) 
      
      current_graph <- ori_g
      for (delfriend in gt_del_friends){
        #edge = paste(user,"|", delfirend)
        #current_graph = delete_edges(current_graph,edge)
        
        deledge = sprintf("%s|%s",user,delfriend)
        current_graph = current_graph - edge(deledge)
      }
      for (delfriend in gt_del_friends){
        #edge = paste(user,"|", delfirend)
        #current_graph = delete_edges(current_graph,edge)
        
        observ_neigbors <- neighbors(current_graph, delfriend)
        print(paste("current del's neibor:", length(observ_neigbors)))
      }
     

      current_friends <- setdiff(ori_friends, gt_del_friends) 

      non_neighbors <- setdiff(V(current_graph), c(user,current_friends))
      #print(paste("length of non-neighbors:", non_neighbors))
      rnum <- length(gt_del_friends)
      res1 <- get_recommend(current_friends, non_neighbors, current_graph, rnum, "CN")
      res2 <- get_recommend(current_friends, non_neighbors, current_graph, rnum, "J")
      res3 <- get_recommend(current_friends, non_neighbors, current_graph, rnum, "AA")

      acc1 <- acc1 + get_accuracy(res1, gt_del_friends)
      acc2 <- acc2 + get_accuracy(res2, gt_del_friends)
      acc3 <- acc3 + get_accuracy(res3, gt_del_friends)
      iter <- iter + 1
    }
    CN_acc <- c(CN_acc, acc1 / 10) 
    J_acc <- c(J_acc, acc2 / 10) 
    AA_acc <- c(AA_acc, acc3 / 10) 
    print(paste("user",i,"completed!"))
  }
  return (c(mean(CN_acc), mean(J_acc), mean(AA_acc)))
}

result <- main_func(fb_415, obj_node)





