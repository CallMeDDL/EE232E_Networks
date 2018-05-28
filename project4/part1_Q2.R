library("igraph")

# Q2: Create a weighted directed actor/actress network
#g = read.graph("graph_edge_list.txt", format = "ncol", directed=TRUE)

g = read.graph("par1_sampled_data_v1/graph_edge_list.txt", format = "ncol", directed=TRUE)

#png(filename="Q2_in_degree_distri.png")
#plot(degree.distribution(g, mode="in"))

# Q4: 
#find top 10 actor id
page_rank <- page_rank(g,damping=0.85, directed = TRUE)$vector
top_actor <- head(sort(page_rank, decreasing=TRUE), 10)

f_actor_idname_map = read.delim("par1_sampled_data_v1/actor_idname_map.txt",header = FALSE, sep="\t")
f_movie_idname_map = read.delim("par1_sampled_data_v1/movie_idname_map.txt",header = FALSE, sep="\t")

# find name
actor_name <- f_actor_idname_map$V3[as.numeric(names(top_actor))+1]
print(actor_name)

# degree and movies of those actors
deg <- degree(g, mode="in")
for (actor_id in as.numeric(names(top_actor)))
{
    print(deg[toString(actor_id)])
    # id to id map 
}


