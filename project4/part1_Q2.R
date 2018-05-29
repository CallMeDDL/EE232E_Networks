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

actor_idname_map = read.delim("par1_sampled_data_v1/actor_idname_map.txt",header = FALSE, sep="\t")
actorid_movieids_map = read.delim("par1_sampled_data_v1/actorid_movieids_map.txt",header = FALSE, sep="\t")

map_file <- file("par1_sampled_data_v1/actorid_movieids_map.txt", "r")
actor_map <- list()
i <- 1
while ( TRUE ) {
    line = readLines(map_file, n = 1)
    if ( length(line) == 0 ) {break}
    vector <- c()
    for (substr in strsplit(line, "\t\t")){
        vector <- c(vector, as.numeric(substr))
    }
    actor_map[[i]] <- vector
    i = i + 1
}
close(con)

# find name
actor_name <- actor_idname_map$V3[as.numeric(names(top_actor))+1]
actor_name <- actor_idname_map$V3[as.numeric(names(top_actor))+1]
print(actor_name)

# degree and movies of those actors
deg <- degree(g, mode="in")
for (actor_id in as.numeric(names(top_actor)))
{
    print(deg[toString(actor_id)])
    print(length(actor_map[[as.numeric(actor_id) + 1]]) - 1)
}

# Q5
actorid_list <- c(14503, 111298, 12812, 27258, 32389, 16878, 62774, 107832, 17285, 53248)
names(actorid_list) <- c("Cruise, Tom", "Watson, Emma (II)", "Clooney, George", "Hanks, Tom", "Johnson, Dwayne (I)", "Depp, Johnny", "Smith, Will (I)", "Streep, Meryl", "DiCaprio, Leonardo", "Pitt, Brad") 
for (actor_id in as.numeric(actorid_list))
{
    print(deg[toString(actor_id)])
    print(length(actor_map[[as.numeric(actor_id) + 1]]) - 1)
}

