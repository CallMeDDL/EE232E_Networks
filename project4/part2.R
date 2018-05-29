library("igraph")

# Q1: Create a weighted undirected movie network
g = read.graph("par1_sampled_data_v1/movie_graph_edge_list.txt", format = "ncol", directed=FALSE)

png(filename="2_1.png")
plot(degree.distribution(g))



