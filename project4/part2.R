library("igraph")

# Q1: Create a weighted undirected movie network

g = read.graph("movie_graph_edge_list.txt", format = "ncol", directed=TRUE)

#png(filename="2_1.png")
plot(degree.distribution(g, mode="in"))



