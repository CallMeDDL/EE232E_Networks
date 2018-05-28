library("igraph")

# Q2: Create a weighted directed actor/actress network
g = read.graph("graph_edge_list.txt", format = "ncol", directed=TRUE)

png(filename="Q2_in_degree_distri.png")
plot(degree.distribution(g, mode="in"))
