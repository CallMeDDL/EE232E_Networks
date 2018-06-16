library(tripack)
library(igraph)
library(data.table)
library(rjson)

csv_f_name = 'san_francisco-censustracts-2017-4-All-MonthlyAggregate.csv'
json_f_name = 'san_francisco_censustracts.json'

# generate g from part2
g = build_travel_graph(csv_f_name, json_f_name)

# Q11
triangles <- tri.mesh(V(g)$loc_x, V(g)$loc_y)
plot(triangles, do.points=FALSE, lwd=0.2)

# find road and length of each road
neighbor<-neighbours(triangles)
euc_dist <- as.matrix(dist(cbind(x=triangles$x, y=triangles$y)))

# Q12
id1 <- vector()
id2 <- vector()
weight <- vector()
for (i in 1:length(neighbor)){
    for(j in neighbor[[i]]){
        id1 <- c(id1, i)
        id2 <- c(id2, j)
        t <- distances(g, v = i, to = j, weights = E(g)$weight)
        v <- euc_dist[i,j] * 69 / 60 / t[1,1]
        w <- 60*60 / (2 + 0.003/v)
        weight <- c(weight, w)
    }
}
road = data.frame(id1, id2, weight)
g_r <- graph.data.frame(d = road, directed = FALSE)

# Q13
max_flow(g_r, source = V(g_r)["2607"], target = V(g_r)["1968"], capacity = E(g_r)$weight)
edge_connectivity(g, source = V(g_r)["2607"], target = V(g_r)["1968"])