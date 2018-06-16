library(tripack)
library(igraph)
library(data.table)
library(rjson)

csv_f_name = 'san_francisco-censustracts-2017-4-All-MonthlyAggregate.csv'
json_f_name = 'san_francisco_censustracts.json'

# generate g from part2
g = build_travel_graph(csv_f_name, json_f_name)

# ==================================Q11====================================
triangles <- tri.mesh(V(g)$loc_x, V(g)$loc_y)
plot(triangles, do.points=FALSE, lwd=0.2)

# ==================================Q12====================================

# find road and length of each road
neighbor<-neighbours(triangles)
euc_dist <- as.matrix(dist(cbind(x=triangles$x, y=triangles$y)))

# calculate car/hor
id1 <- c()
id2 <- c()
weight <- c()
time <- c()
for (i in 1:length(neighbor)){
    for(j in neighbor[[i]]){
        id1 <- c(id1, i)
        id2 <- c(id2, j)
        ei <- get.edge.ids(g, c(i, j))
        t <- E(g)[ei]$weight
        if(length(t) == 0){
            t <- distances(g, v = i, to = j, weights = E(g)$weight)
            t <- t[1,1]
        }
        v <- euc_dist[i,j] * 69 / t
        w <- 2 * 60 * 60 / (2 + 0.003/v)
        weight <- c(weight, w)
        time <- c(time, t)
    }
}
road = data.frame(id1, id2, weight)

# ==================================Q13====================================

# calculate max flow
idx1 <- which(V(g)$name == "2607")
idx2 <- which(V(g)$name == "1968")
flow <- max_flow(g_r, source = V(g_r)[idx1], target = V(g_r)[idx2], capacity = E(g_r)$weight)
path <- edge_connectivity(g_r, source = V(g_r)[idx1], target = V(g_r)[idx2])

# ==================================Q14====================================

# plot map with bridges marked as red color
plot(triangles, do.points=FALSE, lwd=0.2)
bridge1 <- c(-122.475, 37.806, -122.479, 37.83)
bridge2 <- c(-122.479, 37.83, -122.387, 37.93)
bridge3 <- c(-122.273, 37.563, -122.122, 37.627)
bridge4 <- c(-122.142, 37.486, -122.067, 37.54)
bridge5 <- c(-122.388, 37.788, -122.302, 37.825)

segments(bridge1[1], bridge1[2], bridge1[3], bridge1[4], col= 'red', lwd = 0.5)
segments(bridge2[1], bridge2[2], bridge2[3], bridge2[4], col= 'red')
segments(bridge3[1], bridge3[2], bridge3[3], bridge3[4], col= 'red')
segments(bridge4[1], bridge4[2], bridge4[3], bridge4[4], col= 'red')
segments(bridge5[1], bridge5[2], bridge5[3], bridge5[4], col= 'red')

remove <- which(time > 1500)
id1_de = id1[-remove]
id2_de = id2[-remove]
weight_de = weight[-remove]

# generate map after defoliation and plot
g_rde <- graph.data.frame(d = data.frame(id1_de, id2_de, weight_de), directed = FALSE)
V(g_rde)$loc_x <- V(g)$loc_x
V(g_rde)$loc_y <- V(g)$loc_y
lo <- layout.norm(as.matrix(data.frame(unlist(V(g)$loc_x, use.names=FALSE), unlist(V(g)$loc_y, use.names=FALSE))))
plot(g_rde, layout = lo ,vertex.label=NA, vertex.size = 1, edge.curved = 0)

# ==================================Q15====================================

# calculate max flow
max_flow(g_rde, source = V(g_rde)[idx1], target = V(g_rde)[idx2], capacity = E(g_rde)$weight_de)
edge_connectivity(g_rde, source = V(g_rde)[idx1], target = V(g_rde)[idx2])


