build_travel_graph <- function(csv_f_name, json_f_name) {

    library(igraph)
    library(data.table)
    library(rjson)
    
    edge_data <- read.csv(file=csv_f_name, header=TRUE, sep=",")
    edge_data <- as.data.table(edge_data)

    # clean1: only keep December
    edge_data <- edge_data[month == 12] 

    # Select necessary collums
    edge_data <- edge_data[, c("sourceid", "dstid", "mean_travel_time")]
    setnames(edge_data,"mean_travel_time", "weight")

    # generate graph from edge list, with edge weight
    g <- graph.data.frame(d = edge_data, directed = FALSE)


    # add additional node attr

    json_data <- fromJSON(file=json_f_name)

    # get id_list, addr_list, loc_x_list, loc_y_list
    id_list <- c(1:length(json_data$features))
    id_list <- as.character(id_list)
    addr_list <- c()
    loc_x_list <- c()
    loc_y_list <- c()
    for (i in 1:length(json_data$features)){
        token <- json_data$features[[i]]
        addr = token$properties$DISPLAY_NAME
        x = mean(split(unlist(token$geometry$coordinates),1:2)[[1]])
        y = mean(split(unlist(token$geometry$coordinates),1:2)[[2]])
        addr_list <- c(addr_list, addr)
        loc_x_list <- c(loc_x_list, x)
        loc_y_list <- c(loc_y_list, y)
    }


    # assign node attr street_addr from map: id -> street_addr
    id_street_addr_map = as.list(setNames(addr_list, id_list))
    V(g)$street_addr <- id_street_addr_map[V(g)$name]

    # assign node attr loc_x from map: id -> loc_x
    id_loc_x_map = as.list(setNames(loc_x_list, id_list))
    V(g)$loc_x <- id_loc_x_map[V(g)$name]

    # assign node attr loc_y from map: id -> loc_y
    id_loc_y_map = as.list(setNames(loc_y_list, id_list))
    V(g)$loc_y <- id_loc_y_map[V(g)$name]
    
    # remove duplicates
    g <- simplify(g, remove.multiple = TRUE, edge.attr.comb="mean")
    
    # only keep the giant connected component
    clusters <- components(g)
    gcc_g <- induced.subgraph(g, which(clusters$membership == which.max(clusters$csize)))
    
    return(gcc_g)
}



# read file
# travel_f_name = './dataset/san_francisco-censustracts-2017-4-All-MonthlyAggregate.csv'
csv_f_name = './dataset/san_francisco-censustracts-2017-4-All-MonthlyAggregate.csv'
json_f_name = './dataset/san_francisco_censustracts.json'


g = build_travel_graph(csv_f_name, json_f_name)
# g has 4 node attributes, and 1 edge attribute
# 4 node attributes:
# V(g)$name: id  (str)
# V(g)$street_addr: Street Address  (str) 
# V(g)$loc_x: x of its mean location  (float)
# V(g)$loc_y: y of its mean location  (float)

# 1 edge attribute:
# E(g)$weight: weight of edge, "mean traveling times"   (float)


# Q6
print(sprintf("Number of nodes in G: %d", vcount(g)))
print(sprintf("Number of edges in G: %d", ecount(g)))


# Q7
g_mst <- mst(g, weights = E(g)$weight)
sample_num = 5
sample_pairs <- list()
i <- 1
for (i in 1: sample_num){
    edge <- E(g_mst)[i]
    id_pair <- ends(g_mst, edge)
    sample_pairs[[i]] <- c(V(g_mst)[id_pair[1]]$street_addr, V(g_mst)[id_pair[2]]$street_addr, edge$weight)
}
print('sample_pairs: ')
print(sample_pairs)




