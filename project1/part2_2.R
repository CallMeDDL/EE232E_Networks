library('igraph')
library('Matrix')
library('pracma')

create_transition_matrix = function (g) {
    
    # WARNING: make sure your graph is connected (you might input GCC of your graph)
    
    vs = V(g)
    n = vcount(g)
    adj = as_adjacency_matrix(g)
    adj[diag(rowSums(adj) == 0)] = 1  # handle if the user is using the function for networks with isolated nodes by creating self-edges
    z = matrix(rowSums(adj, , 1))
    
    transition_matrix = adj / repmat(z, 1, n)  # normalize to get probabilities
    
    return(transition_matrix)
}

random_walk = function (g, num_steps, start_node, transition_matrix = NULL){
    if(is.null(transition_matrix)) {
        transition_matrix = create_transition_matrix(g)
    }
    
    v = start_node
    for(i in 1:num_steps){
        # fprintf('Step %d: %d\n', i, v)  # COMMENT THIS
        PMF = transition_matrix[v, ]
        v = sample(1:vcount(g), 1, prob = PMF)        
    }

    # fprintf('Step %d: %d\n', i + 1, v)  # COMMENT THIS
    return(v)
}

# 2
# 2.2(a)
# Create an undirected random network g1 with 1000 nodes, and the probability p for drawing an edge between any pair of nodes equal to 0.01.
p = 0.01
n = 1000
g <- sample_pa(n, m = 1, directed = FALSE)
t_matrix = create_transition_matrix(g)
png(filename="./2_2_a_network_1000.png")
plot(g, main="2_2_a_network_1000", vertex.size=2, vertex.label=NA, edge.arrow.size=1)
print(diameter(g))


# 2.2(b)
# Random walk for g1. Measure standard deviation and average of distance.
step_num = 100
random_num = 100
avg <- rep(NA, step_num)
sdv <- rep(NA, step_num)
for (step in (1: step_num)) {
    shortest_path <- rep(NA, random_num)
    start_ids = sample.int(n, random_num)
    for (i in (1: random_num)) {
        start_id = start_ids[i]
        node_id = random_walk(g, step, start_id, t_matrix)
        tmp <- get.shortest.paths(g, from = start_id, to = node_id)
        shortest_path[i] <- length(tmp$vpath[[1]]) - 1
    }
    avg[step] <- mean(shortest_path)
    sdv[step] <- sd(shortest_path)
}

png(filename="./2_2_b_distance.png")
plot(1: step_num, avg, type='l', main="average distance for 1000 nodes", xlab = "step", ylab = "average dist")

png(filename="./2_2_b_deviation.png")
plot(1: step_num, sdv, type='l', main="standard deviation of distance for 1000 nodes", xlab = "step", ylab = "standard deviation dist")

print("network with 1000 nodes' diameter: %d", diameter(g))
print("finished 2.2(b)")


# 2.2(c)
step_num = 100
random_num = 1000
end_ids = rep(NA, random_num)
for (start_id in (1: random_num)) {
    # print(start_id)
    node_id = random_walk(g, step_num, start_id, t_matrix)        
    end_ids[start_id] <- node_id
}
png(filename="./2_2_c_graph_degree.png")
hist(degree(g), main="degree distribution of graph", xlab = "degree", ylab = "frequency")
png(filename="./2_2_c_end_nodes_degree.png")
hist(degree(g, end_ids), main="degree distribution of the end nodes reached", xlab = "degree", ylab = "frequency")

print("finished 2.2(c)")

# 2.2(d)
# 100
p = 0.01
n = 100
g <- sample_pa(n, m = 1, directed = FALSE)
t_matrix = create_transition_matrix(g)
png(filename="./2_2_d_network_100.png")
plot(g, main="2_2_d_network_100", vertex.size=2, vertex.label=NA, edge.arrow.size=1)
print(diameter(g))

step_num = 100
random_num = 100
avg <- rep(NA, step_num)
sdv <- rep(NA, step_num)
for (step in (1: step_num)) {
    shortest_path <- rep(NA, random_num)
    for (start_id in (1: random_num)) {
        node_id = random_walk(g, step, start_id, t_matrix)
        tmp <- get.shortest.paths(g, from = start_id, to = node_id)
        shortest_path[start_id] <- length(tmp$vpath[[1]]) - 1
    }
    avg[step] <- mean(shortest_path)
    sdv[step] <- sd(shortest_path)
}

png(filename="./2_2_d_100_distance.png")
plot(1:step_num, avg, type='l', main="average distance for 100 nodes", xlab = "step", ylab = "average distance")

png(filename="./2_2_d_100_deviation.png")
plot(1:step_num, sdv, type='l', main="standard deviation of distance for 100 nodes", xlab = "step", ylab = "standard deviation")

print("finished 2.2(d) 100 nodes")

# 10000
p = 0.01
n = 10000
g <- sample_pa(n, m = 1, directed = FALSE)
t_matrix = create_transition_matrix(g)
png(filename="./2_2_d_network_10000.png")
plot(g, main="2_2_d_network_10000", vertex.size=2, vertex.label=NA, edge.arrow.size=1)
print(diameter(g))

step_num = 100
random_num = 100
avg <- rep(NA, step_num)
sdv <- rep(NA, step_num)
for (step in (1: step_num)) {
    shortest_path <- rep(NA, random_num)
    start_ids = sample.int(n, random_num)
    for (i in (1: random_num)) {
        start_id = start_ids[i]
        node_id = random_walk(g, step, start_id, t_matrix)
        tmp <- get.shortest.paths(g, from = start_id, to = node_id)
        shortest_path[i] <- length(tmp$vpath[[1]]) - 1
    }
    avg[step] <- mean(shortest_path)
    sdv[step] <- sd(shortest_path)
}

png(filename="./2_2_d_10000_distance.png")
plot(1:step_num, avg, type='l', main="average distance for 10000 nodes", xlab = "step", ylab = "average distance")

png(filename="./2_2_d_10000_deviation.png")
plot(1:step_num, sdv, type='l', main="standard deviation of distance for 10000 nodes", xlab = "step", ylab = "standard deviation")

print("finished 2.2(d) 10000 nodes")
dev.off()
