library('igraph')
library('Matrix')
library('pracma')


create_transition_matrix = function (g, adj) {
    # WARNING: make sure your graph is connected (you might input GCC of your graph)
    vs = V(g)
    n = vcount(g)
    adj[diag(rowSums(adj) == 0)] = 1  # handle if the user is using the function for networks with isolated nodes by creating self-edges
    z = matrix(rowSums(adj, , 1))
    
    transition_matrix = adj / repmat(z, 1, n)  # normalize to get probabilities
    
    return(transition_matrix)
}
create_transition_matrix = function (g, adj) {
    # WARNING: make sure your graph is connected (you might input GCC of your graph)
    vs = V(g)
    n = vcount(g)
    adj[diag(rowSums(adj) == 0)] = 1  # handle if the user is using the function for networks with isolated nodes by creating self-edges
    z = matrix(rowSums(adj, , 1))
    
    transition_matrix = adj / repmat(z, 1, n)  # normalize to get probabilities
    
    return(transition_matrix)
}

get_transition_matrix = function (g, alpha=0){
    walk_matrix_adj = as_adjacency_matrix(g)
    jump_matrix_adj = walk_matrix_adj
    jump_matrix_adj[walk_matrix_adj == 0] = 1
    jump_matrix_adj[walk_matrix_adj == 1] = 1
    # print(jump_matrix_adj)

    adj = walk_matrix_adj
    adj2 = jump_matrix_adj

    walk_t_matrix <- create_transition_matrix(g, adj)
    jump_t_matrix <- create_transition_matrix(g, adj2)
    return((1 - alpha) * walk_t_matrix + alpha * jump_t_matrix)
}

random_walk = function (g, num_steps, start_node, t_matrix){
    v = start_node
    for(i in 1: num_steps){
        # fprintf('Step %d: %d\n', i, v)  # COMMENT THIS
        PMF = t_matrix[v, ]
        v = sample(1:vcount(g), 1, prob = PMF)    
    }
    return(v)
}

# 3
n = 1000
g <- sample_pa(n, m = 4,directed = TRUE)
png(filename="./3_network.png")
plot(g, main="Network", vertex.size=2, vertex.label=NA, edge.arrow.size=1 )

step_num = 200
random_num = 1000

start_ids = sample.int(n, random_num)

deg = degree(g)

# 3(a)

count_ids <- rep(0, n)
start_ids = sample.int(n, random_num)

t_matrix = get_transition_matrix(g, 0)

for (i in (1: random_num)) {
    print(i)    
    start_id = start_ids[i]
    node_id = random_walk(g, step_num, start_id, t_matrix)
    count_ids[node_id] = count_ids[node_id] + 1
}

prob = count_ids / random_num

png(filename="./3_a_rw_prob.png")
plot(1:n, prob, main="the prob nodes visited.", xlab = "node_id", ylab = "probability")

png(filename="./3_a_rw_prob_by_degree.png")
plot(degree(g), prob, main="the prob nodes visited by degree.", xlab = "degree", ylab = "probability")
print(max(deg))

# 3(b)
count_ids <- rep(0, n)
start_ids = sample.int(n, random_num)

t_matrix = get_transition_matrix(g, 0.15)

for (i in (1: random_num)) {
    print(i)    
    start_id = start_ids[i]
    node_id = random_walk(g, step_num, start_id, t_matrix)
    count_ids[node_id] = count_ids[node_id] + 1
}

prob = count_ids / random_num

png(filename="./3_b_rw_prob_teleport.png")
plot(1:n, prob, main="the prob nodes visited with teleportation.", xlab = "node_id", ylab = "probability")

png(filename="./3_b_rw_prob_teleport_by_degree.png")
plot(degree(g), prob, main="the prob nodes visited with teleportation by degree.", xlab = "degree", ylab = "probability")

