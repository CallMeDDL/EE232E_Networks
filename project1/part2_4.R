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

get_transition_matrix = function (g, pagerank, alpha=0){
    adj = as_adjacency_matrix(g)
  
    adj = matrix(adj)
    dim(adj) <- c(1000, 1000)
    adj2 = adj
    adj2 = matrix(adj2)
    dim(adj2) <- c(1000, 1000)
  
    adj2[adj==0] = repmat(pagerank, 1000, 1)[adj==0]
    adj2[adj==1] = 0

    walk_t_matrix <- create_transition_matrix(g, adj)
    jump_t_matrix <- create_transition_matrix(g, adj2)
    return((1 - alpha) * walk_t_matrix + alpha * jump_t_matrix)
}

get_transition_matrix_two_nodes = function (g, index1, index2, alpha=0){
    adj = as_adjacency_matrix(g)
  
    adj = matrix(adj)
    dim(adj) <- c(1000,1000)
  
    adj2 = matrix(0,1000,1000)
    adj2[, index1] = 0.5
    adj2[, index2] = 0.5
  
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


# 4
n = 1000
random_num = 1000
step_num = 200

g <- sample_pa(n, m = 4, directed = TRUE)

deg = degree(g, mode = "in")

# or use prob result calculated in part2_3 (same)
pagerank <- page_rank(g)$vector


# 4(a)
t_matrix <- get_transition_matrix(g, pagerank, 0.15)

count_ids <- rep(0, n)
start_ids = sample.int(n, random_num)

for (i in (1: random_num)) {
    print(i)    
    start_id = start_ids[i]
    node_id = random_walk(g, step_num, start_id, t_matrix)
    count_ids[node_id] = count_ids[node_id] + 1
}

prob = count_ids / random_num
png(filename="./4_a_rw_prob.png")
plot(1: n, prob, main="the prob nodes visited.", xlab = "node_id", ylab = "probability")

png(filename="./4_a_rw_prob_by_degree.png")
plot(degree(g), prob, main="the prob nodes visited by degree.", xlab = "degree", ylab = "probability")
print(max(deg))


# 4(b)

sorted_page_rank = sort(pagerank, index.return = TRUE)
v1 = sorted_page_rank$x[500]
v2 = sorted_page_rank$x[501]
index1 = 0
index2 = 0
tmp = 0
for (i in 1: 1000) {
    if(tmp != v1 && sorted_page_rank$x[i] == v1) {
        index1 = sorted_page_rank$ix[i]
    }
    if(tmp == v2 && sorted_page_rank$x[i] != v2) {
        index2 = sorted_page_rank$ix[i - 1]
    }
    tmp = sorted_page_rank$x[i]
}

t_matrix = get_transition_matrix_two_nodes(g, index1, index2, 0.15)

count_ids <- rep(0, n)
start_ids = sample.int(n, random_num)

for (i in (1: random_num)) {
    print(i)    
    start_id = start_ids[i]
    node_id = random_walk(g, step_num, start_id, t_matrix)
    count_ids[node_id] = count_ids[node_id] + 1
}

prob = count_ids / random_num
png(filename="./4_b_rw_prob.png")
plot(1:n, prob, main="the prob nodes visited.", xlab = "node_id", ylab = "probability")

png(filename="./4_b_rw_prob_by_degree.png")
plot(degree(g), prob, main="the prob nodes visited by degree.", xlab = "degree", ylab = "probability")
print(max(deg))



