library('igraph')

# Problem 18 
# create directed personal networks for users who have more than 2 circles

dir_path = "gplus/"
file.names <- dir(dir_path, pattern =".circles")

total_count <- 0
more_than_2circle_count <- 0
for (i in 1:length(file.names)) {
    # not empty file
    if (file.size(paste0(dir_path, file.names[i])) != 0) {
        data <- read.table(paste0(dir_path,file.names[i]), sep = " ")
        # count users who have more than 2 circles
        if (dim(data)[1] > 2) {
            more_than_2circle_count <- more_than_2circle_count + 1
        }
    }
    total_count <- total_count + 1
}
print(sprintf("There are %d personal networks in total.", total_count))
print(sprintf("There are %d personal networks for users who have more than 2 circles.", more_than_2circle_count))




node_id_list <- c("109327480479767108490", 
               "115625564993990145546",
               "101373961279443806744")

for (node_id in node_id_list) {
    
    # build personal net for user_id
    extension <- ".edges"
    edges_file = paste("gplus/", node_id, extension, sep="")
    g_network = read.graph(edges_file, format = "ncol", directed=TRUE)
    g_network = add.vertices(g_network, nv=1, name = node_id)
    edges = c()
    for(node in 1: (vcount(g_network) - 1)) {
        edges = c(edges, c(vcount(g_network), node))
    }
    g_network = add_edges(g_network, edges)
    
    # Problem 19
    # plot degree distribution for in degree 
    png(filename=paste("./", node_id, "_in_degree_distri.png"))
    plot(degree.distribution(g_network, mode="in"), main=paste("Degree Distribution for in-degree for node", node_id), xlab="Degree", ylab="Frequency", cex = 0.7, cex.axis=0.8, cex.main=0.7)

    # plot degree distribution for out degree 
    png(filename=paste("./", node_id, "_out_degree_distri.png"))
    plot(degree.distribution(g_network, mode="out"), main=paste("Degree Distribution for out-degree for node", node_id), xlab="Degree",ylab="Frequency", cex = 0.7, cex.axis=0.8, cex.main=0.7)
    
    
    # Problem 20
    walktrap_community <- walktrap.community(g_network)
    for(i in 1: length(walktrap_community)) {
        print(sprintf("Size of No. %d community is %d", i, sizes(walktrap_community)[i]))
    }
    # plot communities
    png(filename=paste("./", node_id, "walktrap_community.png"))
    plot(walktrap_community, g_network, vertex.label = NA, vertex.size = 3, edge.color = 'black', edge.width=0.05, edge.arrow.size=0.01, main=paste("Walktrap community for node", node_id))
    # modularity score
    print(sprintf("Modularity score for node %s: %f", node_id, modularity(walktrap_community)))

    
    # Problem 22
    extension <- ".circles"
    circles_file = paste("gplus/", node_id, extension, sep="")
    circles_lines = readLines(file(circles_file, open="r"))

    circles = c()
    a = c()
    for(i in 1: length(circles_lines)) {
        tmp = strsplit(circles_lines[i], "\t")
        circles = c(circles, list(tmp[[1]][-1]))
        a = c(a, length(tmp[[1]][-1]))

    }

    people = unique(unlist(circles))
    N = length(people)

    b = c()
    for (i in 1:length(walktrap_community)) {
        bi = length(intersect(unlist(walktrap_community[i]), people))
        b = c(b, bi)
    }

    # print(a)
    # print(b)
    # print(N)
    
    C = matrix(0, length(walktrap_community), length(circles))
    for( j in 1: length(walktrap_community)) {
        for(i in 1: length(circles)) {
             C[j, i] = length(intersect(unlist(walktrap_community[j]), unlist(circles[i])))
        }
    }

    HC = 0
    HK = 0
    HCGK = 0
    HKGC = 0
    
    for(i in 1: length(circles)) {
        if(a[i] != 0) {
            HC = HC - a[i] / N * log(a[i] / N)
        }
    }
    
    for(i in 1: length(walktrap_community)) {
        if(b[i] != 0) {
            HK = HK - b[i] / N * log(b[i] / N)
        }
    }
    
    for(j in 1: length(walktrap_community)) {
        for(i in 1: length(circles)) {
            if(C[j, i] != 0) {
                HCGK = HCGK - C[j, i] / N * log(C[j, i] / b[j])
            }
        }
    }
    
    for(i in 1: length(circles)) {
        for(j in 1: length(walktrap_community)) {
            if(C[j, i] != 0) {
                HKGC = HKGC - C[j, i] / N * log(C[j, i] / a[i])
            }
        }
    }

    h = 1 - HCGK / HC
    c = 1 - HKGC / HK

    print(sprintf("Homogeneity for node %s: %f", node_id, h))
    print(sprintf("Completeness for node %s: %f", node_id, c))
}    

dev.off()
