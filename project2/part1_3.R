library('igraph')
library('Matrix')
library('pracma')

rm(list=ls()) # clear workspace 
g <- read.graph("facebook_combined.txt", directed = FALSE)
#is.connected(g)
#plot(g, vertex.size=3, vertex.label=NA)
deg <- degree(g)
cout <- 0
avg <- 0
for (i in 1:length(deg)) {
    if (deg[i] > 200){
        cout <- cout + 1
        avg <- avg + deg[i]
    }
}
avg <- avg / cout
print(cout)
print(avg)

g1 <-induced_subgraph(g,c(1,neighbors(g,1)), impl = "auto")
disps <- integer(length(V(g1)))
V(g1)$name = V(g1)
for(i in V(g1))
{
    friend <- intersect(neighbors(g1,1),neighbors(g1,i))
    subg <- delete.vertices(g1,c(which(V(g1)$name==i),which(V(g1)$name==i)))
    shortp <- 0
    for(j in 1:length(friend))
    {
        for(k in (j+1): length(friend))
        {
            shortp = c(shortp, shortest.paths(subg, which(V(subg)$name == friend[j]), which(V(subg)$name == friend[k])))
        }
    }
    disps[i] <- sum(shortp)
}
hist_disps <- hist(disps, breaks = seq(min(disps)-0.5, max(disps)+0.5,by=1 ), plot='FALSE')
plot(hist_disps$breaks[-1], hist_disps$counts / length(V(g1)), type='h',
     xlab = "dispersion",ylab = "frequency",main="The distribution of dispersion of the network with ID 1")