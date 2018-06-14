
library(igraph)
library(ggplot2)
# Q2
# read stock names to a vector
filepath <- '/Users/wendycui/Documents/Bruin Life/EE232/project5/finance_data/Name_sector.csv'
df_stock_name <- read.csv(filepath, stringsAsFactors = FALSE)
stocks <- df_stock_name$Symbol # store the stocks' names
stocks <- as.vector(stocks)
clean_stocks <- stocks
# construct r_i list ( list of list, key is the name of stock, value is the r_i list of the stock)
r_i_list <- list()
for(st_nm in stocks) { 
  cat("calculating", st_nm, "r_is", "\n")
  fp = paste('/Users/wendycui/Documents/Bruin Life/EE232/project5/finance_data/data/', st_nm, '.csv', sep = "")
  if(nrow(read.csv(fp, stringsAsFactors = FALSE)) == 765){ # clean data
    close_prices = read.csv(fp, stringsAsFactors = FALSE)$Close 
    current_r_i = numeric(0) # array
    for (index in 2:length(close_prices)){
      current_r_i[index - 1] = log(close_prices[index]) - log(close_prices[index - 1])  #calculate r_i 
    }
    r_i_list [[st_nm]] = current_r_i
  } 
  else{
    clean_stocks <- clean_stocks[-which(clean_stocks == st_nm)]
  }
}
# construct correlation dataframe(three colunms: stock i, stock j, weight)
df_corr <- data.frame(stock_i = character(0), stock_j = character(0),  weight = double(0), stringsAsFactors = FALSE) # use this dataframe to build graph

# df_corr$stock_i <- as.vector(df_corr$stock_i)
# df_corr$stock_j <- as.vector(df_corr$stock_j)
# df_corr$weight <- as.vector(df_corr$weight)


#all_w_ij = vector() # weight array
for( i in 1 : ( length(clean_stocks) - 1)) {
  stock_i = as.character(clean_stocks[i]) 
  for( j in (i + 1) : length(clean_stocks)){
    stock_j = as.character(clean_stocks[j])
    cat("compute weight of",stock_i, "and", stock_j,":","\n")
    R_i = r_i_list[[stock_i]]
    R_j = r_i_list[[stock_j]]
    avg_i = mean(R_i) 
    avg_i2 = mean(R_i^2) 
    avg_j = mean(R_j)
    avg_j2 = mean(R_j^2)
    p_ij = (mean(R_i * R_j)  - avg_i * avg_j) / sqrt((avg_i2 - avg_i^2) * (avg_j2 - avg_j^2)) # cross_corelation formula
    w_ij = sqrt(2 * (1 - p_ij))
    #all_w_ij <- append(all_w_ij, w_ij)
    #all_d_ijs = c(all_d_ijs , d_ij)
    row<- c(stock_i, stock_j, w_ij)
    #print(row)
    df_corr <- rbind(df_corr, row,  stringsAsFactors = FALSE) # add one row to correlation dataframe
  }
}
colnames(df_corr) = c("stock_i", "stock_j", "weight")
# create graph and plot degree distribution
# every two vertice have an edge, so all vertices have the same degree
corr_graph <-  graph_from_data_frame(df_corr, directed = FALSE)
plot(degree.distribution(corr_graph), main = "the degree distribution of the correlation graph")

# plot edge weight distribution(un-normalized???)
edge_weights <- df_corr$weight
edge_weights <- as.numeric(edge_weights)
#var(edge_weights)
#mean(edge_weights)
hist( x = edge_weights, col = "dodgerblue3", breaks = seq(from = round(min(edge_weights)), to = round(max(edge_weights)), by = (max(edge_weights)-min(edge_weights))/50), 
      main = "Histogram of Edge Weights", xlab = "edge weights", ylab = "Frequencies")

# create MST

MST <- mst(corr_graph , weights = df_corr$weight)

#assign color to vertices in MST

sectors <- df_stock_name$Sector # all sectors 
sectors <- unique(sectors) # remove dulicates
# color options (11)
colors <- c("brown1", "blueviolet", "chartreuse2", "cyan", "dodgerblue3", "deeppink", "yellow","tomato","gray70","darkorange", "firebrick")
for (i in 1 : length(V(MST))){
  #V(MST)[i]$color = NULL
  V(MST)[i]$color = colors[which(sectors == df_stock_name$Sector[df_stock_name == V(MST)$name[i]])]
}

plot(corr_graph)
# plot MST
# with layout
plot(MST, layout = layout_with_dh, vertex.size = 10, vertex.label.cex = 0.3, main = "MST of the Weighted Correlation Graph")
# without layout
plot(MST, vertex.size = 10, vertex.label = NA, main = "MST of the Weighted Correlation Graph")

plot(MST,vertex.size = 10, vertex.label = NA)


# Q4-1
alpha_case1 = 0
for(v in V(MST)) {
    neighbors = neighbors(MST, v)
    sector = V(MST)[v]$color
    count = 0
    for(n in neighbors) {
        if(sector == V(MST)[n]$color) {
            count = count + 1
        }
    }
    p = count * 1.0 / length(neighbors)
    alpha_case1 = alpha_case1 + p
}
    
alpha_case1 = alpha_case1 * 1.0 / length(V(MST))
print(sprintf("alpha_case1: %f", alpha_case1))

# Q4-2
alpha_case2 = 0
sector_freq = sort(table(V(MST)$color), decreasing = TRUE)
sector_freq = as.data.frame(sector_freq)
for(v in V(MST)) {
    sector = V(MST)[v]$color
    count = sector_freq[sector_freq$Var1==sector,]$Freq
    p = count * 1.0 / length(V(MST))
    alpha_case2 = alpha_case2 + p
}
alpha_case2 = alpha_case2 * 1.0 / length(V(MST))
print(sprintf("alpha_case2: %f", alpha_case2))
