## To install package bnlearn, run:
## install.packages("bnlearn");
library(bnlearn);

args = commandArgs(trailingOnly=TRUE)

#filename = 'asia.bif'
#m = 1000
#outputFile = 'testing.csv'
filename = args[1]
m = as.integer(args[2])
outputFile = args[3]

net = read.bif(paste('bif/',filename, sep=''))
data0 = rbn(net, m); ### <--- sampling
card = c() ## cardinality: used for gobnlip
### processing the data, factors to integers
for(i in 1:dim(data0)[2]){
    data0[,i] = as.numeric(as.integer(factor(data0[,i])))
    card = c(card,length(unique(data0[,i])))
}
data0 = data0-1

data0 = rbind(card,data0)

### mod 2 to change 2 to 0
# data0 = data0%%2

#for(i in 1:dim(data0)[2]){
#  if(max(data0[,i])>2){
#    mean_tmp = mean(data0[,i])
#    data0[,i] = as.integer(data0[,i]>mean_tmp)
#  }
#}

### save file
dataFile = paste('./data/',outputFile,sep='')
write.table(data0, dataFile, sep=',', row.names = FALSE)
print(paste('Sampling has been finished. File saved in data/',outputFile, sep=''))
