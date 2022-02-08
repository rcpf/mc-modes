# Install/load libraries
require(NbClust)

validate_max_nc <- function(max_nc, ds){
    size <- nrow(unique(ds))
    if(size > max_nc + 1){
        return(max_nc)
    }else{
        if(size > 5){
            return(size - 2)
        }else{
            return(1) # Only one cluster
        }
    }
}


computePartitions <- function(ds, index_name, max_nc){
    # Validate number of examples vs. number of clusters
    max_nc <- validate_max_nc(max_nc, ds)
    partitions <- list()
    for (i in index_name){
        res <- NbClust(ds, distance="euclidean", min.nc=2, max.nc=max_nc, method ="kmeans", index=i)
        n <- res$Best.nc[1][['Number_clusters']]
        if (!is.element(paste('', n ,sep=""), partitions)){
            partitions[[paste('', n ,sep="")]] <- res$Best.partition
        }
        gc()
    }
    return (partitions)
}