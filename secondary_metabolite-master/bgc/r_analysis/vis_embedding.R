install.packages('ggplot2')
install.packages('RColorBrewer')
install.packages('cowplot')
install.packages('plot3D')


require(ggplot2)
require(RColorBrewer)
require(cowplot)
require(plot3D)

pcs$weight <- sqrt(pcs$weight)

screen_plot <- ggplot(pcs, aes(x=pc, y=weight)) +
                geom_bar(stat = 'identity') +
                xlab('Principal Component') + ylab('Proportion of Variance Explained') +
                theme_minimal()
screen_plot



visualize_2D_embedding <- function(data, name, lbls, title) {
  p <- ggplot(data=data) +
        geom_point(mapping = aes(x=Dim1, y=Dim2, color=as.factor(label))) +
        scale_color_brewer(palette = 'Dark2', name=name, 
                           labels=lbls) +
        ggtitle(title) +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5))
      
  return(p)
}



visualize_accuracy <- function(data, name, lbls) {
  p <- ggplot(data=data) +
        geom_boxplot(mapping = aes(x=model, y=acc, color=as.factor(model))) +
        scale_color_brewer(palette = 'Dark2', name=name,
                           labels=lbls)
  
  return(p)
    
}

p_nca <- visualize_2D_embedding(nca, 'BGC Clusters', lbls, 'Neighboorhood Component Analysis')
p_nca
p_lda <- visualize_2D_embedding(lda, 'BGC Clusters', lbls, 'Linear Discrminant Analysis')
p_lda


umap <- visualize_2D_embedding(umap_train_euclidean_chebyshev, 'BGC Clusters', lbls, 'Non-Parametric UMAP - Training Set Euclidean-Chebyshev')
umap

test <- visualize_2D_embedding(umap_test_euclidean_chebyshev, 'BGC Clusters', lbls, 'Non-Parametric UMAP - Testing Set Euclidean-Chebyshev')
test



lbls <- c('PKS', 'NRPS', 'Terpene')
ratios <- c(384, 275, 227, 252, 64, 127)

df <- data.frame(BGCS=lbls,dist=ratios)
df

p <- ggplot(df, mapping = aes(lbls, dist, fill=lbls)) +
        geom_bar(stat='identity') +
        scale_fill_brewer(palette = 'Dark2', name='Biosynthetic Gene Clusters') +
        xlab('') + ylab('Distribution of BGCs') +
        theme_minimal()

p

t.test(validation_rmse$test_rmse, validation_rmse$validation_rmse, conf.level = 0.95, var.equal = FALSE)
t.test(validation_rmse$test_rmse, validation_rmse$validation_rmse, conf.level = 0.95, var.equal = TRUE)
