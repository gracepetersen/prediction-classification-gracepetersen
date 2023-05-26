# KNN

# Load packages
library(tidyverse)
library(tidymodels)
library(tictoc)
library(doMC)

# Handle common conflicts
tidymodels_prefer()

# Load in tuning data
load("results/tuning_setup")
load("results/recipe1")

# Set up parallel processing -----
## Unix and macOS only
registerDoMC(cores = 8)

# Create model
knn_model <- nearest_neighbor(mode = "classification",
                              neighbors = tune()) %>% 
  set_engine("kknn")

# Params 
knn_params <- parameters(knn_model)

# Create grid 
knn_grid <- grid_regular(knn_params, levels = 5)

# Random forest
knn_workflow1 <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(recipe1)


# Tune model
knn_tuned1 <- tune_grid(knn_workflow1, 
                        folds,
                        grid = knn_grid,
                        control = control_grid(save_pred = TRUE, 
                                               save_workflow = TRUE,
                                               verbose = TRUE,
                                               parallel_over = "everything"))

#save as rda
save(knn_tuned1,
     file = "results/knn_tuned1.rda")

load("results/knn_tuned1.rda")

knn_tuned1 %>% 
  show_best()

