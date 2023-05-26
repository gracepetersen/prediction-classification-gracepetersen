# Boosted trees for ensemble

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(doMC)

# Seed
set.seed(3013)

# Handle common conflicts
tidymodels_prefer()

# Load in tuning data
load("results/tuning_setup")
load("results/recipe2")

# Set up parallel processing -----
## Unix and macOS only
registerDoMC(cores = 8)

# Create model
boost_model <- boost_tree(mode = "classification",
                          min_n = tune(),
                          mtry = tune(),
                          learn_rate = tune()) %>% 
  set_engine("xgboost", importance = "impurity")

# Params
boost_params <- parameters(boost_model) %>% 
  update(mtry = mtry(range = c(1,10))) %>% 
  update(learn_rate = learn_rate(range = c(-5, -0.2)))

# Create grid 
boost_grid <- grid_regular(boost_params, levels = 5)

# Random forest
boost_workflow <- workflow() %>% 
  add_model(boost_model) %>% 
  add_recipe(recipe2)

ctrl_grid <- control_stack_grid()


# Tuning/fitting ----
bt_res2 <- boost_workflow %>%
  tune_grid(
    resamples = folds,
    grid =  boost_grid,
    control = ctrl_grid
  )

# Write out results & workflow
save(bt_res2, file = "model_info/bt_res2.rda")
