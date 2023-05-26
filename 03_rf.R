# Boosted trees for ensemble

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)

# Seed
set.seed(3013)

# Handle common conflicts
tidymodels_prefer()

# Load in tuning data
load("results/tuning_setup")
load("results/recipe2")

# Define model ----------------------------------------------------------
rf_model <- rand_forest(mode = "classification",
                        trees = 50,
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger", importance = "impurity")

# Params 
rf_params <- parameters(rf_model) %>% 
  recipes::update(mtry = mtry(range = c(5, 10)))

# Create grid 
rf_grid <- grid_regular(rf_params, levels = 5)

# workflow ----
rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe2)

keep_pred <- control_resamples(save_pred = TRUE)

ctrl_grid <- control_stack_grid()


# Tuning/fitting ----
rf_res <- rf_workflow %>%
  tune_grid(
    resamples = folds,
    grid = rf_grid,
    control = ctrl_grid
  )

# Write out results & workflow
save(rf_res, file = "model_info/rf_res.rda")
