# Neural Network

# Load packages
library(tidyverse)
library(tidymodels)
library(doMC)

# Handle common conflicts
tidymodels_prefer()

# Load in tuning data
load("results/tuning_setup")

# Set up parallel processing -----
## Unix and macOS only
registerDoMC(cores = 8)

# Set up model 
nn_model <- mlp(
  mode = "classification",
  hidden_units = tune(),
  penalty = tune()
) %>%
  set_engine("nnet")

# Params
nn_params <- hardhat::extract_parameter_set_dials(nn_model)

# Create grid 
nn_grid <- grid_regular(nn_params, levels = 5)

load("results/recipe1")
# Set up workflow
nn_workflow1 <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(recipe1)

# Tune model
nn_tuned1 <- tune_grid(nn_workflow1, 
                      folds,
                      grid = nn_grid,
                      control = control_grid(save_pred = TRUE, # Create an extra column for each prediction
                                             save_workflow = TRUE, # Lets you use extract_workflow
                                             verbose = TRUE,
                                             parallel_over = "everything")) 


#save as rda
save(nn_tuned1,
     file = "results/nn_tuned1.rda")

load('results/nn_tuned1.rda')

# .558
nn_tuned1 %>% 
  show_best()

nn_workflow_tuned <- nn_workflow1 %>% 
  finalize_workflow(select_best(nn_tuned1, metric = "roc_auc"))

# Fit entire training data set to workflow
nn1_results <- fit(nn_workflow_tuned, training)

nn1_pred <- testing %>% 
  bind_cols(predict(nn1_results, testing)) %>% 
  select(id, .pred_class) %>% 
  rename(y = .pred_class)

write_csv(nn1_pred, file = "submissions/nn1_pred.csv")