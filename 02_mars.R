# Mars

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

############################################################################
# USING RECIPE 2 ###########################################################

# Load recipe 2
load("results/recipe2")

# Set up model -----
mars_model <- mars(
  mode = "classification",
  num_terms = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth")

# Params
mars_params <- parameters(mars_model) %>% 
  update(num_terms = num_terms(range = c(1, 23)))

# Create grid 
mars_grid <- grid_regular(mars_params, levels = 5)

# Set up workflow
mars_workflow2 <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(recipe2)


# Random forest
mars_tuned2 <- tune_grid(mars_workflow2, 
                        folds,
                        grid = mars_grid,
                        control = control_grid(save_pred = TRUE, # Create an extra column for each prediction
                                               save_workflow = TRUE, # Lets you use extract_workflow
                                               verbose = TRUE,
                                               parallel_over = "everything")) 

#save as rda
save(mars_tuned2,
     file = "results/mars_tuned2.rda")

load("results/mars_tuned2.rda")

mars_tuned2 %>% 
  show_best()


############################################################################
# USING RECIPE 3 LASSO #####################################################

# Load recipe 3
load("results/recipe3_lasso")

# Set up model -----
mars_model <- mars(
  mode = "classification",
  num_terms = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth")

# Params
mars_params <- parameters(mars_model) %>% 
  update(num_terms = num_terms(range = c(10, 40)))

# Create grid 
mars_grid <- grid_regular(mars_params, levels = 5)

# Set up workflow
mars_workflow3 <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(recipe3_lasso)


# Random forest
mars_tuned3_lasso <- tune_grid(mars_workflow3, 
                         folds,
                         grid = mars_grid,
                         control = control_grid(save_pred = TRUE, # Create an extra column for each prediction
                                                save_workflow = TRUE, # Lets you use extract_workflow
                                                verbose = TRUE,
                                                parallel_over = "everything")) 

#save as rda
save(mars_tuned3_lasso,
     file = "results/mars_tuned3.2_lasso.rda")

load("results/mars_tuned3.2_lasso.rda")

mars_table <- mars_tuned3_lasso %>% 
  show_best()

save(mars_table, file = "results/mars_table")

mars_workflow3_tuned <- mars_workflow3 %>% 
  finalize_workflow(select_best(mars_tuned3_lasso, metric = "roc_auc"))

# Fit entire training data set to workflow
mars3_results <- fit(mars_workflow3_tuned, training)

mars3_pred <- testing %>% 
  bind_cols(predict(mars3_results, testing)) %>% 
  select(id, .pred_class) %>% 
  rename(y = .pred_class)

write_csv(mars3_pred, file = "submissions/mars3.2_pred.csv")

