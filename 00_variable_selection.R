# Variable selection

# Load package(s) ####################################################
library(tidymodels)
library(tidyverse)
library(doMC)

# handle common conflicts
tidymodels_prefer()

load("results/tuning_setup")
registerDoMC(cores = 5)

############## initial recipe for lasso selection ############################
initial_recipe <- recipe(y ~., data = training) %>%
  step_rm(id) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_impute_mean(all_predictors())

initial_recipe %>% 
  prep() %>% 
  bake(new_data = NULL)

############## variable selection using lasso ###################################
lasso_mod <- logistic_reg(mode = "classification",
                          penalty = tune(),
                          mixture = 1) %>% 
  set_engine("glmnet")

lasso_params <- extract_parameter_set_dials(lasso_mod)
lasso_grid <- grid_regular(lasso_params, levels = 5)

lasso_workflow <- workflow() %>% 
  add_model(lasso_mod) %>% 
  add_recipe(initial_recipe)

lasso_tune <- lasso_workflow %>% 
  tune_grid(resamples = folds,
            grid = lasso_grid)

save(lasso_tune, file = "results/lasso_tune.rda")

lasso_wkflw_final <- lasso_workflow %>% 
  finalize_workflow(select_best(lasso_tune, metric = "roc_auc"))

lasso_fit <- fit(lasso_wkflw_final, data = training)

lasso_tidy <- lasso_fit %>% 
  tidy() %>%
  filter(estimate != 0 & estimate > 1e-05 | estimate < -1e-05)

View(lasso_tidy)

recipe3_lasso <- recipe(y ~ x017+x023+x024+x034+x093+
                        x122+x153+x155 +x158+x178+x181+
                        x182+x212+x242+x274+x289+x297+
                        x308+x319+x337+x352+x361+x387+
                        x401+x412+x416+x418+x426+x433+
                        x453+x468+x496+x511+x531+x533+
                        x534+x561+x562+x594+x603+x614+
                        x627+x642+x664+x665+x674+x675+
                        x687+x725+x728+x730+x737+x764,
                  data = training) %>% 
  step_impute_mean(all_numeric_predictors()) %>% 
  step_impute_mode(all_factor_predictors()) %>% 
  step_corr(all_predictors()) %>% 
  # Remove variables with zero variance
  step_nzv(all_predictors()) %>% 
  # Center and scale all predictors
  step_normalize(all_numeric_predictors()) %>% 
  step_YeoJohnson(all_numeric_predictors())

save(recipe3_lasso, file = "results/recipe3_lasso")