# RESEARCH QUESTION 1: PREDICTING PURCHASE INFLUENCE 
# Load libraries
library(tidyverse)
library(tidymodels)
library(xgboost)
library(ranger)
library(glmnet)
library(shapviz)
library(probably)
library(future.apply)
library(yardstick)
library(clustMixType)
library(mclust)

# 0. LOAD DATA
survey2023 <- read_csv('results_2023.csv')

# 1. FILTER TO RELEVANT COHORT
filteredRespondents <- survey2023 %>%
  filter(
    MainBranch == "I am a developer by profession",
    str_detect(Employment, "Employed, full-time"),
    !is.na(PurchaseInfluence)
  )
c(raw = nrow(survey2023), cohort = nrow(filteredRespondents))

# STEP 2: FEATURE ENGINEERING (BEFORE any column drops)
filteredRespondents <- filteredRespondents %>%
  mutate(usesAiTools = as.integer(!is.na(`AIToolCurrently Using`)))

# Tool indicators above a 5% prevalence floor 
makeToolIndicators <- function(data, col, prefix, minPrev = 0.05) {
  tokens <- data %>%
    select(rowID, all_of(col)) %>%
    separate_rows(all_of(col), sep = ";") %>%
    mutate(token = str_trim(.data[[col]])) %>%
    filter(!is.na(token), token != "")
  
  keep <- tokens %>%
    distinct(rowID, token) %>%
    count(token) %>%
    filter(n / nrow(data) >= minPrev) %>%
    pull(token)
  
  wide <- tokens %>%
    filter(token %in% keep) %>%
    distinct(rowID, token) %>%
    mutate(value = 1L,
           token = paste0(prefix, "_", make.names(token))) %>%
    pivot_wider(names_from = token, values_from = value, values_fill = 0L)
  
  data %>%
    left_join(wide, by = "rowID") %>%
    mutate(across(starts_with(paste0(prefix, "_")), ~ replace_na(.x, 0L)))
}

filteredRespondents <- filteredRespondents %>%
  mutate(rowID = row_number()) %>%
  makeToolIndicators("LanguageHaveWorkedWith",  "lang", minPrev = 0.05) %>%
  makeToolIndicators("DatabaseHaveWorkedWith",  "db",   minPrev = 0.05) %>%
  makeToolIndicators("PlatformHaveWorkedWith",  "plat", minPrev = 0.05) %>%
  makeToolIndicators("WebframeHaveWorkedWith",  "web",  minPrev = 0.05) %>%
  makeToolIndicators("ToolsTechHaveWorkedWith", "tool", minPrev = 0.05)

# Technologies surviving the 5% floor, for Data
retainedTools <- filteredRespondents %>%
  select(starts_with("lang_"), starts_with("db_"), 
         starts_with("plat_"), starts_with("web_"), starts_with("tool_"))
sort(colSums(retainedTools), decreasing = TRUE)

# Theory-driven flags
filteredRespondents <- filteredRespondents %>%
  mutate(
    usesEnterpriseCloud = if_else(
      plat_Amazon.Web.Services..AWS. == 1 | 
        plat_Microsoft.Azure == 1 | 
        plat_Google.Cloud == 1, 
      1L, 0L, missing = 0L
    ),
    
    usesDevopsTooling = if_else(
      tool_Docker == 1 | 
        tool_Kubernetes == 1 | 
        tool_Terraform == 1 | 
        tool_Ansible == 1, 
      1L, 0L, missing = 0L
    ),
    
    usesMsEnterprise = if_else(
      lang_C. == 1 | 
        lang_PowerShell == 1 | 
        db_Microsoft.SQL.Server == 1 | 
        web_ASP.NET == 1 | 
        web_ASP.NET.CORE == 1 | 
        tool_MSBuild == 1 | 
        tool_NuGet == 1 | 
        tool_Visual.Studio.Solution == 1, 
      1L, 0L, missing = 0L
    ),
    
        usesModernJsStack = if_else(
      lang_TypeScript == 1 | 
        web_React == 1 | 
        web_Next.js == 1 | 
        web_Vue.js == 1 | 
        plat_Vercel == 1 | 
        plat_Netlify == 1 | 
        tool_Vite == 1 | 
        tool_npm == 1 | 
        tool_pnpm == 1, 
      1L, 0L, missing = 0L
    ),
    
    usesSystemsProgramming = if_else(
      lang_C == 1 | 
        lang_C.. == 1 | 
        lang_Rust == 1 | 
        lang_Go == 1 | 
        tool_Make == 1 | 
        tool_CMake == 1 | 
        tool_GNU.GCC == 1 | 
        tool_LLVM.s.Clang == 1, 
      1L, 0L, missing = 0L
    ),
    
    usesTraditionalWeb = if_else(
      lang_PHP == 1 | 
        web_WordPress == 1 | 
        web_jQuery == 1 | 
        web_AngularJS == 1, 
      1L, 0L, missing = 0L
    )
  ) %>%
  select(-c(
    plat_Amazon.Web.Services..AWS., plat_Microsoft.Azure, plat_Google.Cloud,
    tool_Docker, tool_Kubernetes, tool_Terraform, tool_Ansible,
    lang_C., lang_PowerShell, db_Microsoft.SQL.Server, web_ASP.NET, web_ASP.NET.CORE, 
    tool_MSBuild, tool_NuGet, tool_Visual.Studio.Solution,
    lang_TypeScript, web_React, web_Next.js, web_Vue.js, plat_Vercel, plat_Netlify, 
    tool_Vite, tool_npm, tool_pnpm,
    lang_C, lang_C.., lang_Rust, lang_Go, tool_Make, tool_CMake, tool_GNU.GCC, tool_LLVM.s.Clang,
    lang_PHP, web_WordPress, web_jQuery, web_AngularJS
  ))

# Collapse developer types into known categories (management, DevOps/Infra,
# Data/ML, Embedded/Desktop, Mobile, Full-stack, Back-end, Front-end, Other)
filteredRespondents <- filteredRespondents %>%
  mutate(
    devRole = case_when(
      is.na(DevType) ~ NA_character_,
      str_detect(DevType, regex("manager|executive|vp", ignore_case = TRUE)) ~ "Management",
      str_detect(DevType, regex("devops|cloud|site reliability|security|system administrator|database administrator", ignore_case = TRUE)) ~ "DevOps/Infra",
      str_detect(DevType, regex("data scientist|machine learning|engineer, data|data or business analyst|scientist", ignore_case = TRUE)) ~ "Data/ML",
      str_detect(DevType, regex("embedded|desktop|hardware|game", ignore_case = TRUE)) ~ "Embedded/Desktop",
      str_detect(DevType, regex("mobile", ignore_case = TRUE))     ~ "Mobile",
      str_detect(DevType, regex("full-stack", ignore_case = TRUE)) ~ "Full-stack",
      str_detect(DevType, regex("back-end", ignore_case = TRUE))   ~ "Back-end",
      str_detect(DevType, regex("front-end", ignore_case = TRUE))  ~ "Front-end",
      TRUE ~ "Other"
    )
  ) %>%
  select(-DevType)

# Bin Years of Professional Coding
filteredRespondents <- filteredRespondents %>%
  mutate(
    yearsNumeric = case_when(
      YearsCodePro == "Less than 1 year" ~ 0,
      YearsCodePro == "More than 50 years" ~ 51,
      TRUE ~ as.numeric(YearsCodePro)
    ),
    experienceLevel = case_when(
      is.na(yearsNumeric) ~ NA_character_,
      yearsNumeric <= 3   ~ "Junior (0-3 yrs)",
      yearsNumeric <= 8   ~ "Mid-Level (4-8 yrs)",
      yearsNumeric <= 15  ~ "Senior (9-15 yrs)",
      TRUE                 ~ "Veteran (16+ yrs)"
    )
  ) %>%
  select(-YearsCodePro, -yearsNumeric)

# Create breadth count for use of tools and technologies
filteredRespondents <- filteredRespondents %>%
  mutate(
    across(
      .cols = ends_with("HaveWorkedWith") | ends_with("WantToWorkWith"),
      .fns  = ~ if_else(is.na(.x), 0L, str_count(.x, ";") + 1L),
      .names = "{.col}_Count"
    )
  ) %>%
  select(-(ends_with("HaveWorkedWith") | ends_with("WantToWorkWith")))

# STEP 3: DROP IRRELEVANT FEATURES
analyticalSample <- filteredRespondents %>%
  select(
    -rowID,
    -ResponseId, -Q120, -MainBranch,
    -SurveyLength, -SurveyEase,
    -TBranch, -ICorPM, -WorkExp,
    -starts_with("Knowledge_"),
    -starts_with("Frequency_"),
    -TimeSearching, -TimeAnswering,
    -ProfessionalTech, -Industry,
    -Currency, -CompTotal, -ConvertedCompYearly,
    -TechList, -BuyNewTool,
    -LearnCodeOnline, -`OpSysPersonal use`
  )

# STEP 4: RECODE TARGET VARIABLE
analyticalSample <- analyticalSample %>%
  mutate(
    binaryInfluence = case_when(
      PurchaseInfluence %in% c("I have little or no influence",
                               "I have some influence") ~ "Low_Influence",
      PurchaseInfluence == "I have a great deal of influence" ~ "High_Influence",
      TRUE ~ NA_character_
    ),
    # First level = event class in yardstick's default (event_level = "first")
    binaryInfluence = factor(binaryInfluence,
                             levels = c("High_Influence", "Low_Influence"))
  ) %>%
  select(-PurchaseInfluence)

# Visualize the class distribution of the target variable
analyticalSample %>% count(binaryInfluence) %>%
  mutate(pct = round(100 * n / sum(n), 1))

# STEP 5: MISSINGNESS-BASED COLUMN DROP (>30%)
missingPercentage <- colMeans(is.na(analyticalSample)) * 100
columnsToDrop <- names(missingPercentage[missingPercentage > 30])

analyticalSample <- analyticalSample %>% select(-all_of(columnsToDrop))
columnsToDrop  # columns removed for >30% missingness

# STEP 6: TRAIN-TEST STRATIFIED SPLIT (80/20)
analyticalSample <- analyticalSample %>%
  mutate(across(where(is.character), as.factor))

set.seed(2026)
dataSplit <- initial_split(analyticalSample, prop = 0.80, strata = binaryInfluence)
trainData <- training(dataSplit)
testData  <- testing(dataSplit)

set.seed(2026)
cvFolds <- vfold_cv(trainData, v = 5, strata = binaryInfluence)

# STEP 7: PRE-PROCESSING "RECIPES" FROM tidymodels PACKAGE

# XGBoost — native NA handling, no imputation, no normalization
xgbRecipe <- recipe(binaryInfluence ~ ., data = trainData) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.03) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

# Random Forest — ranger CANNOT handle NAs, so impute;
rfRecipe <- recipe(binaryInfluence ~ ., data = trainData) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.03) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

# Elastic net logistic regression — imputation + normalization
linearRecipe <- recipe(binaryInfluence ~ ., data = trainData) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.80) %>%
  step_other(all_nominal_predictors(), threshold = 0.03) %>%
  step_dummy(all_nominal_predictors(), one_hot = FALSE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# STEP 8: MODEL SPECIFICATIONS & TUNING GRIDS
# XGBoost
xgbSpec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  min_n          = tune(),
  mtry           = tune(),
  loss_reduction = tune(),
  sample_size    = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

set.seed(2026)
xgbGrid <- grid_space_filling(
  trees(range = c(300, 1500)),
  tree_depth(range = c(3, 10)),
  learn_rate(range = c(-2.5, -0.7)),   
  min_n(range = c(5, 40)),
  mtry(range = c(5, 30)),
  loss_reduction(),
  sample_size = sample_prop(range = c(0.6, 1.0)),
  size = 30
)

xgbWf <- workflow() %>% add_recipe(xgbRecipe) %>% add_model(xgbSpec)

# Random Forest 
rfSpec <- rand_forest(
  trees = 500,
  mtry  = tune(),
  min_n = tune()
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

set.seed(2026)
rfGrid <- grid_space_filling(
  mtry(range = c(5, 25)),
  min_n(range = c(2, 40)),
  size = 10
)

rfWf <- workflow() %>% add_recipe(rfRecipe) %>% add_model(rfSpec)

# Elastic net logistic regression
logSpec <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

logGrid <- grid_regular(
  penalty(range = c(-4, -1)),
  mixture(range = c(0, 1)),
  levels = c(penalty = 5, mixture = 3)
)

logWf <- workflow() %>% add_recipe(linearRecipe) %>% add_model(logSpec)

# STEP 9: TUNING 
tuneMetrics <- metric_set(roc_auc, pr_auc)
ctrl <- control_grid(save_pred = TRUE)
set.seed(2026)
xgbTuneResults <- tune_grid(xgbWf, resamples = cvFolds, grid = xgbGrid,
                              metrics = tuneMetrics, control = ctrl)
set.seed(2026)
rfTuneResults <- tune_grid(rfWf, resamples = cvFolds, grid = rfGrid,
                             metrics = tuneMetrics, control = ctrl)
set.seed(2026)
logTuneResults <- tune_grid(logWf, resamples = cvFolds, grid = logGrid,
                              metrics = tuneMetrics, control = ctrl)


# CV comparison (mean +/- SE across the same folds)
cvSummary <- bind_rows(
  show_best(xgbTuneResults, metric = "roc_auc", n = 1) %>% mutate(model = "XGBoost"),
  show_best(rfTuneResults,  metric = "roc_auc", n = 1) %>% mutate(model = "Random Forest"),
  show_best(logTuneResults, metric = "roc_auc", n = 1) %>% mutate(model = "Elastic Net LR")
) %>%
  select(model, mean, std_err, n)

cvSummary

bestXgbParams <- select_best(xgbTuneResults, metric = "roc_auc")
bestRfParams  <- select_best(rfTuneResults,  metric = "roc_auc")
bestLogParams <- select_best(logTuneResults, metric = "roc_auc")

# Save all data
saveRDS(xgbTuneResults, "xgb_tune.rds")
saveRDS(rfTuneResults,  "rf_tune.rds")
saveRDS(logTuneResults, "log_tune.rds")

# STEP 10: FINALIZE MODELS & EVALUATE ON TEST SET
finalMetrics <- metric_set(roc_auc, pr_auc, f_meas, accuracy, mn_log_loss)
finalXgbRes <- finalize_workflow(xgbWf, bestXgbParams) %>%
  last_fit(split = dataSplit, metrics = finalMetrics)
finalRfRes  <- finalize_workflow(rfWf, bestRfParams) %>%
  last_fit(split = dataSplit, metrics = finalMetrics)
finalLogRes <- finalize_workflow(logWf, bestLogParams) %>%
  last_fit(split = dataSplit, metrics = finalMetrics)

# Evaluate Naive Baseline 
nullSpec <- null_model() %>% 
  set_engine("parsnip") %>% 
  set_mode("classification")

nullWf <- workflow() %>% 
  add_formula(binaryInfluence ~ .) %>%   
  add_model(nullSpec)

finalNullRes <- last_fit(nullWf, split = dataSplit, metrics = finalMetrics)

# Threshold-free and default-threshold (0.5) test-set metrics
resultsDefault <- bind_rows(
  collect_metrics(finalNullRes) %>% mutate(model = "Null baseline"),
  collect_metrics(finalLogRes)  %>% mutate(model = "Elastic Net LR"),
  collect_metrics(finalRfRes)   %>% mutate(model = "Random Forest"),
  collect_metrics(finalXgbRes)  %>% mutate(model = "XGBoost")
) %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

# STEP 11: PER-MODEL THRESHOLD ANALYSIS
# Derive F1-optimal threshold from CV predictions
getThreshold <- function(tuneResults, bestParams) {
  collect_predictions(tuneResults, parameters = bestParams) %>%
    probably::threshold_perf(
      truth      = binaryInfluence,
      estimate   = .pred_High_Influence,
      thresholds = seq(0.05, 0.95, by = 0.01),
      metrics    = metric_set(f_meas)
    ) %>%
    filter(.metric == "f_meas") %>%
    slice_max(.estimate, n = 1, with_ties = FALSE) %>%
    pull(.threshold)
}

thrXgb <- getThreshold(xgbTuneResults, bestXgbParams)
thrRf  <- getThreshold(rfTuneResults,  bestRfParams)
thrLog <- getThreshold(logTuneResults, bestLogParams)

thresholdsByModel <- tibble(
  model     = c("Elastic Net LR", "Random Forest", "XGBoost"),
  threshold = c(thrLog, thrRf, thrXgb)
)
thresholdsByModel

# Apply threshold to model test-set predictions
classifyAtThreshold <- function(finalRes, threshold) {
  collect_predictions(finalRes) %>%
    mutate(predClassTuned = factor(
      if_else(.pred_High_Influence >= threshold,
              "High_Influence", "Low_Influence"),
      levels = levels(testData$binaryInfluence)
    ))
}

tunedTestMetrics <- function(finalRes, threshold, modelName) {
  preds <- classifyAtThreshold(finalRes, threshold)
  bind_rows(
    f_meas(preds,      truth = binaryInfluence, estimate = predClassTuned),
    accuracy(preds,    truth = binaryInfluence, estimate = predClassTuned),
    roc_auc(preds,     truth = binaryInfluence, .pred_High_Influence),
    pr_auc(preds,      truth = binaryInfluence, .pred_High_Influence),
    mn_log_loss(preds, truth = binaryInfluence, .pred_High_Influence)
  ) %>%
    mutate(model = modelName) %>%
    select(model, .metric, .estimate) %>%
    pivot_wider(names_from = .metric, values_from = .estimate)
}

resultsTuned <- bind_rows(
  tunedTestMetrics(finalLogRes, thrLog, "Elastic Net LR (tuned threshold)"),
  tunedTestMetrics(finalRfRes,  thrRf,  "Random Forest (tuned threshold)"),
  tunedTestMetrics(finalXgbRes, thrXgb, "XGBoost (tuned threshold)")
)

conf_mat(classifyAtThreshold(finalXgbRes, thrXgb),
         truth = binaryInfluence, estimate = predClassTuned)
resultsTable <- bind_rows(resultsDefault, resultsTuned)
resultsTable
write_csv(resultsTable, "results_test_set.csv")

# STEP 12: SHAP FEATURE IMPORTANCE AND BEESWARM PLOT
fittedWf     <- extract_workflow(finalXgbRes)
fittedRecipe <- extract_recipe(fittedWf)
finalXgbFit <- extract_fit_engine(fittedWf)

bakedTest <- bake(fittedRecipe, new_data = testData) %>%
  select(-binaryInfluence) %>%
  as.matrix()

shp <- shapviz(finalXgbFit, X_pred = bakedTest)

# APA theme for APA-appropriate SHAP plots
apaTheme <- theme_classic() +
  theme(
    text = element_text(size = 12, color = "black"),
    axis.line = element_line(color = "black"),
    axis.text = element_text(color = "black"),
    axis.title = element_text(face = "bold"),
    legend.title = element_text(face = "bold"),
    plot.title = element_blank(),    
    plot.subtitle = element_blank() 
  )

# Plot SHAP Beeswarm
shapBeeswarm <- sv_importance(shp, kind = "beeswarm", max_display = 20) +
  apaTheme +
  labs(
    x = "SHAP Value (Impact on Model Output)",
    y = "" # Leaves Y-axis blank since feature names are self-explanatory
  )
print(shapBeeswarm)
ggsave("shapBeeswarm.png", plot = shapBeeswarm, width = 8, height = 6, dpi = 300)

# Plot SHAP Bar Plot
shapBar <- sv_importance(shp, kind = "bar", max_display = 20) +
  apaTheme +
  labs(
    x = "Mean Absolute SHAP Value (Global Importance)",
    y = ""
  )
print(shapBar)
ggsave("shapBar.png", plot = shapBar, width = 8, height = 6, dpi = 300)

saveRDS(xgbTuneResults, "xgb_tune.rds")
saveRDS(rfTuneResults,  "rf_tune.rds")
saveRDS(logTuneResults, "log_tune.rds")

# Gain Analysis for the final XGBoost model on the test set
testPreds <- collect_predictions(finalXgbRes)
gainData <- gain_curve(testPreds, truth = binaryInfluence, .pred_High_Influence)
gainPlot <- autoplot(gainData) +
  labs(
    title = "Cumulative Gains Curve for High-Value Customer Identification",
    x = "% Tested (Ranked by Predicted Probability)",
    y = "% of True Champions Found"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.minor = element_blank()
  )
print(gainPlot)
ggsave("cumulativeGains.png", plot = gainPlot, width = 8, height = 6, dpi = 300)

# Lift Analysis for the final XGBoost model on the test set
liftData <- lift_curve(testPreds, truth = binaryInfluence, .pred_High_Influence)
liftPlot <- autoplot(liftData) +
  labs(
    title = "Predictive Lift Curve Relative to Random Baseline",
    x = "% Tested (Ranked by Predicted Probability)",
    y = "Lift (Multiplier over Random Baseline)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.minor = element_blank()
  )
print(liftPlot)
ggsave("liftChart.png", plot = liftPlot, width = 8, height = 6, dpi = 300)

# Print exact numbers for Results section (lift at top 10%, gain at the top 20%)
liftData %>%
  slice_min(abs(.percent_tested - 10)) %>%
  select(.percent_tested, .lift)
gainData %>%
  slice_min(abs(.percent_tested - 20)) %>%
  select(.percent_tested, .percent_found)

# RESEARCH QUESTION 2: DEVELOPER PERSONA CLUSTERING

# STEP 1: PREPARE FULL DATASET & PREPROCESS
clusteringSample <- analyticalSample %>% 
  select(-binaryInfluence) %>%
  mutate(across(
    c(starts_with("lang_"), starts_with("db_"), starts_with("plat_"),
      starts_with("web_"), starts_with("tool_"), starts_with("uses")),
    ~ factor(.x, levels = c(0L, 1L), labels = c("No", "Yes"))
  ))

kprotoRecipeFinal <- recipe(~ ., data = clusteringSample) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.02) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.80) %>% 
  step_normalize(all_numeric_predictors())

finalDataPrepped <- prep(kprotoRecipeFinal) %>% 
  bake(new_data = NULL) %>%
  mutate(across(where(is.character), as.factor)) %>%
  as.data.frame()

# STEP 2: Find optimal number of clusters k
set.seed(2026)
valIndices <- sample(seq_len(nrow(finalDataPrepped)),
                      size = min(4000, nrow(finalDataPrepped)))
validationDataPrepped <- finalDataPrepped[valIndices, ]

# Determining optimal k using Elbow + Silhouette methods
kValues <- 2:10
wssScores <- numeric(length(kValues))
silScores <- numeric(length(kValues))

set.seed(2026)
for (i in seq_along(kValues)) {
  kp <- kproto(
    x = validationDataPrepped,
    k = kValues[i],
    nstart = 10,
    keep.data = TRUE,     
    verbose = FALSE
  )
  wssScores[i] <- kp$tot.withinss
  silScores[i] <- validation_kproto(method = "silhouette", object = kp)
  cat(sprintf("   Tested k = %d\n", kValues[i]))
}

# Plot metrics side-by-side for visual inspection
par(mfrow = c(1, 2))
plot(kValues, wssScores, type = "b", pch = 19, col = "darkred", lwd = 2,
     xlab = "Number of Clusters (k)",
     ylab = "Total Within-Cluster Distance (WSS)",
     main = "Elbow Method")
plot(kValues, silScores, type = "b", pch = 19, col = "darkblue", lwd = 2,
     xlab = "Number of Clusters (k)",
     ylab = "Mean Silhouette Score",
     main = "Silhouette Analysis")
par(mfrow = c(1, 1))

# STEP 3: FIT FINAL K-PROTOTYPES MODEL BASED ON SELECTED OPTIMAL k
optimalK <- 2
set.seed(2026)
finalKproto <- kproto(
  x = finalDataPrepped, 
  k = optimalK, 
  nstart = 10,
  verbose = FALSE
)

# STEP 4: EVALUATING CLUSTER STABILITY (via Bootstrapping & Adjusted Rand Index)
plan(multisession, workers = 4) 

nBootstraps <- 200
ariScores <- future_lapply(seq_len(nBootstraps), function(i) {
  bootstrapIndices <- sample(seq_len(nrow(finalDataPrepped)), replace = TRUE)
  bootstrapData    <- finalDataPrepped[bootstrapIndices, ]
  
  bootstrapModel <- kproto(bootstrapData, k = optimalK, nstart = 5, verbose = FALSE)
  bootstrapPredictions <- predict(bootstrapModel, newdata = finalDataPrepped)
  adjustedRandIndex(finalKproto$cluster, bootstrapPredictions$cluster)
}, future.seed = 2026)

ariScores <- unlist(ariScores)
plan(sequential)

meanAri <- mean(ariScores)
ciAri <- quantile(ariScores, probs = c(0.025, 0.975))

# Print cluster stability results
cat(sprintf("Mean ARI: %.3f\n", meanAri))
cat(sprintf("95%% CI:  [%.3f, %.3f]\n", ciAri[1], ciAri[2]))

# STEP 5: PROFILING THE CLUSTERS
clusteringResults <- analyticalSample %>%
  mutate(Cluster = as.factor(finalKproto$cluster))

influenceSummary <- clusteringResults %>%
  group_by(Cluster) %>%
  summarise(
    clusterSize = n(),
    champions = sum(binaryInfluence == "High_Influence", na.rm = TRUE),
    influencePercentage = round((champions / clusterSize) * 100, 1),
    .groups = "drop"
  )

getMode <- function(x) {
  uniqueValues <- unique(na.omit(x))
  uniqueValues[which.max(tabulate(match(x, uniqueValues)))]
}

getTopCategories <- function(x, n = 3) {
  x <- na.omit(x)
  if(length(x) == 0) return(NA_character_)
  counts <- sort(table(x), decreasing = TRUE)
  pcts <- round((counts / sum(counts)) * 100, 1)
  top_n <- head(pcts, n)
  formatted_strings <- paste0(names(top_n), " (", top_n, "%)")
  paste(formatted_strings, collapse = " | ")
}

categoricalProfiles <- clusteringResults %>%
  group_by(Cluster) %>%
  summarise(
    topRoles = getTopCategories(devRole, n = 3),
    topExperience = getTopCategories(experienceLevel, n = 3),
    topOrgSize = getTopCategories(OrgSize, n = 3),
    topWorkStyle = getTopCategories(RemoteWork, n = 2),
    .groups = "drop"
  )

# 1. Update numericProfiles to include all 7 composite flags
numericProfiles <- clusteringResults %>%
  group_by(Cluster) %>%
  summarise(
    meanPlatformCount = round(mean(PlatformHaveWorkedWith_Count, na.rm = TRUE), 1),
    meanAICount = round(mean(AIDevHaveWorkedWith_Count, na.rm = TRUE), 1),
    meanToolsWorkedWith = round(mean(ToolsTechHaveWorkedWith_Count, na.rm = TRUE), 1),
    meanProgLanguageCount = round(mean(LanguageHaveWorkedWith_Count, na.rm = TRUE), 1),
    
    # All 7 categorical flags converted to adoption percentages
    aiPercentage = round(mean(usesAiTools, na.rm = TRUE) * 100, 1),
    enterpriseCloudPercentage = round(mean(usesEnterpriseCloud, na.rm = TRUE) * 100, 1),
    devOpsPercentage = round(mean(usesDevopsTooling, na.rm = TRUE) * 100, 1),
    sysEngPercentage = round(mean(usesSystemsProgramming, na.rm = TRUE) * 100, 1),
    msEnterprisePercentage = round(mean(usesMsEnterprise, na.rm = TRUE) * 100, 1),
    modernJsPercentage = round(mean(usesModernJsStack, na.rm = TRUE) * 100, 1),
    traditionalWebPercentage = round(mean(usesTraditionalWeb, na.rm = TRUE) * 100, 1),
    .groups = "drop"
  )

comprehensiveProfiling <- influenceSummary %>%
  left_join(categoricalProfiles, by = "Cluster") %>%
  left_join(numericProfiles, by = "Cluster") %>%
  arrange(desc(influencePercentage))

write_csv(clusteringResults, "rq2ClusterAssignmentsFinal.csv")
write_csv(comprehensiveProfiling, "rq2PersonaProfilesFinal.csv")
saveRDS(finalKproto, "rq2KprotoModelFinal.rds")
stabilityResults <- tibble(meanAri = meanAri, ciLower = ciAri[1], ciUpper = ciAri[2])
write_csv(stabilityResults, "rq2ClusterStabilityFinal.csv")

# Generating visuals for Results section 5.2

# STEP 6: GENERATE PROFILING VISUALS

# 1. Purchase Influence by Cluster
plot_influence <- ggplot(influenceSummary, aes(x = Cluster, y = influencePercentage, fill = Cluster)) +
  geom_col(width = 0.6, color = "black") +
  geom_text(aes(label = paste0(influencePercentage, "%")), vjust = -0.5, fontface = "bold") +
  scale_fill_manual(values = c("#2c3e50", "#e74c3c")) +
  labs(
    y = "Percentage of High Influence Champions",
    x = "Developer Persona (Cluster)"
  ) +
  theme_classic() +
  theme(legend.position = "none",
        text = element_text(size = 12, color = "black"),
        axis.title = element_text(face = "bold"))

print(plot_influence)
ggsave("cluster_influence_bar.png", plot = plot_influence, width = 6, height = 5, dpi = 300)


# 2. Experience Level Distribution per Cluster
plot_experience <- clusteringResults %>%
  count(Cluster, experienceLevel) %>%
  group_by(Cluster) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = Cluster, y = pct, fill = experienceLevel)) +
  geom_col(position = "fill", color = "black") +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_brewer(palette = "Blues") +
  labs(
    y = "Proportion of Cluster",
    x = "Cluster",
    fill = "Experience Level"
  ) +
  theme_classic() +
  theme(text = element_text(size = 12, color = "black"),
        axis.title = element_text(face = "bold"))

print(plot_experience)
ggsave("cluster_experience_dist.png", plot = plot_experience, width = 7, height = 5, dpi = 300)

# 2. Generate the comprehensive Behavioral and Technological Adoption bar chart
plot_tech_adoption <- numericProfiles %>%
  select(Cluster, aiPercentage, enterpriseCloudPercentage, devOpsPercentage, 
         sysEngPercentage, msEnterprisePercentage, modernJsPercentage, traditionalWebPercentage) %>%
  pivot_longer(cols = -Cluster, names_to = "Metric", values_to = "Percentage") %>%
  # Clean up the metric names for the plot axis
  mutate(Metric = str_replace_all(Metric, "Percentage", "")) %>%
  ggplot(aes(x = reorder(Metric, -Percentage), y = Percentage, fill = Cluster)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, color = "black") +
  
  # --- NEW CODE: ADD EXACT PERCENTAGES ---
  geom_text(aes(label = paste0(Percentage, "%")), 
            position = position_dodge(width = 0.8), # Must match geom_col width
            vjust = -0.5,                           # Pushes text just above the bar
            size = 3.5,                             # Adjust text size if it looks cluttered
            fontface = "bold") +
  # ---------------------------------------

scale_fill_manual(values = c("#2c3e50", "#e74c3c")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) + # Gives text breathing room at the top
  labs(
    y = "Adoption Rate (%)",
    x = "Technology Category"
  ) +
  theme_classic() +
  theme(text = element_text(size = 12, color = "black"),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1))

print(plot_tech_adoption)
ggsave("cluster_tech_adoption_with_labels.png", plot = plot_tech_adoption, width = 9, height = 5, dpi = 300)

# 1. Developer Role Distribution per Cluster
plot_role <- clusteringResults %>%
  filter(!is.na(devRole)) %>% # Remove NAs for a cleaner legend
  count(Cluster, devRole) %>%
  ggplot(aes(x = Cluster, y = n, fill = devRole)) +
  # position = "fill" scales the bars to 100% proportions
  geom_col(position = "fill", color = "black") + 
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_viridis_d(option = "turbo") + # Distinct colors for many categories
  labs(
    y = "Proportion of Persona",
    x = "Developer Persona (Cluster)",
    fill = "Developer Role"
  ) +
  theme_classic() +
  theme(
    text = element_text(size = 12, color = "black"),
    axis.title = element_text(face = "bold")
  )

print(plot_role)
ggsave("cluster_devRole_dist.png", plot = plot_role, width = 8, height = 6, dpi = 300)

# 3. Organization Size Distribution per Cluster
plot_orgsize <- clusteringResults %>%
  filter(!is.na(OrgSize)) %>%
  # Ensure correct ordering from smallest to largest
  mutate(OrgSize = factor(OrgSize, 
                          levels = c("Just me - I am a freelancer, sole proprietor, etc.",
                                     "2 to 9 employees",
                                     "10 to 19 employees",
                                     "20 to 99 employees",
                                     "100 to 499 employees",
                                     "500 to 999 employees",
                                     "1,000 to 4,999 employees",
                                     "5,000 to 9,999 employees",
                                     "10,000 or more employees"))) %>%
  count(Cluster, OrgSize) %>%
  ggplot(aes(x = Cluster, y = n, fill = OrgSize)) +
  geom_col(position = "fill", color = "black") +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_brewer(palette = "Purples") +
  labs(
    y = "Proportion of Persona",
    x = "Developer Persona (Cluster)",
    fill = "Organization Size"
  ) +
  theme_classic() +
  theme(
    text = element_text(size = 12, color = "black"),
    axis.title = element_text(face = "bold")
  )

print(plot_orgsize)
ggsave("cluster_orgsize_dist.png", plot = plot_orgsize, width = 9, height = 6, dpi = 300)

# RQ2: statistical signficant difference between clusters?
tab <- table(clusteringResults$Cluster, clusteringResults$binaryInfluence)
tab
prop.table(tab, 1) * 100          # row percentages (24.8 / 16.2)

cs <- chisq.test(tab, correct = FALSE)
cs

cramersV <- sqrt(as.numeric(cs$statistic) / sum(tab))
cramersV                         

# For hyperparameter Appendix:

# For the Elastic Net Logistic Regression
print(bestLogParams)

# For the Random Forest
print(bestRfParams)

# For the XGBoost Model
print(bestXgbParams)