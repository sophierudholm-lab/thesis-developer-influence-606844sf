# RESEARCH QUESTION 1: PREDICTING PURCHASE INFLUENCE (ROBUSTNESS CHECK)
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

# Collapse developer types into known categories
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
      TRUE                ~ "Veteran (16+ yrs)"
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

# STEP 4: RECODE TARGET VARIABLE (ROBUSTNESS CHECK DEFINITION)
# Grouping "Some influence" and "Great deal of influence" together vs "Little or no influence"
analyticalSample <- analyticalSample %>%
  mutate(
    binaryInfluence = case_when(
      PurchaseInfluence == "I have little or no influence" ~ "Low_Influence",
      PurchaseInfluence %in% c("I have some influence", 
                               "I have a great deal of influence") ~ "High_Or_Some_Influence",
      TRUE ~ NA_character_
    ),
    # First level = event class in yardstick's default (event_level = "first")
    binaryInfluence = factor(binaryInfluence,
                             levels = c("High_Or_Some_Influence", "Low_Influence"))
  ) %>%
  select(-PurchaseInfluence)

# Visualize the class distribution of the target variable
analyticalSample %>% count(binaryInfluence) %>%
  mutate(pct = round(100 * n / sum(n), 1))

# STEP 5: MISSINGNESS-BASED COLUMN DROP (>30%)
missingPercentage <- colMeans(is.na(analyticalSample)) * 100
columnsToDrop <- names(missingPercentage[missingPercentage > 30])

analyticalSample <- analyticalSample %>% select(-all_of(columnsToDrop))

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
xgbRecipe <- recipe(binaryInfluence ~ ., data = trainData) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.03) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

rfRecipe <- recipe(binaryInfluence ~ ., data = trainData) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.03) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

linearRecipe <- recipe(binaryInfluence ~ ., data = trainData) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.80) %>%
  step_other(all_nominal_predictors(), threshold = 0.03) %>%
  step_dummy(all_nominal_predictors(), one_hot = FALSE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# STEP 8: MODEL SPECIFICATIONS & TUNING GRIDS
xgbSpec <- boost_tree(
  trees         = tune(),
  tree_depth    = tune(),
  learn_rate    = tune(),
  min_n         = tune(),
  mtry          = tune(),
  loss_reduction = tune(),
  sample_size   = tune()
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
xgbTuneResults <- tune_grid(xgbWf, resamples = cvFolds, grid = xgbGrid, metrics = tuneMetrics, control = ctrl)
set.seed(2026)
rfTuneResults <- tune_grid(rfWf, resamples = cvFolds, grid = rfGrid, metrics = tuneMetrics, control = ctrl)
set.seed(2026)
logTuneResults <- tune_grid(logWf, resamples = cvFolds, grid = logGrid, metrics = tuneMetrics, control = ctrl)

bestXgbParams <- select_best(xgbTuneResults, metric = "roc_auc")
bestRfParams  <- select_best(rfTuneResults,  metric = "roc_auc")
bestLogParams <- select_best(logTuneResults, metric = "roc_auc")

# STEP 10: FINALIZE MODELS & EVALUATE ON TEST SET
finalMetrics <- metric_set(roc_auc, pr_auc, f_meas, accuracy, mn_log_loss)
finalXgbRes <- finalize_workflow(xgbWf, bestXgbParams) %>% last_fit(split = dataSplit, metrics = finalMetrics)
finalRfRes  <- finalize_workflow(rfWf, bestRfParams) %>% last_fit(split = dataSplit, metrics = finalMetrics)
finalLogRes <- finalize_workflow(logWf, bestLogParams) %>% last_fit(split = dataSplit, metrics = finalMetrics)

# Naive Baseline 
nullSpec <- null_model() %>% set_engine("parsnip") %>% set_mode("classification")
nullWf <- workflow() %>% add_formula(binaryInfluence ~ .) %>% add_model(nullSpec)
finalNullRes <- last_fit(nullWf, split = dataSplit, metrics = finalMetrics)

resultsDefault <- bind_rows(
  collect_metrics(finalNullRes) %>% mutate(model = "Null baseline"),
  collect_metrics(finalLogRes)  %>% mutate(model = "Elastic Net LR"),
  collect_metrics(finalRfRes)   %>% mutate(model = "Random Forest"),
  collect_metrics(finalXgbRes)  %>% mutate(model = "XGBoost")
) %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

# STEP 11: PER-MODEL THRESHOLD ANALYSIS
getThreshold <- function(tuneResults, bestParams) {
  collect_predictions(tuneResults, parameters = bestParams) %>%
    probably::threshold_perf(
      truth      = binaryInfluence,
      estimate   = .pred_High_Or_Some_Influence,
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

classifyAtThreshold <- function(finalRes, threshold) {
  collect_predictions(finalRes) %>%
    mutate(predClassTuned = factor(
      if_else(.pred_High_Or_Some_Influence >= threshold,
              "High_Or_Some_Influence", "Low_Influence"),
      levels = levels(testData$binaryInfluence)
    ))
}

tunedTestMetrics <- function(finalRes, threshold, modelName) {
  preds <- classifyAtThreshold(finalRes, threshold)
  bind_rows(
    f_meas(preds,      truth = binaryInfluence, estimate = predClassTuned),
    accuracy(preds,    truth = binaryInfluence, estimate = predClassTuned),
    roc_auc(preds,     truth = binaryInfluence, .pred_High_Or_Some_Influence),
    pr_auc(preds,      truth = binaryInfluence, .pred_High_Or_Some_Influence),
    mn_log_loss(preds, truth = binaryInfluence, .pred_High_Or_Some_Influence)
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

resultsTable <- bind_rows(resultsDefault, resultsTuned)
print(resultsTable)
write_csv(resultsTable, "results_test_set_robustness.csv")

# STEP 12: SHAP FEATURE IMPORTANCE FOR ROBUSTNESS MODEL
fittedWf     <- extract_workflow(finalXgbRes)
fittedRecipe <- extract_recipe(fittedWf)
finalXgbFit  <- extract_fit_engine(fittedWf)

bakedTest <- bake(fittedRecipe, new_data = testData) %>%
  select(-binaryInfluence) %>%
  as.matrix()

shp <- shapviz(finalXgbFit, X_pred = bakedTest)

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

shapBeeswarm <- sv_importance(shp, kind = "beeswarm", max_display = 20) +
  apaTheme +
  labs(x = "SHAP Value (Impact on Model Output)", y = "")
ggsave("shapBeeswarm_robustness.png", plot = shapBeeswarm, width = 8, height = 6, dpi = 300)

shapBar <- sv_importance(shp, kind = "bar", max_display = 20) +
  apaTheme +
  labs(x = "Mean Absolute SHAP Value (Global Importance)", y = "")
ggsave("shapBar_robustness.png", plot = shapBar, width = 8, height = 6, dpi = 300)