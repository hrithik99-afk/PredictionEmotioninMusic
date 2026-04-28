# PREDICTING EMOTION IN MUSIC (VALENCE & ENERGY PREDICTION)
rm(list = ls())



#  1 PACKAGES

install.packages("tidyverse")
install.packages("caret")
install.packages("randomForest")
install.packages("xgboost")
install.packages("glmnet")
install.packages("kknn")
install.packages("corrplot")
install.packages("patchwork")
install.packages("viridis")

library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(glmnet)
library(kknn)
library(corrplot)
library(patchwork)
library(viridis)

set.seed(42)


# 2 DATA LOADING & PREPARATION


data <- read_csv("data/dataset.csv", show_col_types = FALSE)
names(data) <- gsub(" ", "_", names(data))

data$explicit       <- as.integer(data$explicit == "True")
data$key            <- as.factor(data$key)
data$mode           <- as.factor(data$mode)
data$time_signature <- as.factor(data$time_signature)

cat("Rows:", nrow(data), " Columns:", ncol(data), "\n")

# Genre groups
G1 <- c("salsa", "latin", "reggaeton")       # Rhythmic
G2 <- c("opera", "classical", "piano")       # Classical 
G3 <- c("punk-rock", "heavy-metal", "punk")  # Hard Rock 

g1_data <- data %>% filter(track_genre %in% G1)
g2_data <- data %>% filter(track_genre %in% G2)
g3_data <- data %>% filter(track_genre %in% G3)

groups <- list(
  G1_Rhythmic  = g1_data,
  G2_Classical = g2_data,
  G3_HardRock  = g3_data
)

all_groups <- bind_rows(
  g1_data %>% mutate(Group = "G1: Rhythmic"),
  g2_data %>% mutate(Group = "G2: Classical"),
  g3_data %>% mutate(Group = "G3: Hard Rock")
)

cat("\nGroup sizes:\n")
cat("  G1 Rhythmic  :", nrow(g1_data), "songs\n")
cat("  G2 Classical :", nrow(g2_data), "songs\n")
cat("  G3 Hard Rock :", nrow(g3_data), "songs\n")


#  3. DESCRIPTIVE STATISTICS


desc_vars <- c("valence", "energy", "danceability", "loudness",
               "speechiness", "acousticness", "instrumentalness",
               "liveness", "tempo", "popularity", "duration_ms")

# Summary function
describe_var <- function(x, varname) {
  data.frame(
    Variable = varname,
    N        = sum(!is.na(x)),
    Mean     = round(mean(x,           na.rm = TRUE), 4),
    SD       = round(sd(x,             na.rm = TRUE), 4),
    Min      = round(min(x,            na.rm = TRUE), 4),
    Q1       = round(quantile(x, 0.25, na.rm = TRUE), 4),
    Median   = round(median(x,         na.rm = TRUE), 4),
    Q3       = round(quantile(x, 0.75, na.rm = TRUE), 4),
    Max      = round(max(x,            na.rm = TRUE), 4),
    Skewness = round(
      mean((x - mean(x, na.rm = TRUE))^3, na.rm = TRUE) /
        sd(x, na.rm = TRUE)^3, 4),
    Kurtosis = round(
      mean((x - mean(x, na.rm = TRUE))^4, na.rm = TRUE) /
        sd(x, na.rm = TRUE)^4, 4),
    stringsAsFactors = FALSE
  )
}

# 3.1  Overall summary 


overall_desc <- bind_rows(lapply(desc_vars, function(v)
  describe_var(all_groups[[v]], v)))
print(as.data.frame(overall_desc), row.names = FALSE)

# 3.2  Per group summary

for (grp in c("G1: Rhythmic", "G2: Classical", "G3: Hard Rock")) {
  gdf <- all_groups %>% filter(Group == grp)
  tbl <- bind_rows(lapply(desc_vars, function(v) describe_var(gdf[[v]], v)))
  cat("── Group:", grp, "(n =", nrow(gdf), ") ──\n")
  print(as.data.frame(tbl), row.names = FALSE)
  cat("\n")
}

# 3.3  Per genre summary (valence & energy) 

genre_desc <- all_groups %>%
  group_by(track_genre, Group) %>%
  summarise(
    N              = n(),
    Valence_mean   = round(mean(valence),   4),
    Valence_sd     = round(sd(valence),     4),
    Valence_median = round(median(valence), 4),
    Valence_min    = round(min(valence),    4),
    Valence_max    = round(max(valence),    4),
    Energy_mean    = round(mean(energy),    4),
    Energy_sd      = round(sd(energy),      4),
    Energy_median  = round(median(energy),  4),
    Energy_min     = round(min(energy),     4),
    Energy_max     = round(max(energy),     4),
    .groups = "drop"
  ) %>%
  arrange(Group, track_genre)
print(as.data.frame(genre_desc), row.names = FALSE)

# 3.4  Skewness & kurtosis flags 

skew_flags <- overall_desc %>%
  select(Variable, Mean, SD, Skewness, Kurtosis) %>%
  mutate(
    Notable_skew = ifelse(abs(Skewness) > 1, "YES", "no"),
    Heavy_tails  = ifelse(Kurtosis > 4,      "YES", "no")
  )
print(as.data.frame(skew_flags), row.names = FALSE)

# 3.5  Missing value check 

na_tbl <- data.frame(
  Variable    = desc_vars,
  NA_count    = sapply(desc_vars, function(v) sum(is.na(all_groups[[v]]))),
  Pct_missing = round(
    sapply(desc_vars, function(v) mean(is.na(all_groups[[v]]))) * 100, 2)
)
print(as.data.frame(na_tbl), row.names = FALSE)

# 3.6  Outlier counts 

outlier_tbl <- data.frame(
  Variable     = desc_vars,
  N_outliers   = sapply(desc_vars, function(v) {
    x  <- all_groups[[v]]
    q1 <- quantile(x, 0.25, na.rm = TRUE)
    q3 <- quantile(x, 0.75, na.rm = TRUE)
    sum(x < q1 - 1.5 * (q3 - q1) | x > q3 + 1.5 * (q3 - q1), na.rm = TRUE)
  }),
  Pct_outliers = round(
    sapply(desc_vars, function(v) {
      x  <- all_groups[[v]]
      q1 <- quantile(x, 0.25, na.rm = TRUE)
      q3 <- quantile(x, 0.75, na.rm = TRUE)
      mean(x < q1 - 1.5 * (q3 - q1) | x > q3 + 1.5 * (q3 - q1),
           na.rm = TRUE) * 100
    }), 2)
)
print(as.data.frame(outlier_tbl), row.names = FALSE)

# 3.7  Mean ± SD Valence & Energy by group 

mean_sd_tbl <- all_groups %>%
  group_by(Group) %>%
  summarise(
    N       = n(),
    Valence = paste0(round(mean(valence), 3), " \u00b1 ",
                     round(sd(valence), 3)),
    Energy  = paste0(round(mean(energy),  3), " \u00b1 ",
                     round(sd(energy),  3)),
    .groups = "drop"
  )
print(as.data.frame(mean_sd_tbl), row.names = FALSE)

# 3.8 Pairwise correlations per group 

cor_features <- c("danceability", "loudness", "speechiness",
                  "acousticness", "instrumentalness", "liveness",
                  "tempo", "popularity")

for (grp in c("G1: Rhythmic", "G2: Classical", "G3: Hard Rock")) {
  gdf <- all_groups %>% filter(Group == grp)
  cat("── Group:", grp, "──\n")
  cor_tbl <- data.frame(
    Feature          = cor_features,
    Cor_with_valence = round(sapply(cor_features, function(f)
      cor(gdf[[f]], gdf$valence, use = "complete.obs")), 4),
    Cor_with_energy  = round(sapply(cor_features, function(f)
      cor(gdf[[f]], gdf$energy,  use = "complete.obs")), 4)
  )
  print(as.data.frame(cor_tbl), row.names = FALSE)
  cat("\n")
}

# 3.9  Kruskal Wallis tests 

kw_valence <- kruskal.test(valence ~ Group, data = all_groups)
kw_energy  <- kruskal.test(energy  ~ Group, data = all_groups)

cat("Valence: H =", round(kw_valence$statistic, 2),
    " | df =", kw_valence$parameter,
    " | p =", format.pval(kw_valence$p.value, digits = 4), "\n")
cat("Energy : H =", round(kw_energy$statistic, 2),
    " | df =", kw_energy$parameter,
    " | p =", format.pval(kw_energy$p.value, digits = 4), "\n")



#  4. EXPLORATORY DATA ANALYSIS


GROUP_COLORS <- c("G1: Rhythmic"  = "#4e9af1",
                  "G2: Classical" = "#e85d04",
                  "G3: Hard Rock" = "#06d6a0")

# Plot 1 Valence distribution
p1 <- ggplot(all_groups, aes(x = valence, fill = Group)) +
  geom_histogram(bins = 40, color = "white", alpha = 0.85) +
  facet_wrap(~Group, ncol = 1) +
  scale_fill_manual(values = GROUP_COLORS) +
  labs(title    = "Valence Distribution by Genre Group",
       subtitle = "Higher valence = more positive / happy songs",
       x = "Valence", y = "Count") +
  theme_bw(base_size = 13) + theme(legend.position = "none")
print(p1)

# Plot 2 Energy distribution
p2 <- ggplot(all_groups, aes(x = energy, fill = Group)) +
  geom_histogram(bins = 40, color = "white", alpha = 0.85) +
  facet_wrap(~Group, ncol = 1) +
  scale_fill_manual(values = GROUP_COLORS) +
  labs(title    = "Energy Distribution by Genre Group",
       subtitle = "Higher energy = louder, faster, more intense songs",
       x = "Energy", y = "Count") +
  theme_bw(base_size = 13) + theme(legend.position = "none")
print(p2)

# Plot 3 Circumplex scatter (Valence vs Energy)
p3 <- ggplot(all_groups, aes(x = valence, y = energy, color = track_genre)) +
  geom_point(alpha = 0.15, size = 0.8) +
  geom_density_2d(linewidth = 0.4, alpha = 0.8) +
  facet_wrap(~Group) +
  labs(title    = "Circumplex Model of Affect: Valence vs Energy",
       subtitle = "Russell (1980) — each point is one song",
       x = "Valence (positive \u2194 negative)",
       y = "Energy (calm \u2194 intense)",
       color = "Genre") +
  theme_bw(base_size = 12) + theme(legend.position = "bottom")
print(p3)

# Plot 4 Boxplots
p4a <- ggplot(all_groups, aes(x = Group, y = valence, fill = Group)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.2) +
  scale_fill_manual(values = GROUP_COLORS) +
  labs(title = "Valence by Genre Group", x = NULL, y = "Valence") +
  theme_bw(base_size = 12) + theme(legend.position = "none")

p4b <- ggplot(all_groups, aes(x = Group, y = energy, fill = Group)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.2) +
  scale_fill_manual(values = GROUP_COLORS) +
  labs(title = "Energy by Genre Group", x = NULL, y = "Energy") +
  theme_bw(base_size = 12) + theme(legend.position = "none")
print(p4a + p4b)

# Plot 5 Violin plots 
p5a <- ggplot(all_groups, aes(x = Group, y = valence, fill = Group)) +
  geom_violin(alpha = 0.7) +
  geom_boxplot(width = 0.1, fill = "white", alpha = 0.8) +
  scale_fill_manual(values = GROUP_COLORS) +
  labs(title = paste0("Valence — KW H=",
                      round(kw_valence$statistic, 1), ", p<0.001"),
       x = NULL, y = "Valence") +
  theme_bw(base_size = 12) + theme(legend.position = "none")

p5b <- ggplot(all_groups, aes(x = Group, y = energy, fill = Group)) +
  geom_violin(alpha = 0.7) +
  geom_boxplot(width = 0.1, fill = "white", alpha = 0.8) +
  scale_fill_manual(values = GROUP_COLORS) +
  labs(title = paste0("Energy — KW H=",
                      round(kw_energy$statistic, 1), ", p<0.001"),
       x = NULL, y = "Energy") +
  theme_bw(base_size = 12) + theme(legend.position = "none")
print(p5a + p5b)

# Plot 6 Correlation matrices per group
num_vars <- c("popularity", "danceability", "energy", "loudness",
              "speechiness", "acousticness", "instrumentalness",
              "liveness", "valence", "tempo")

cat("\nCorrelation Matrix — G1: Rhythmic\n")
corrplot(cor(g1_data %>% select(all_of(num_vars)) %>% drop_na()),
         method = "color", type = "upper", tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.7,
         title = "Correlations — G1: Rhythmic", mar = c(0, 0, 2, 0))

cat("\nCorrelation Matrix — G2: Classical\n")
corrplot(cor(g2_data %>% select(all_of(num_vars)) %>% drop_na()),
         method = "color", type = "upper", tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.7,
         title = "Correlations — G2: Classical", mar = c(0, 0, 2, 0))

cat("\nCorrelation Matrix — G3: Hard Rock\n")
corrplot(cor(g3_data %>% select(all_of(num_vars)) %>% drop_na()),
         method = "color", type = "upper", tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.7,
         title = "Correlations — G3: Hard Rock", mar = c(0, 0, 2, 0))

# Plot 7 Average normalised feature values by group
feature_means <- all_groups %>%
  group_by(Group) %>%
  summarise(
    Danceability     = mean(danceability),
    Energy           = mean(energy),
    Loudness_norm    = (mean(loudness) - min(all_groups$loudness)) /
                       (max(all_groups$loudness) - min(all_groups$loudness)),
    Speechiness      = mean(speechiness),
    Acousticness     = mean(acousticness),
    Instrumentalness = mean(instrumentalness),
    Liveness         = mean(liveness),
    Valence          = mean(valence),
    Tempo_norm       = (mean(tempo) - min(all_groups$tempo)) /
                       (max(all_groups$tempo) - min(all_groups$tempo))
  ) %>%
  pivot_longer(-Group, names_to = "Feature", values_to = "Mean")

p7 <- ggplot(feature_means, aes(x = Feature, y = Mean, fill = Group)) +
  geom_col(position = "dodge", alpha = 0.85) +
  scale_fill_manual(values = GROUP_COLORS) +
  labs(title    = "Average Feature Values by Genre Group",
       subtitle = "Loudness and Tempo normalised to [0,1] for comparison",
       x = NULL, y = "Mean Value") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
print(p7)

# Plot 8 Within-group genre comparisons (violin)
p8_g1v <- ggplot(g1_data,
                 aes(x = track_genre, y = valence, fill = track_genre)) +
  geom_violin(alpha = 0.7) + geom_boxplot(width = 0.1, fill = "white") +
  scale_fill_brewer(palette = "Blues") +
  labs(title = "G1 Rhythmic — Valence", x = NULL, y = "Valence") +
  theme_bw(base_size = 11) + theme(legend.position = "none")

p8_g1e <- ggplot(g1_data,
                 aes(x = track_genre, y = energy, fill = track_genre)) +
  geom_violin(alpha = 0.7) + geom_boxplot(width = 0.1, fill = "white") +
  scale_fill_brewer(palette = "Blues") +
  labs(title = "G1 Rhythmic — Energy", x = NULL, y = "Energy") +
  theme_bw(base_size = 11) + theme(legend.position = "none")

p8_g2v <- ggplot(g2_data,
                 aes(x = track_genre, y = valence, fill = track_genre)) +
  geom_violin(alpha = 0.7) + geom_boxplot(width = 0.1, fill = "white") +
  scale_fill_brewer(palette = "Oranges") +
  labs(title = "G2 Classical — Valence", x = NULL, y = "Valence") +
  theme_bw(base_size = 11) + theme(legend.position = "none")

p8_g2e <- ggplot(g2_data,
                 aes(x = track_genre, y = energy, fill = track_genre)) +
  geom_violin(alpha = 0.7) + geom_boxplot(width = 0.1, fill = "white") +
  scale_fill_brewer(palette = "Oranges") +
  labs(title = "G2 Classical — Energy", x = NULL, y = "Energy") +
  theme_bw(base_size = 11) + theme(legend.position = "none")

p8_g3v <- ggplot(g3_data,
                 aes(x = track_genre, y = valence, fill = track_genre)) +
  geom_violin(alpha = 0.7) + geom_boxplot(width = 0.1, fill = "white") +
  scale_fill_brewer(palette = "Greens") +
  labs(title = "G3 Hard Rock — Valence", x = NULL, y = "Valence") +
  theme_bw(base_size = 11) + theme(legend.position = "none")

p8_g3e <- ggplot(g3_data,
                 aes(x = track_genre, y = energy, fill = track_genre)) +
  geom_violin(alpha = 0.7) + geom_boxplot(width = 0.1, fill = "white") +
  scale_fill_brewer(palette = "Greens") +
  labs(title = "G3 Hard Rock — Energy", x = NULL, y = "Energy") +
  theme_bw(base_size = 11) + theme(legend.position = "none")

print(
  (p8_g1v + p8_g1e) / (p8_g2v + p8_g2e) / (p8_g3v + p8_g3e) +
  plot_annotation(title = "Within Group Genre Comparisons Valence & Energy")
)

# Plot 9 Key feature scatter plots
p9a <- ggplot(all_groups, aes(x = danceability, y = valence, color = Group)) +
  geom_point(alpha = 0.15, size = 0.6) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1.2) +
  scale_color_manual(values = GROUP_COLORS) +
  labs(title    = "Danceability vs Valence by Group",
       x = "Danceability", y = "Valence") +
  theme_bw(base_size = 12)

p9b <- ggplot(all_groups, aes(x = loudness, y = energy, color = Group)) +
  geom_point(alpha = 0.15, size = 0.6) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1.2) +
  scale_color_manual(values = GROUP_COLORS) +
  labs(title    = "Loudness vs Energy by Group",
       ,
       x = "Loudness (dB)", y = "Energy") +
  theme_bw(base_size = 12)

p9c <- ggplot(all_groups, aes(x = acousticness, y = valence, color = Group)) +
  geom_point(alpha = 0.15, size = 0.6) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1.2) +
  scale_color_manual(values = GROUP_COLORS) +
  labs(title    = "Acousticness vs Valence by Group",,
       x = "Acousticness", y = "Valence") +
  theme_bw(base_size = 12)

print(p9a / p9b / p9c +
      plot_annotation(title = "Key Feature Relationships by Group"))



#  5. Calculating FUNCTIONS

# RMSE, MAE, R²
get_metrics <- function(predicted, actual) {
  errors <- actual - predicted
  data.frame(
    RMSE = round(sqrt(mean(errors^2)),                                4),
    MAE  = round(mean(abs(errors)),                                   4),
    R2   = round(1 - sum(errors^2) / sum((actual - mean(actual))^2), 4)
  )
}

# 75/25 stratified split + z-score scaling
split_data <- function(df, target) {
  features <- c(target, "popularity", "duration_ms", "explicit",
                "danceability", "loudness", "speechiness",
                "acousticness", "instrumentalness", "liveness", "tempo")
  df <- df %>% select(all_of(features)) %>% drop_na()

  idx   <- createDataPartition(df[[target]], p = 0.75, list = FALSE)
  train <- df[idx, ];  test <- df[-idx, ]

  num_cols   <- setdiff(names(train), target)
  means      <- sapply(train[num_cols], mean)
  sds        <- sapply(train[num_cols], sd);  sds[sds == 0] <- 1

  train_sc <- train;  test_sc <- test
  train_sc[num_cols] <- sweep(sweep(train[num_cols], 2, means, "-"), 2, sds, "/")
  test_sc[num_cols]  <- sweep(sweep(test[num_cols],  2, means, "-"), 2, sds, "/")

  list(train    = train,    test    = test,
       train_sc = train_sc, test_sc = test_sc,
       y_train  = train[[target]],
       y_test   = test[[target]])
}

# Extract one tree from a Random Forest (max depth = 4)
extract_tree_layout <- function(rf_model, tree_num = 1, max_depth = 4) {
  raw <- as.data.frame(getTree(rf_model, k = tree_num, labelVar = TRUE))
  raw$node_id <- seq_len(nrow(raw))

  info  <- data.frame(node_id = integer(), depth = integer(),
                      slot = integer(), parent_id = integer(),
                      side = character(), stringsAsFactors = FALSE)
  queue <- data.frame(node_id = 1L, depth = 0L, slot = 1L,
                      parent_id = NA_integer_, side = "",
                      stringsAsFactors = FALSE)
  slot_counter <- integer(20);  slot_counter[1] <- 1L

  while (nrow(queue) > 0) {
    cur   <- queue[1, ];  queue <- queue[-1, ]
    nd    <- cur$node_id;  d <- cur$depth
    if (d > max_depth) next
    info <- rbind(info, cur)

    row <- raw[nd, ]
    l   <- as.integer(row[["left daughter"]])
    r   <- as.integer(row[["right daughter"]])

    if (!is.na(l) && l > 0 && (d + 1) <= max_depth) {
      slot_counter[d + 2] <- slot_counter[d + 2] + 1L
      queue <- rbind(queue, data.frame(node_id = l, depth = d + 1L,
                                       slot = slot_counter[d + 2],
                                       parent_id = nd, side = "left",
                                       stringsAsFactors = FALSE))
    }
    if (!is.na(r) && r > 0 && (d + 1) <= max_depth) {
      slot_counter[d + 2] <- slot_counter[d + 2] + 1L
      queue <- rbind(queue, data.frame(node_id = r, depth = d + 1L,
                                       slot = slot_counter[d + 2],
                                       parent_id = nd, side = "right",
                                       stringsAsFactors = FALSE))
    }
  }

  info <- info %>%
    group_by(depth) %>%
    mutate(n_at_depth = n(),
           x = (slot - 1) / max(n_at_depth - 1, 1) * 10) %>%
    ungroup() %>%
    mutate(y = -depth) %>%
    left_join(raw %>% select(node_id,
                              split_var   = `split var`,
                              split_point = `split point`,
                              status      = status,
                              prediction  = prediction),
              by = "node_id") %>%
    mutate(
      is_leaf = (status == -1),
      label   = ifelse(is_leaf,
                       paste0("pred\n", round(prediction, 3)),
                       paste0(split_var, "\n<= ", round(split_point, 3)))
    )

  edges <- info %>%
    filter(!is.na(parent_id)) %>%
    left_join(info %>% select(node_id, px = x, py = y),
              by = c("parent_id" = "node_id")) %>%
    rename(cx = x, cy = y) %>%
    select(cx, cy, px, py, side, split_var, split_point)

  list(nodes = info, edges = edges)
}

# Render one decision tree as a ggplot
draw_tree <- function(rf_model, title_text,
                      tree_num = 1, max_depth = 4, target_name = "value") {
  tl    <- extract_tree_layout(rf_model, tree_num, max_depth)
  nodes <- tl$nodes
  edges <- tl$edges %>%
    mutate(mx     = (cx + px) / 2,
           my     = (cy + py) / 2,
           elabel = ifelse(side == "left",
                           paste0("<= ", round(split_point, 2)),
                           paste0(">  ", round(split_point, 2))))

  ggplot() +
    geom_segment(data = edges,
                 aes(x = px, y = py, xend = cx, yend = cy),
                 colour = "grey50", linewidth = 0.7) +
    geom_label(data = edges,
               aes(x = mx, y = my, label = elabel),
               size = 2.4, fill = "white", colour = "grey30",
               label.size = 0.2,
               label.padding = unit(0.15, "lines")) +
    geom_point(data = nodes %>% filter(is_leaf),
               aes(x = x, y = y, fill = prediction),
               shape = 21, size = 12, colour = "white", stroke = 1.2) +
    scale_fill_gradient2(low = "#d73027", mid = "#ffffbf", high = "#1a9850",
                         midpoint = 0.5, name = target_name,
                         limits = c(0, 1)) +
    geom_point(data = nodes %>% filter(!is_leaf),
               aes(x = x, y = y),
               shape = 22, size = 14,
               fill = "#4e9af1", colour = "white", stroke = 1.2) +
    geom_text(data = nodes,
              aes(x = x, y = y, label = label),
              size = 2.6, fontface = "bold", colour = "white",
              lineheight = 0.85) +
    scale_y_continuous(breaks = 0:(-max_depth),
                       labels = paste("Depth", 0:max_depth)) +
    labs(title    = title_text,
         subtitle = paste0("Tree #", tree_num,
                           " extracted from Random Forest  |  ",
                           "Blue square = split node  |  ",
                           "Coloured circle = predicted ", target_name),
         x = NULL, y = NULL) +
    theme_minimal(base_size = 11) +
    theme(panel.grid    = element_blank(),
          axis.text.x   = element_blank(),
          axis.ticks.x  = element_blank(),
          axis.text.y   = element_text(colour = "grey50", size = 9),
          plot.title    = element_text(face = "bold", size = 13),
          legend.position = "right")
}


#  6. MODELLING


targets <- c("valence", "energy")

MODEL_COLORS <- c(G1_Rhythmic  = "#4e9af1",
                  G2_Classical = "#e85d04",
                  G3_HardRock  = "#06d6a0")

group_labels <- c(
  G1_Rhythmic  = "G1: Rhythmic (salsa / latin / reggaeton)",
  G2_Classical = "G2: Classical (opera / classical / piano)",
  G3_HardRock  = "G3: Hard Rock (punk-rock / heavy-metal / punk)"
)

# Linear Model Predictor Sets
energy_predictor_sets <- list(
  "M1: Loudness only"                              = c("loudness"),
  "M2: Loudness + Acousticness + Instrumentalness" = c("loudness",
                                                        "acousticness",
                                                        "instrumentalness"),
  "M3: Tempo + Danceability + Liveness"            = c("tempo",
                                                        "danceability",
                                                        "liveness"),
  "M4: All audio features"                         = c("loudness",
                                                        "acousticness",
                                                        "instrumentalness",
                                                        "tempo",
                                                        "danceability",
                                                        "liveness",
                                                        "speechiness",
                                                        "valence"),
  "M5: Audio + Popularity + Duration"              = c("loudness",
                                                        "acousticness",
                                                        "instrumentalness",
                                                        "tempo",
                                                        "danceability",
                                                        "liveness",
                                                        "speechiness",
                                                        "valence",
                                                        "popularity",
                                                        "duration_ms")
)

valence_predictor_sets <- list(
  "M1: Danceability only"                    = c("danceability"),
  "M2: Danceability + Energy"                = c("danceability", "energy"),
  "M3: Danceability + Energy + Acousticness" = c("danceability", "energy",
                                                  "acousticness"),
  "M4: All audio features"                   = c("danceability", "energy",
                                                  "loudness", "acousticness",
                                                  "instrumentalness",
                                                  "liveness", "speechiness",
                                                  "tempo"),
  "M5: Audio + Popularity + Duration"        = c("danceability", "energy",
                                                  "loudness", "acousticness",
                                                  "instrumentalness",
                                                  "liveness", "speechiness",
                                                  "tempo", "popularity",
                                                  "duration_ms")
)

alpha_grid <- seq(0, 1, by = 0.1)

# Storing
ols_results_all <- list()
en_results_all  <- list()
master_results  <- list()
all_residuals   <- list()
rf_models       <- list()
xgb_models      <- list()



# Group × Target

for (gname in names(groups)) {
  gdata <- groups[[gname]]

  for (tgt in targets) {

    set.seed(42)
    cat("  Group:", gname, " | Target:", tgt, "\n")

    sp        <- split_data(gdata, tgt)
    key       <- paste(gname, tgt, sep = "_")
    pred_sets <- if (tgt == "energy") energy_predictor_sets else
                                      valence_predictor_sets

    loop_results <- list()

    cat("\n 5.1  Linear Regression \n")

    for (mname in names(pred_sets)) {
      preds    <- pred_sets[[mname]]
      df_slice <- gdata %>% select(all_of(c(tgt, preds))) %>% drop_na()
      idx      <- createDataPartition(df_slice[[tgt]], p = 0.75, list = FALSE)
      tr_ols   <- df_slice[ idx, ];  te_ols <- df_slice[-idx, ]

      fit_ols  <- lm(as.formula(paste(tgt, "~",
                                      paste(preds, collapse = " + "))),
                     data = tr_ols)
      pred_ols <- pmax(0, pmin(1, predict(fit_ols, te_ols)))
      m        <- get_metrics(pred_ols, te_ols[[tgt]])

      ols_results_all[[paste(gname, tgt, mname, sep = " | ")]] <- data.frame(
        Group = gname, Target = tgt, Model = "OLS",
        PredictorSet = mname, N_predictors = length(preds),
        RMSE = m$RMSE, MAE = m$MAE, R2 = m$R2,
        stringsAsFactors = FALSE
      )
      cat("    [", mname, "]  R2 =", m$R2, "| RMSE =", m$RMSE, "\n")
    }

    # Train LR 
    lm_fit_sp  <- lm(as.formula(paste(tgt, "~ .")), data = sp$train)
    lm_pred_sp <- pmax(0, pmin(1, predict(lm_fit_sp, sp$test)))
    loop_results[["LinearRegression"]] <- get_metrics(lm_pred_sp, sp$y_test)


    cat("\n 5.2  ElasticNet 5-fold CV) \n")

    en_preds <- pred_sets[["M4: All audio features"]]
    df_en    <- gdata %>% select(all_of(c(tgt, en_preds))) %>% drop_na()
    idx_en   <- createDataPartition(df_en[[tgt]], p = 0.75, list = FALSE)
    tr_en    <- df_en[ idx_en, ];  te_en <- df_en[-idx_en, ]

    X_tr <- as.matrix(tr_en[, en_preds]);  X_te <- as.matrix(te_en[, en_preds])
    y_tr <- tr_en[[tgt]];                  y_te <- te_en[[tgt]]

    cm <- colMeans(X_tr)
    cs <- apply(X_tr, 2, sd);  cs[cs == 0] <- 1
    X_tr_sc <- sweep(sweep(X_tr, 2, cm, "-"), 2, cs, "/")
    X_te_sc <- sweep(sweep(X_te, 2, cm, "-"), 2, cs, "/")

    best_en_r2   <- -Inf
    best_en_pred <- NULL

    for (a in alpha_grid) {
      en_cv   <- cv.glmnet(X_tr_sc, y_tr, alpha = a,
                           nfolds = 5, standardize = FALSE)
      en_pred <- pmax(0, pmin(1,
                              predict(en_cv, X_te_sc, s = "lambda.min")[, 1]))
      m <- get_metrics(en_pred, y_te)

      en_results_all[[paste(gname, tgt,
                            paste0("a=", round(a, 1)), sep = "|")]] <-
        data.frame(Group = gname, Target = tgt, Alpha = a,
                   Lambda = round(en_cv[["lambda.min"]], 6),
                   RMSE = m$RMSE, MAE = m$MAE, R2 = m$R2,
                   stringsAsFactors = FALSE)

      cat("    alpha =", round(a, 1),
          "| lambda =", round(en_cv[["lambda.min"]], 5),
          "| R2 =", m$R2, "\n")

      if (m$R2 > best_en_r2) { best_en_r2 <- m$R2; best_en_pred <- en_pred }
    }
    loop_results[["ElasticNet"]] <- get_metrics(best_en_pred, y_te)

    # EN on standard split 
    en_fit_sp  <- cv.glmnet(as.matrix(sp$train_sc[, -1]),
                             sp$y_train, alpha = 0.5, nfolds = 5)
    en_pred_sp <- pmax(0, pmin(1,
                               predict(en_fit_sp,
                                       as.matrix(sp$test_sc[, -1]),
                                       s = "lambda.min")[, 1]))


    cat("\n 5.3 KNN {5, 11, 21}, 5-fold CV\n")

    knn_fit  <- train(x = sp$train_sc[, -1], y = sp$y_train,
                      method    = "kknn",
                      tuneGrid  = expand.grid(kmax     = c(5, 11, 21),
                                              distance = 2,
                                              kernel   = "optimal"),
                      trControl = trainControl(method = "cv", number = 5))
    knn_pred <- predict(knn_fit, sp$test_sc[, -1])
    loop_results[["KNN"]] <- get_metrics(knn_pred, sp$y_test)
    cat("    Best kmax =", knn_fit$bestTune$kmax,
        "| R2 =", loop_results$KNN$R2, "\n")


    cat("\n KNN + PCA (95% variance retained) \n")

    pca_fit <- prcomp(sp$train_sc[, -1], center = FALSE, scale. = FALSE)
    var_exp <- cumsum(pca_fit$sdev^2) / sum(pca_fit$sdev^2)
    n_comp  <- max(2, which(var_exp >= 0.95)[1])
    tr_pca  <- as.data.frame(predict(pca_fit, sp$train_sc[, -1])[, 1:n_comp])
    te_pca  <- as.data.frame(predict(pca_fit, sp$test_sc[, -1])[, 1:n_comp])

    knn_pca_fit  <- train(x = tr_pca, y = sp$y_train,
                          method    = "kknn",
                          tuneGrid  = expand.grid(kmax     = c(5, 11, 21),
                                                  distance = 2,
                                                  kernel   = "optimal"),
                          trControl = trainControl(method = "cv", number = 5))
    knn_pca_pred <- predict(knn_pca_fit, te_pca)
    loop_results[["KNN_PCA"]] <- get_metrics(knn_pca_pred, sp$y_test)
    cat("    PCA kept", n_comp, "components | Best kmax =",
        knn_pca_fit$bestTune$kmax, "\n")
    cat("    KNN R2 =", loop_results$KNN$R2,
        "| KNN+PCA R2 =", loop_results$KNN_PCA$R2, "\n")


    cat("\n 5.4 Random Forest with mtry {2,4,6,8}, 5-fold CV and ntree=300) \n")

    rf_cv <- train(
      x         = sp$train[, setdiff(names(sp$train), tgt)],
      y         = sp$y_train,
      method    = "rf",
      ntree     = 300,
      tuneGrid  = expand.grid(mtry = c(2, 4, 6, 8)),
      trControl = trainControl(method = "cv", number = 5,
                               verboseIter = FALSE),
      importance = TRUE
    )
    best_mtry <- rf_cv$bestTune$mtry
    cat("    Best mtry =", best_mtry, "\n")

    rf_fit  <- randomForest(as.formula(paste(tgt, "~ .")),
                            data = sp$train, ntree = 300,
                            mtry = best_mtry, importance = TRUE)
    rf_pred <- pmax(0, pmin(1, predict(rf_fit, sp$test)))
    loop_results[["RandomForest"]] <- get_metrics(rf_pred, sp$y_test)
    rf_models[[key]] <- rf_fit
    cat("    RF R2 =", loop_results$RandomForest$R2,
        "| RMSE =", loop_results$RandomForest$RMSE, "\n")

    # Feature importance
    imp_df <- as.data.frame(importance(rf_fit)) %>%
      rownames_to_column("Feature") %>%
      rename(PctIncMSE = `%IncMSE`) %>%
      arrange(desc(PctIncMSE))

    cat("\n    Feature Importance \n")
    print(as.data.frame(imp_df %>% select(Feature, PctIncMSE)),
          row.names = FALSE)

    p_imp <- ggplot(imp_df %>% head(10),
                    aes(x = reorder(Feature, PctIncMSE),
                        y = PctIncMSE, fill = PctIncMSE)) +
      geom_col(alpha = 0.85) + coord_flip() +
      scale_fill_viridis_c(option = "C", direction = -1) +
      labs(title    = paste0("RF Feature Importance — ",
                             gname, " | ", toupper(tgt)),
           x = NULL, y = "% Increase in MSE") +
      theme_bw(base_size = 12) + theme(legend.position = "none")
    print(p_imp)

    # Decision tree visualisation (Tree #1, max depth 4)
    cat("\n    Decision Tree #1 \n")
    p_tree <- draw_tree(
      rf_model    = rf_fit,
      title_text  = paste0("Decision Tree — ", group_labels[gname],
                           "  |  Target: ", toupper(tgt)),
      tree_num = 1, max_depth = 4, target_name = tgt
    )
    print(p_tree)


    cat("\n 5.5 XGBoost nrounds via 5-fold CV \n")

    dtrain <- xgb.DMatrix(as.matrix(sp$train_sc[, -1]), label = sp$y_train)
    dtest  <- xgb.DMatrix(as.matrix(sp$test_sc[, -1]),  label = sp$y_test)

    xgb_params <- list(
      objective        = "reg:squarederror",
      max_depth        = 6,
      eta              = 0.05,
      subsample        = 0.8,
      colsample_bytree = 0.8,
      seed             = 42
    )

    xgb_cv    <- xgb.cv(params = xgb_params, data = dtrain,
                        nrounds = 300, nfold = 5,
                        early_stopping_rounds = 20, verbose = FALSE)
    best_rounds <- xgb_cv$best_iteration
    if (is.null(best_rounds) || best_rounds < 1) best_rounds <- 100

    xgb_fit  <- xgb.train(params = xgb_params, data = dtrain,
                           nrounds = best_rounds, verbose = FALSE)
    xgb_pred <- pmax(0, pmin(1, predict(xgb_fit, dtest)))
    loop_results[["XGBoost"]] <- get_metrics(xgb_pred, sp$y_test)
    xgb_models[[key]] <- list(model   = xgb_fit,
                               dtest   = dtest,
                               X_train = as.matrix(sp$train_sc[, -1]))
    cat("    Best nrounds =", best_rounds,
        "| R2 =", loop_results$XGBoost$R2,
        "| RMSE =", loop_results$XGBoost$RMSE, "\n")

    
    all_residuals[[key]] <- data.frame(
      Group     = gname,  Target    = tgt,
      Actual    = sp$y_test, Predicted = xgb_pred,
      Residual  = sp$y_test - xgb_pred
    )


    cat("\n 5.6 Stacked Ensemble linear meta-learner\n")

    n_train  <- nrow(sp$train)
    fold_ids <- sample(rep(1:5, length.out = n_train))

    oof_lm  <- numeric(n_train);  oof_en  <- numeric(n_train)
    oof_knn <- numeric(n_train);  oof_rf  <- numeric(n_train)
    oof_xgb <- numeric(n_train)

    for (fold in 1:5) {
      val_idx <- which(fold_ids == fold)
      trn_idx <- which(fold_ids != fold)

      f_tr    <- sp$train[trn_idx, ];     f_te    <- sp$train[val_idx, ]
      f_tr_sc <- sp$train_sc[trn_idx, ];  f_te_sc <- sp$train_sc[val_idx, ]

      # Linear Regression
      f_lm            <- lm(as.formula(paste(tgt, "~ .")), data = f_tr)
      oof_lm[val_idx] <- pmax(0, pmin(1, predict(f_lm, f_te)))

      # ElasticNet (alpha=0.5)
      f_en <- cv.glmnet(as.matrix(f_tr_sc[, -1]),
                        f_tr_sc[[tgt]], alpha = 0.5, nfolds = 5)
      oof_en[val_idx] <- pmax(0, pmin(1, as.numeric(
        predict(f_en, as.matrix(f_te_sc[, -1]), s = "lambda.min"))))

      # KNN 
      f_knn <- train(x = f_tr_sc[, -1], y = f_tr_sc[[tgt]],
                     method    = "kknn",
                     tuneGrid  = expand.grid(kmax = 11, distance = 2,
                                             kernel = "optimal"),
                     trControl = trainControl(method = "none"))
      oof_knn[val_idx] <- pmax(0, pmin(1, predict(f_knn, f_te_sc[, -1])))

      # Random Forest 
      f_rf <- randomForest(as.formula(paste(tgt, "~ .")),
                           data = f_tr, ntree = 150, mtry = best_mtry)
      oof_rf[val_idx] <- pmax(0, pmin(1, predict(f_rf, f_te)))

      # XGBoost
      f_xgb <- xgb.train(
        params  = xgb_params,
        data    = xgb.DMatrix(as.matrix(f_tr_sc[, -1]),
                              label = f_tr_sc[[tgt]]),
        nrounds = best_rounds, verbose = FALSE
      )
      oof_xgb[val_idx] <- pmax(0, pmin(1,
        predict(f_xgb,
                xgb.DMatrix(as.matrix(f_te_sc[, -1])))))
    }

    # Level-2 meta-learner
    meta_tr  <- data.frame(lm = oof_lm, en = oof_en, knn = oof_knn,
                           rf = oof_rf, xgb = oof_xgb, y = sp$y_train)
    meta_fit <- lm(y ~ lm + en + knn + rf + xgb - 1, data = meta_tr)

    meta_te    <- data.frame(lm  = lm_pred_sp, en  = en_pred_sp,
                             knn = knn_pred,   rf  = rf_pred,
                             xgb = xgb_pred)
    stack_pred <- pmax(0, pmin(1, predict(meta_fit, meta_te)))
    loop_results[["StackedEnsemble"]] <- get_metrics(stack_pred, sp$y_test)

    cat("\n    Meta-learner coefficients \n")
    print(round(coef(meta_fit), 4))
    cat("    Stacked R2 =", loop_results$StackedEnsemble$R2,
        "| XGBoost R2 =",   loop_results$XGBoost$R2, "\n")

    
    res_df <- bind_rows(loop_results, .id = "Model") %>%
      mutate(Group = gname, Target = tgt)
    master_results[[key]] <- res_df
  }
}


#  7. RESULTS TABLES


#  7.1  Linear Model Comparision



lm_master <- bind_rows(ols_results_all)

cat("Linear Model Results \n")
print(as.data.frame(lm_master %>% arrange(Group, Target, PredictorSet)),
      row.names = FALSE)

cat("\n── Best Predictor per Group and Target \n")
print(as.data.frame(
  lm_master %>%
    group_by(Group, Target) %>% slice_max(R2, n = 1) %>%
    select(Group, Target, PredictorSet, N_predictors, RMSE, R2)
), row.names = FALSE)

pOLS_1 <- ggplot(lm_master,
                 aes(x = reorder(PredictorSet, R2), y = R2, fill = Group)) +
  geom_col(position = "dodge", alpha = 0.85) +
  geom_text(aes(label = R2), position = position_dodge(0.9),
            hjust = -0.1, size = 2.6, fontface = "bold") +
  facet_wrap(~Target, ncol = 1) + coord_flip() +
  scale_fill_manual(values = MODEL_COLORS) + ylim(0, 1.15) +
  labs(title = "R\u00b2 by Predictor Set",
       x = NULL, y = "Test R\u00b2") +
  theme_bw(base_size = 12) + theme(legend.position = "bottom")
print(pOLS_1)

pOLS_2 <- lm_master %>%
  mutate(SetNum = as.integer(gsub("M(\\d+):.*", "\\1", PredictorSet))) %>%
  ggplot(aes(x = SetNum, y = R2, color = Group, group = Group)) +
  geom_line(linewidth = 1.2) + geom_point(size = 3.5) +
  facet_wrap(~Target, ncol = 1) +
  scale_x_continuous(breaks = 1:5,
    labels = c("M1\n(1)", "M2\n(3)", "M3\n(3)", "M4\n(8)", "M5\n(10)")) +
  scale_color_manual(values = MODEL_COLORS) +
  labs(title    = "R\u00b2 as More Predictors Are Added",
       x = "Predictor Set (simple \u2192 complex)", y = "Test R\u00b2") +
  theme_bw(base_size = 13)
print(pOLS_2)

# 7.2  ElasticNet alpha tuning 

en_master <- bind_rows(en_results_all)

cat("ElasticNet Results \n")
print(as.data.frame(en_master %>% arrange(Group, Target, Alpha)),
      row.names = FALSE)

cat("\n Best ElasticNet per Group and Target\n")
print(as.data.frame(
  en_master %>%
    group_by(Group, Target) %>%
    slice_max(R2, n = 1, with_ties = FALSE) %>%
    select(Group, Target, Alpha, Lambda, RMSE, R2)
), row.names = FALSE)

pEN_1 <- ggplot(en_master,
                aes(x = Alpha, y = R2, color = Group, group = Group)) +
  geom_line(linewidth = 1.2) + geom_point(size = 3) +
  facet_wrap(~Target, ncol = 1) +
  scale_color_manual(values = MODEL_COLORS) +
  scale_x_continuous(breaks = seq(0, 1, 0.1),
    labels = c("0\n(Ridge)", ".1", ".2", ".3", ".4",
               ".5", ".6", ".7", ".8", ".9", "1\n(Lasso)")) +
  labs(title    = "ElasticNet — R\u00b2 Across the Full Alpha Path",
       subtitle = "Lambda selected by 5-fold CV at each alpha",
       x = "Alpha", y = "Test R\u00b2") +
  theme_bw(base_size = 13)
print(pEN_1)

pEN_2 <- ggplot(en_master,
                aes(x = Alpha, y = Lambda, color = Group,
                    linetype = Target,
                    group = interaction(Group, Target))) +
  geom_line(linewidth = 1) + geom_point(size = 2.5) +
  scale_color_manual(values = MODEL_COLORS) +
  scale_linetype_manual(values = c(valence = "solid", energy = "dashed")) +
  scale_y_log10() +
  labs(title    = "ElasticNet Optimal Lambda at Each Alpha",
       x = "Alpha", y = "Best Lambda (log scale)") +
  theme_bw(base_size = 13) + theme(legend.position = "bottom")
print(pEN_2)

# 7.3  All model comparison 

master <- bind_rows(master_results)

print(as.data.frame(master %>% arrange(Group, Target, RMSE)),
      row.names = FALSE)

# 7.4  KNN vs KNN+PCA 

cat("\n7.4  KNN vs KNN+PCA Comparison\n")
print(as.data.frame(
  master %>%
    filter(Model %in% c("KNN", "KNN_PCA")) %>%
    select(Group, Target, Model, RMSE, R2) %>%
    arrange(Group, Target, Model)
), row.names = FALSE)

pKNN <- master %>%
  filter(Model %in% c("KNN", "KNN_PCA")) %>%
  mutate(Model = recode(Model, "KNN_PCA" = "KNN + PCA")) %>%
  ggplot(aes(x = paste(Group, Target, sep = "\n"), y = R2, fill = Model)) +
  geom_col(position = "dodge", alpha = 0.85) +
  geom_text(aes(label = R2), position = position_dodge(0.9),
            vjust = -0.4, size = 3, fontface = "bold") +
  scale_fill_manual(values = c("KNN" = "#4e9af1", "KNN + PCA" = "#f77f00")) +
  ylim(0, 1) +
  labs(title    = "Plain KNN vs KNN + PCA (95% variance retained)",
       x = NULL, y = "R\u00b2") +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom", axis.text.x = element_text(size = 9))
print(pKNN)

# 7.5  Stacked Ensemble vs base models

cat("\n 7.5  Stacked Ensemble vs Base Models\n")
print(as.data.frame(
  master %>%
    filter(Model %in% c("LinearRegression", "ElasticNet", "KNN",
                        "KNN_PCA", "RandomForest", "XGBoost",
                        "StackedEnsemble")) %>%
    select(Group, Target, Model, RMSE, R2) %>%
    arrange(Group, Target, RMSE)
), row.names = FALSE)

pStack <- master %>%
  filter(Model %in% c("LinearRegression", "ElasticNet", "KNN",
                      "KNN_PCA", "RandomForest", "XGBoost",
                      "StackedEnsemble")) %>%
  mutate(IsEnsemble = ifelse(Model == "StackedEnsemble",
                             "Stacked Ensemble", "Base Model")) %>%
  ggplot(aes(x = reorder(Model, R2), y = R2, fill = IsEnsemble)) +
  geom_col(alpha = 0.85) +
  geom_text(aes(label = R2), hjust = -0.15, size = 2.8, fontface = "bold") +
  facet_grid(Group ~ Target) + coord_flip() +
  scale_fill_manual(values = c("Stacked Ensemble" = "#ff6b6b",
                                "Base Model"       = "#4e9af1"),
                    name = NULL) +
  labs(title    = "Stacked Ensemble vs Base Models",
       x = NULL, y = "R\u00b2") +
  theme_bw(base_size = 11) + theme(legend.position = "bottom")
print(pStack)



#  8. MODEL ANALYSIS


# 8.1  decision tree plot 

cat(" 8.1  RANDOM FOREST \n")


tree_plots <- lapply(names(rf_models), function(key) {
  gn <- sub("_(valence|energy)$", "", key)
  tg <- ifelse(grepl("valence$", key), "valence", "energy")
  draw_tree(rf_models[[key]],
            title_text  = paste0(group_labels[gn],
                                 "\nTarget: ", toupper(tg)),
            tree_num = 1, max_depth = 4, target_name = tg)
})
names(tree_plots) <- names(rf_models)

print(
  (tree_plots[["G1_Rhythmic_valence"]]  + tree_plots[["G1_Rhythmic_energy"]]) /
  (tree_plots[["G2_Classical_valence"]] + tree_plots[["G2_Classical_energy"]]) /
  (tree_plots[["G3_HardRock_valence"]]  + tree_plots[["G3_HardRock_energy"]]) +
  plot_annotation(
    title    = "Random Forest — Representative Decision Trees (Tree #1, max depth 4)",
    subtitle = "Left = Valence | Right = Energy | Blue squares = split nodes | Circles = leaf predictions (red=low, green=high)",
    theme    = theme(plot.title    = element_text(face = "bold", size = 14),
                     plot.subtitle = element_text(size = 10, colour = "grey40"))
  )
)

# 8.2 XGBoost residual

cat(" 8.2  XGBOOST RESIDUAL DIAGNOSTICS (Residuals vs Fitted,\n")
cat("      Actual vs Predicted, Q-Q Plot\n")


all_resid_df <- bind_rows(all_residuals)

for (gname in names(groups)) {
  for (tgt in targets) {
    df    <- all_resid_df %>% filter(Group == gname, Target == tgt)
    label <- paste(gname, toupper(tgt), sep = "  ")

    pr <- ggplot(df, aes(x = Predicted, y = Residual)) +
      geom_point(alpha = 0.3, color = "#3a86ff", size = 0.8) +
      geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
      geom_smooth(method = "loess", se = TRUE,
                  color = "darkred", linewidth = 0.8) +
      labs(title = paste("Residuals vs Fitted ", label),
           x = "Predicted", y = "Residual") +
      theme_bw(base_size = 12)

    pa <- ggplot(df, aes(x = Actual, y = Predicted)) +
      geom_point(alpha = 0.3, color = "#06d6a0", size = 0.8) +
      geom_abline(slope = 1, intercept = 0,
                  color = "red", linetype = "dashed", linewidth = 0.9) +
      labs(title = paste("Actual vs Predicted ", label),
           x = "Actual", y = "Predicted") +
      theme_bw(base_size = 12)

    pq <- ggplot(df, aes(sample = Residual)) +
      stat_qq(alpha = 0.4, color = "#8338ec") +
      stat_qq_line(color = "red", linewidth = 0.8) +
      labs(title = paste("Q-Q Plot of Residuals ", label)) +
      theme_bw(base_size = 12)

    print(pr + pa + pq)
  }
}

# 8.3  SHAP feature divergence (XGBoost, valence) 

cat(" 8.3  SHAP FEATURE DIVERGENCE (XGBoost valence models)\n")


shap_rows <- list()
for (sgname in names(groups)) {
  skey <- paste0(sgname, "_valence")
  if (!is.null(xgb_models[[skey]])) {
    sX       <- xgb_models[[skey]]$X_train
    sm       <- xgb_models[[skey]]$model
    contrib  <- predict(sm, sX, predcontrib = TRUE)
    contrib  <- contrib[, colnames(contrib) != "BIAS"]
    mean_abs <- colMeans(abs(contrib))
    shap_rows[[sgname]] <- data.frame(
      feature = names(mean_abs), group = sgname,
      shap    = as.numeric(mean_abs)
    )
  }
}

if (length(shap_rows) > 0) {
  shap_wide <- bind_rows(shap_rows) %>%
    pivot_wider(names_from = group, values_from = shap, values_fill = 0) %>%
    rename(base = feature)

  gcols <- intersect(c("G1_Rhythmic", "G2_Classical", "G3_HardRock"),
                     names(shap_wide))
  shap_wide$max_diff       <- apply(shap_wide[, gcols], 1,
                                    function(r) max(r) - min(r))
  shap_wide$dominant_group <- apply(shap_wide[, gcols], 1,
                                    function(r) gcols[which.max(r)])

  cat("SHAP Table \n")
  print(as.data.frame(shap_wide %>% arrange(desc(max_diff))), row.names = FALSE)

  shap_long <- shap_wide %>%
    select(base, all_of(gcols)) %>%
    pivot_longer(-base, names_to = "Group", values_to = "SHAP_Importance")

  pSHAP_1 <- ggplot(shap_long,
                    aes(x = reorder(base, SHAP_Importance),
                        y = SHAP_Importance, fill = Group)) +
    geom_col(position = "dodge", alpha = 0.85) + coord_flip() +
    scale_fill_manual(values = c(G1_Rhythmic  = "#4e9af1",
                                  G2_Classical = "#e85d04",
                                  G3_HardRock  = "#06d6a0")) +
    labs(title    = "SHAP Feature Importance by Genre Group (Valence)",
         x = NULL, y = "Mean |SHAP|") +
    theme_bw(base_size = 12)
  print(pSHAP_1)

  pSHAP_2 <- ggplot(shap_wide %>% arrange(desc(max_diff)),
                    aes(x = reorder(base, max_diff),
                        y = max_diff, fill = dominant_group)) +
    geom_col(alpha = 0.85) + coord_flip() +
    scale_fill_manual(values = c(G1_Rhythmic  = "#4e9af1",
                                  G2_Classical = "#e85d04",
                                  G3_HardRock  = "#06d6a0")) +
    labs(title    = "SHAP Cross Group Divergence",
         x = NULL, y = "Max SHAP Difference Across Groups",
         fill = "Dominant Group") +
    theme_bw(base_size = 12)
  print(pSHAP_2)
}


cat(" 8.4  BOOTSTRAP 95% CONFIDENCE INTERVALS (XGBoost R\u00b2)\n")
cat("      n_boot = 200 | percentile method\n")

set.seed(42)
n_boot    <- 200
boot_rows <- list()

for (bkey in names(xgb_models)) {
  bgname  <- sub("_(valence|energy)$", "", bkey)
  btgt    <- ifelse(grepl("valence$", bkey), "valence", "energy")
  bresid  <- all_residuals[[bkey]]
  ba      <- bresid$Actual;  bp <- bresid$Predicted

  r2_boot <- replicate(n_boot, {
    idx <- sample(length(ba), replace = TRUE)
    a   <- ba[idx];  p <- bp[idx]
    1 - sum((a - p)^2) / sum((a - mean(a))^2)
  })

  boot_rows[[bkey]] <- data.frame(
    Group  = bgname, Target = btgt,
    R2     = round(median(r2_boot),              4),
    CI_low = round(quantile(r2_boot, 0.025),     4),
    CI_hi  = round(quantile(r2_boot, 0.975),     4)
  )
}

ci_data <- bind_rows(boot_rows)
cat("── Bootstrap CI Table ──\n")
print(as.data.frame(ci_data), row.names = FALSE)

p_ci <- ggplot(ci_data, aes(x = paste(Group, Target, sep = "\n"),
                             y = R2, color = Target)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = CI_low, ymax = CI_hi),
                width = 0.25, linewidth = 1) +
  scale_color_manual(values = c(valence = "#4e9af1", energy = "#e85d04")) +
  geom_hline(yintercept = 0.5, linetype = "dashed",
             color = "grey50", linewidth = 0.5) +
  labs(title    = "XGBoost R\u00b2 with 95% Bootstrap Confidence Intervals",
       x = NULL, y = "R\u00b2") +
  theme_bw(base_size = 13) +
  theme(axis.text.x = element_text(size = 9))
print(p_ci)



cat("9. FINAL SUMMARY\n")

all_models <- c("LinearRegression", "ElasticNet", "KNN", "KNN_PCA",
                "RandomForest", "XGBoost", "StackedEnsemble")

plot_master <- master %>% filter(Model %in% all_models)

# R² heatmap 
p_heat <- ggplot(plot_master,
                 aes(x = factor(Model, levels = all_models),
                     y = paste(Group, Target, sep = "\n"),
                     fill = R2)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = R2), size = 3.5, fontface = "bold") +
  scale_fill_gradientn(colors = c("#d73027", "#fee08b", "#1a9850"),
                       limits = c(0, 1), name = "R\u00b2") +
  labs(title    = "R\u00b2 Heatmap — All 7 Models, All Groups, Both Targets",
       subtitle = "Green = better prediction | Red = worse",
       x = "Model", y = NULL) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
print(p_heat)

# RMSE bar chart
p_rmse <- ggplot(plot_master,
                 aes(x = reorder(Model, RMSE), y = RMSE, fill = Target)) +
  geom_col(position = "dodge", alpha = 0.85) +
  facet_wrap(~Group, ncol = 1, scales = "free_y") +
  coord_flip() +
  scale_fill_manual(values = c(valence = "#4e9af1", energy = "#e85d04")) +
  labs(title    = "RMSE by Model, Group, and Target",
       subtitle = "Lower RMSE = better model",
       x = NULL, y = "RMSE") +
  theme_bw(base_size = 12)
print(p_rmse)

# Model complexity vs R² (valence only)
complexity_order <- c(LinearRegression = 1, ElasticNet = 2,
                      KNN = 3, KNN_PCA = 3.5, RandomForest = 4,
                      XGBoost = 5, StackedEnsemble = 6)

p_complex <- plot_master %>%
  filter(Target == "valence") %>%
  mutate(Complexity = complexity_order[Model]) %>%
  filter(!is.na(Complexity)) %>%
  ggplot(aes(x = Complexity, y = R2, color = Group, group = Group)) +
  geom_line(linewidth = 1.2) + geom_point(size = 4) +
  scale_x_continuous(breaks = c(1, 2, 3, 3.5, 4, 5, 6),
    labels = c("LR", "EN", "KNN", "KNN\n+PCA", "RF", "XGB", "Stack")) +
  scale_color_manual(values = MODEL_COLORS) +
  labs(title    = "Model Complexity vs Valence R\u00b2 by Group",
       x = "Model (simple \u2192 complex)", y = "Test R\u00b2") +
  theme_bw(base_size = 13)
print(p_complex)

# Best model per group and target
best_models <- plot_master %>%
  group_by(Group, Target) %>%
  slice_min(RMSE, n = 1) %>%
  ungroup()

cat(" Best Model per Group and Target \n")
print(as.data.frame(best_models %>% select(Group, Target, Model, RMSE, R2)),
      row.names = FALSE)

p_best <- ggplot(best_models,
                 aes(x = Group, y = R2, fill = Model,
                     label = round(R2, 3))) +
  geom_col(alpha = 0.85) +
  geom_text(vjust = -0.4, size = 4, fontface = "bold") +
  facet_wrap(~Target) +
  scale_fill_viridis_d(option = "D", begin = 0.2, end = 0.9) +
  ylim(0, 1.05) +
  labs(title    = "Best Model R\u00b2 per Group and Target",
       x = NULL, y = "Test R\u00b2") +
  theme_bw(base_size = 12) + theme(legend.position = "bottom")
print(p_best)

