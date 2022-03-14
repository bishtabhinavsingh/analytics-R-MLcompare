library(keras)
library(tidyverse)
library(tensorflow)

 
setwd("/Users/abi/Documents/")
set.seed(123)
n_sample <- 3000; maxlen <- 200; max_features <- 3000
imdb = read_rds("imdb.rds")
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb # Loads the data
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
sample_indicators = sample(1:nrow(x_train), n_sample)

x_train <- x_train[sample_indicators,] # use a subset of reviews for training
y_train <- y_train[sample_indicators]  # use a subset of reviews for training
x_test <- x_test[sample_indicators,] # use a subset of reviews for testing
y_test <- y_test[sample_indicators]  # use a subset of reviews for testing


##########simple RNN
model_RNN <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 32) %>%
  layer_simple_rnn(units = 32)  %>% # return_sequences = FALSE, by default
  layer_dense(units = 1, activation = "sigmoid")

model_RNN %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

start_time <- Sys.time() ##########stopwatch - begin RNN

history_RNN <- model_RNN %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)


##########stopwatch - end RNN
end_time <- Sys.time()
RNN <- end_time - start_time


##########LSTM

model_LSTM <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 32) %>%
  layer_lstm(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")

model_LSTM %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

start_time <- Sys.time() ##########stopwatch - begin RNN

history_LSTM <- model_LSTM %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

##########stopwatch - end RNN
end_time <- Sys.time()
LSTM <- end_time - start_time


####### bi-directional LSTM

model_bLSTM <- keras_model_sequential() %>%
  layer_embedding(input_dim = 100000, output_dim = 32) %>%
  bidirectional(
    layer_lstm(units = 32)
  ) %>%
  layer_dense(units = 1, activation = "sigmoid")

model_bLSTM %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

start_time <- Sys.time() ##########stopwatch - begin bLSTM
history_bLSTM <- model_bLSTM %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

##########stopwatch - end bLSTM
end_time <- Sys.time()
bLSTM <- end_time - start_time


#############GRU

model_GRU <- keras_model_sequential() %>%
  layer_embedding(input_dim = 100000, output_dim = 32) %>%
  layer_gru(units = maxlen, return_sequences = FALSE, name = "gru") %>%
  layer_dense(units = 1, activation = "sigmoid")

model_GRU %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)


start_time <- Sys.time() ##########stopwatch - begin GRU
history_GRU <- model_GRU %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

##########stopwatch - end GRU
end_time <- Sys.time()
GRU <- end_time - start_time

#############bi-directional GRU

model_bGRU <- keras_model_sequential() %>%
  layer_embedding(input_dim = 100000, output_dim = 32) %>%
  bidirectional(layer_gru(units = maxlen, return_sequences = FALSE, name = "gru")) %>%
  layer_dense(units = 1, activation = "sigmoid")

model_bGRU %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)


start_time <- Sys.time() ##########stopwatch - begin bGRU
history_bGRU <- model_bGRU %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

##########stopwatch - end bGRU
end_time <- Sys.time()
bGRU <- end_time - start_time


#############1D covnet

model_1d_cov <- keras_model_sequential() %>%
  layer_embedding(input_dim = 100000, output_dim = 32) %>%
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 5) %>%
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 1, activation = "sigmoid")

model_1d_cov %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)


start_time <- Sys.time() ##########stopwatch - begin 1d_cov
history_1d_cov <- model_1d_cov %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

##########stopwatch - end 1d_cov
end_time <- Sys.time()
T_1d_cov <- end_time - start_time

####### History to RDS
history_RNN <- write_rds(history_RNN, "history_RNN.rds")
history_LSTM <- write_rds(history_LSTM, "history_LSTM.rds")
history_bLSTM <- write_rds(history_bLSTM, "history_bLSTM.rds")
history_GRU <- write_rds(history_GRU, "history_GRU.rds")
history_bGRU <- write_rds(history_bGRU, "history_bGRU.rds")
history_1d_cov <- write_rds(history_1d_cov, "history_1d_cov.rds")

##### Saving model
model_RNN %>% save_model_hdf5("model_RNN.h5")
model_LSTM %>% save_model_hdf5("model_LSTM.h5")
model_bLSTM %>% save_model_hdf5("model_bLSTM.h5")
model_GRU %>% save_model_hdf5("model_GRU.h5")
model_bGRU %>% save_model_hdf5("model_bGRU.h5")
model_1d_cov %>% save_model_hdf5("model_1d_cov.h5")

#####x_test and y_test to rds files
write_rds(x_test, "x_test.rds")
write_rds(y_test, "y_test.rds")

#####Creating time-table for reference later
df_t <- data.frame(Model = c('RNN', 'LSTM', 'bLSTM', 'GRU', 'bGRU', 'T_1d_cov'),
                   Time = c(RNN, LSTM, bLSTM, GRU, bGRU, T_1d_cov))
write.csv(df_t, "time_t.csv")