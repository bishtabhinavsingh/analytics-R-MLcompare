# Overall objective

Train certain deep learning models (Simple RNN, LSTM, bidirectional LSTM, GRU, bidirectional GRU, 1D Covnet) based on a given dataset (imdb rds file), using given x_train and y_train code. Objective is to:

  - train these models with a minimum of 60% accuracy
  - save all training history files, all models, x_test and y_test and load them to the R-markdown
  - plot all history files
  - evaluate all models
  - display true positive, true negative, false positive and false negative values for each model
  - discuss models based on above parameters

These objectives were met in two part files. Firstly, R file that trains and times the models. Secondly, RMD file that displays the result in markdown.


## Objective 1: Read all output files from the R file
> Assuming all files have been extracted to the same folder that contains the R-markdown file.


## Objective 2: Show statistics for x_test and y_train

This includes:

  - Number of reviews in the test set
  - Number of positive reviews in the test set
  - Number of negative reviews in the test set
 
![Alt text](/artifacts/0.png)

## Objective 3: Display Models and History
![Alt text](/artifacts/1.png)
![Alt text](/artifacts/1a.png)

![Alt text](/artifacts/2.png)
![Alt text](/artifacts/2a.png)

![Alt text](/artifacts/3.png)
![Alt text](/artifacts/3a.png)

![Alt text](/artifacts/4.png)
![Alt text](/artifacts/4a.png)

![Alt text](/artifacts/5.png)
![Alt text](/artifacts/5a.png)

![Alt text](/artifacts/6.png)
![Alt text](/artifacts/6a.png)

## Objective 4: Compare Models
![Alt text](/artifacts/7.png)

