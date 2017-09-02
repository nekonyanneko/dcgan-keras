library(dplyr)

create_data_num <- 100
time_range <- 100
anomaly_num <- 50
period_num <- 2

# Creat Data
white_noize <- abs(as.integer(rnorm(time_range)*10))
df = data.frame(white_noize)
for(i in 2:create_data_num){
  white_noize <- abs(as.integer(rnorm(time_range)*10))
  period <-rep(c(rep(0,40),as.integer(runif(10, min = 40, max = 50))),period_num)
  df[,i] <- white_noize + period[0:100]
  #df[,i] <- white_noize
  #plot(df[,i],type="l")
}


# Anomaly Data
Anomaly <- as.integer(runif(anomaly_num, min = 90, max = 100))
target_index <- as.integer(runif(anomaly_num, min = 51, max = time_range))
for(i in 1:anomaly_num){
  df[target_index[i],i] <- Anomaly[i]
  #plot(df[,i],type="l")
}


# OutPut
#out <- file("./data.txt", "w")
#out <- file("./data_ft.txt", "w")
#out <- file("./data_ft_test.txt", "w")
#out <- file("./data_lstm.txt", "w")
out <- file("./data_lstm_test.txt", "w")
for(i in 1:create_data_num) {
  #data_ft.txt only
  #if(i <= anomaly_num) writeLines(text = sprintf("__label__1"), con = out, sep=" ")
  #else writeLines(text = sprintf("__label__2"), con = out, sep=" ")
  
  for(j in 1:time_range){
    if(j == time_range) writeLines(text = sprintf(toString(df[,i][j])), con = out, sep="\n")
    #else writeLines(text = sprintf(toString(df[,i][j])), out, sep=",")
    else writeLines(text = sprintf(toString(df[,i][j])), out, sep=" ")
  }
}
close(out)
