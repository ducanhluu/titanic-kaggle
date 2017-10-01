Sys.setenv(LANG = "en")
#- Load some useful packages
#library('ggplot2')
library('dplyr')

#- Load data
train_data <- read.csv("./data/train.csv")
test_data  <- read.csv("./data/test.csv")

full <- bind_rows(train_data, test_data)
#- Check data
# str(train_data)
# str(test_data)

#- Feature engineering with passenger name
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)
print(full$Name)
#- Show title counts by sex
table(full$Sex, full$Title)

#- Grab rare titles
rare_title <- c("Capt", "Col", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir", "the Countess")

#- Reassign some title correctly
full$Title[full$Title == "Mlle"] <- "Miss"
full$Title[full$Title == "Ms"] <- "Miss"
full$Title[full$Title == "Mme"] <- "Mrs"
full$Title[full$Title %in% rare_title] <- "Rare Title"

table(full$Sex, full$Title)

#- Grab surname from passenger name
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = '[,.]')[[1]][1])