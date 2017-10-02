Sys.setenv(LANG = "en")
#- Load some useful packages
library('ggplot2')
library('ggthemes')
library('dplyr')
library('mice') # imputation
library('scales')

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
#cat(paste('We have <b>', nlevels(factor(full$Surname)), '</b> unique sur names'))

#- Create a family size variable including the passenger themselves
full$Fsize <- full$SibSp + full$Parch + 1

full$Family <- paste(full$Surname, full$Fsize, sep = "_")

ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()

#- Create discret family size variable
full$FsizeD[full$Fsize == 1] <- "singleton"
full$FsizeD[full$Fsize > 1 & full$Fsize < 5] <- "small"
full$FsizeD[full$Fsize > 4] <- "large"

#- Showing a mosaic plot
mosaicplot(table(full$FsizeD, full$Survived), main = "Family Size by Survival", shade = TRUE)

#- Create a Deck variable
full$Deck <- sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1])

embark_fare <- full %>% filter(PassengerId != 62 & PassengerId != 830)
#- Print embarked fare
# ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
#   geom_boxplot() +
#   geom_hline(aes(yintercept=80), 
#              colour='red', linetype='dashed', lwd=2) +
#   scale_y_continuous(labels=dollar_format()) +
#   theme_few()

#- Fill missing value in Embarked
full$Embarked[c(62, 830)] <- "C"
full[1044, ]
ggplot(full[full$Embarked == "S" & full$Pclass == 3, ], aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) +
  