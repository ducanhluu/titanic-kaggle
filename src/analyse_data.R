Sys.setenv(LANG = "en")
#- Load some useful packages
library('ggplot2')
library('ggthemes')
library('dplyr')
library('mice') # imputation
library('scales')
library("VIM")
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

#- Draw fare distribution
ggplot(full[full$Embarked == "S" & full$Pclass == 3, ], aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) +
  geom_vline(aes(xintercept = median(Fare, na.rm = TRUE)), colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()

#- Fill in missing value in Fare
full$Fare[which(is.na(full$Fare))] <- median(full[full$Embarked == "S" & full$Pclass == 3, "Fare"], na.rm = TRUE)

# Make variables factors into factors
factor_vars <- c('PassengerId','Pclass','Sex','Embarked', 'Title','Surname','Family','FsizeD')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

md.pattern(full)
aggr_plot <- aggr(full, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))

#- Mice imputation
mice_mod <- mice(full[ ,!names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method = 'pmm', seed = 129)
mice_output <- complete(mice_mod)

# Plot age distributions
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

#- Filling missing value in Age
full$Age <- mice_output$Age

any(is.na(full$Age))

# Dealing with mother-children relation
ggplot(full[1:891, ], aes(Age, fill = factor(Survived))) +
  geom_histogram() + 
  facet_grid(.~Sex) +
  theme_few()
  