# Duplicate Report Dedection

## Project Description
This project aims to detect the duplicates among the data rows provided which comprise of reported distress calls for disaster-victims, and helps out a human eye to pick out highly similar rows via classifying the rows by attributes of similarity and providing the similarity rates between rows considering the `name` and `address` information. 

At this point, the similarity analysis in this project mainly utilizes `term frequency–inverse document frequency (TF-IDF)` measure and meticulous preprocessing, and it works. The preprocessing phase can also utilize a well-functioning named-entity recognition (NER) model.

### High-Level Process Flow (Tentative)

![image](https://user-images.githubusercontent.com/29688260/220365355-a5636a15-e4b3-4ff9-ae3a-86c70b323a59.png)

### High-Level Preprocess Flow (Tentative)

![image](https://user-images.githubusercontent.com/29688260/220373156-877a2764-49ae-4132-9f15-b43c79a5053a.png)




