# Duplicate Information Detection

## Project Description
This project aims to detect the duplicates among the data rows provided which comprise of reported distress calls for disaster-victims, and helps out a human eye to pick out highly similar rows via classifying the rows by attributes of similarity and providing the similarity rates between rows considering the `name` and `address` information. 

At this point, the similarity analysis in this project mainly utilizes `term frequency–inverse document frequency (TF-IDF)` measure and meticulous preprocessing, and it works. The preprocessing phase can also utilize a well-functioning named-entity recognition (NER) model.

### High-Level Process Flow (Tentative)

![image](/img/process_flow.png)

### High-Level Preprocess Flow (Tentative)

![image](/img/preprocess.png)




