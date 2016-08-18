# Community evolution and Crime prediction

>'Don't go there, you will get shot!'

This is the word my friend told me at my first day in US. For all these years, I always concern about the safety when I visit a new area.  
In this project i get the open data of individual crime incidents from 5 cities: San Francisco, Los Angles, Detroit, Louisville and Philidephia. There are several million individual crimes. I will try to answer the following few questions:  
  1. How can i seperate the community one from another? How can i categorize them?  
  2. Over the years, how the crime situation changes in each of the community?  
  3. If I visit a area, can i predict what kind of crime would happen given the time and location?  

### Data Preparation
Data are coming from [publicsafetydataportal.org](https://publicsafetydataportal.org/). Each individual crime incident are consisting with time, location and crime category. But, the police departments in different cities are using different terminology and category system, so the data has to convert into same format and put into the same 10 major crime categories and put them into integer number: `1: Theft/Larcery, 2: Robbery, 3: Nacotic/Alcochol, 4: Assault, 5: Grand Auto Theft, 6: Vandalism, 7: Burglary, 8: Homicide, 9: Sex Crime, 10: DUI`. The geological information are given in different forms across the cities, the missing information are getting from the map API (Bing maps, HERE maps). The data cleaning code can be found in `clean_data.py`.  

### Community Categorization and Evolution
In order to compare between community, the data has to be aggregate for each zip code and by each type of the crime in a desired time interval. Here is the screenshot for LA data:  
![aggregate data sample][la_data]
[la_data]:img/la_data_agg.png
The problem is the minor crimes like theft have much higher statistics than the series crime like homocide. So the min-max normalization has been applied before the data running into the model, which transforms all data into between 0 and 1. Zero stands for no such crime in this time interval, One stands for highest record of such crime in this time interval. Normalization help reduce the scale problem for distance based method like k-mean. Another advantage of min-max normalization is keeping the data non-negative, which can be used in NMF decomposition.  
For this clustering problem, I consider the k-mean as first. Using the empirical elbow method as well as the Silhouette score, I can determine the number of clusters in the data, 4 for San Francisco and 5 for Los Angeles. LA elbow plot is shown as following:  
![aggregate data sample][elbow_la]
[elbow_la]:img/elbow_la.png
The centroids of the k-means showed that k-means separate the communities by the magnitude of all kinds crime, the communities can be categorize as as low crime rate, high crime rate, extra high crime rate _etc_. Most of the communities barely move between the clusters.  
However, i am more interesting about the dominant type or types crime in communities, and the transition between the clusters. So, i considered to use NMF (non-negative matrix factorization), which are widely used to get the topic or latent feature for NLP or sound waves. In this case, the matrix decomposition can supply the crime pattern in each type of the communities.
![aggregate data sample][year_H]
[year_H]:img/year_H.png
