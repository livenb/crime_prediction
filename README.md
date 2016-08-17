# Community evolution and Crime prediction

>'Don't go there, you will get shot!'

This is the word my friend told me at my first day in US. For all these years, I always concern about the safety when I visit a new area.  
In this project i get the open data of individual crime incidents from 5 cities: San Francisco, Los Angles, Detroit, Louisville and Philidephia. There are several million individual crimes. I will try to answer the following few questions:  
  1. How can i seperate the community one from another? How can i categorize them?  
  2. Over the years, how the crime situation changes in each of the community?  
  3. If I visit a area, can i predict what kind of crime would happen given the time and location?  

### Data Preparation
Data are coming from [publicsafetydataportal.org](https://publicsafetydataportal.org/). Each individual crime incident are consisting with time, location and crime category. But, the police departments in different cities are using different terminology and category system, so the data has to convert into same format and put into the same 10 major crime categories and put them into integer number: `1: Theft/Larcery, 2: Robbery, 3: Nacotic/Alcochol, 4: Assault, 5: Grand Auto Theft, 6: Vandalism, 7: Burglary, 8: Homicide, 9: Sex Crime, 10: DUI`. The geological information are given in different forms across the cities, the missing information are getting from the map API (Bing maps, HERE maps). The data cleaning code can be found in `clean_data.py`.  

### Community Categorization
In order to compare between community, the data has to be aggregate for each zip code and by each type of the crime in a desired time interval. Here is the screenshot for LA data.:
![alt text][la_data]
[la_data]:img\la_data_agg.png
