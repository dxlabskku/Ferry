# Ferry

Ferry services are one of the vital transportation methods in several nations which connect a huge number of islands and mainlands. However, a major disadvantage of ferry services is that it is crucially affected by weather conditions. Thus, informing customers about the regular ferry service operations is important. Considering this, we aim to predict whether ferry services can be provided in a timely manner through machine learning approaches with their meteorological (6 ~ 48 h prior) and operation datasets.
  
   
   
## Dataset
We used ferry operation and maritime meteorological datasets in Incheon, which is one of the main passenger terminals in South Korea.
|Dataset|Features|
|------|---|
|Wind|wind speed (m/s), wind direction (deg), gust windspee (m/s)|
|Temperature|air temperature (°C), water temperature (°C)|
|Wave|maximum wave height (m), significant wave height (m), average wave height (m), wave period (sec), wave direction (deg)|
|Others|observated date, humidity, barometric pressure (hpa)|
#### Table : variables of the collected datasets in Incheon Buoy (data from 2016 to 2022.6)  
   
    
## Experiment
Four machine learning techniques(Decision Tree, Random Forest, Adaboost, K-Nearest Neighbor) have been employed by us for our experiment. And we paired ferry operation datasets with meteorological dataset for each time period (-6 ~ -48 h). 
<img width="1100" alt="shift_process" src="https://user-images.githubusercontent.com/122080807/213160638-f7efe0c4-65d1-4572-9d79-e8c38e597542.png">
#### Figure : Overview of data pairing procedures
When we build a model using Random Forest, prediction performances were higher than others. We achieves accuracy levels of 90.50%(6h) and 88.78% (48h) in timely ferry services. Compared with regulation-oriented determination, our predictive model performed better.
