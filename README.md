Name: Martijn Kok

Studentnumber: 18094627

Teachers: J. Vuurens &  R. Vermeij & T. Andrioli

Project: Wheels

# Portfolio
Individual portfolio of the contribution to the project, my abilities and what i have learned

## Introduction
My name is Martijn Kok and i am a fourth year electrical engineering student that has done Applied Data Science as a minor. During de Applied Data Science minor i learned a lot about data and what can be achieved with just data. Through out the whole minor i have improved my abilities like Project Communication and Data Visualization, but i also learned a lot of new abilities like Predictive Models, Data preparation and Evaluation. This portfolio will be proof of what i have learned during this minor


## Datacamp online course
The main coding language in this minor is python. To get familiar with this coding language i began a online coding tutorial named Datacamp. In Datacamp you have a multiple courses with different goals. The main subject are: handling with pandas dataframes, preparing data, applying models and validating models. 

After finishing 100% of the assigned datacamp courses, i have learned a lot about machine learning and python. Through Datacamp i improve my knowledge about how to use python with lage datasets and improve these datasets for machine learning. Futhermore datacamp tought me what machine learning is and how you can implement different models like KNearestNeighbor, Linear Regression and DecisionTree with python. 
My process of Datacamp can be seen through this link: [Datacamp progress](Datacamp/Assignments-DataCampLearn.pdf)



## Research project

### Task definition

### Evaluation

### Conclusions

### Planning



## Predictive Analytics

### Selecting a Model

#### CNN
The wheels team got a large time series dataset of multiple gyroscope datapoints. I researched into models that would accept a time series as a input. After researching different models i came across a 1D-Convolutional Neural Network: https://www.sciencedirect.com/science/article/pii/S2666307421000140. In this research paper they also use a 1D-CNN Model for sensordata (gyroscope data) and this model got the best results from the used models. 

#### RNN

### Configuring a Model
#### CNN
After desiding that i would use a 1D-CNN model i needed to config a model. This took a long time because there aren't that many 1D-CNN models online. After trail and error i made two different CNN models see link: [CNN1](Datacamp/Assignments-DataCampLearn.pdf), [CNN2](Datacamp/Assignments-DataCampLearn.pdf). These models are based on the same main model. This model consist of a multiple convolutions and linears layers. The difference between the models are that the first model was configured to get a input of a big tensor with multiple features, this was done because the model will be able to detect patterns over the whole dataset and not just one specific moment. The second model was configured to receive tensors that were split into windows of 1 second with a overlap of 0.5 seconds, this was done so the results of could easily be compared to models used by other team members. 

#### RNN

### Training a Model
After the two models were made i trained them with the main dataset of the project. I splitted the dataset into two parts, the train part and test part. The train part was 75% of the data en the test part was 25%. The first time i trained the model had set the epochs on 3000, ofcourse this overfitted the model. To ensure there was no overfitting i visualized the accuracy, loss, recall and precision over the amount of epochs, see: [CNN](Datacamp/Assignments-DataCampLearn.pdf). After looking at the visualizations i saw the model was overfitting and the learning rate of the model was to high. After playing with the learning rate, epochs and size of the convolutional/linear layers, i achieved the best results possible for the model.

### Evaluation a Model
After training the model the results needed to be evaluated. This was easilier said then done, because the dataset that was provided by our problem owner wasn't completed. This meant the the results of the models couldn't be evaluated on the normal way. To evaluate the models i started with looking at the false positives of the models, this was done to check if the false positives are really wrong or if the dataset was wrong. This was done with visualization code that i wrote: [False Positives Visualization](Data_Visualization/Check_False_Negatives.pdf). This code helped me and Daan to check the first false positives of the sprint code. After checking these false positives we found almost 50% of the false positives were actually true positives. After adding these true positives to the dataset i started to evaluate the CNN and RNN models, while other team members continued with checking the false positives. 

For the evaluation of the models the Recall was most important because this was data i knew about it was correct. For both models i made confusion matrixes of the results of the test set. 

### Visualizing the outcome of the Model


## Domain knowledge

### Introduction of the subject field

### Literature research

### Explanation of the Terminology, jargon and definitions



## Data preproccessing 

### Data Exploration

### Data Cleansing

### Data Preparation

### Data explanation

### Data visualization



## Communication

### Presentations

### Writing Paper



## Reflection and evaluation

### Reflection on own contribution on the project 
### Reflection on own learning objectives.
### Evaluation on the group project as a whole
