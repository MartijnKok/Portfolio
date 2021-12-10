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
When i researched for models that can be used on IMU (gyroscope data) i found a interesting research paper about detection specific motion from IMU data with LSTM (Long Short-Term Memory) and a RNN (Recurrent Neural Network), link: [Research Paper LSTM&RNN](Predictive_Models/sensors-21-01264-v2.pdf). This research paper had a lot in commmen with our project. So i decided that i would give a RNN with LSTM a go. 

### Configuring a Model
#### CNN
After deciding that i would use a 1D-CNN model i needed to configure a model. This took a long time because there aren't that many 1D-CNN models online. After trail and error i made a CNN model see link: [CNN](Predictive_Models/1D_CNN_Dataloader.ipynb). This model consist of a multiple convolution and linear layers. The CNN mode is configured to receive tensors that were split into windows of 1 second with a overlap of 0.5 seconds, this was done so the results of could easily be compared to models used by other team members. For every 1 second window the model will decide if there is a sprint or not.

#### RNN
The RNN model is a basic RNN model with a hidden layer and a LSTM classifier. This model has the same input as the CNN model so most parts of the CNN code can be used for the RNN model, link: [RNN](Predictive_Models/RNN_Overlap_Dataloader.ipynb). The model was based on a RNN model that J. Vuurens showed during a lecture. 


### Training a Model
After the two models were made i trained them with the main dataset of the project. I splitted the dataset into two parts, the train part and test part. The train part was 75% of the data en the test part was 25%. The first time i trained the model i set the epochs on 3000, ofcourse this overfitted the model. To ensure there was no overfitting i visualized the accuracy, loss, recall and precision over the amount of epochs, see: [CNN](Predictive_Models/1D_CNN_Dataloader.ipynb) and [RNN](Predictive_Models/RNN_Overlap_Dataloader.ipynb). After looking at the visualizations i saw the model was overfitting and the learning rate of the model was to high. To fix the overfitting i used a dataloader on both models also i played with the learning rate and the epochs. For the CNN i also tuned the size of the Convolusional layers and the linear layers. I tuned the RNN with different hidden layer sizes. For both model i played around and looked at the results for every change and finaly tunned the model to achieve the highest Recall with a acceptable precision.

### Evaluation a Model
After training the model the results needed to be evaluated. This was easilier said then done, because the dataset that was provided by our problem owner wasn't completed. This meant the the results of the models couldn't be evaluated on the normal way. To evaluate the models i started with looking at the false positives of the models, this was done to check if the false positives are really wrong or if the dataset was wrong. This was done with visualization code that i wrote: [False Positives Visualization](Data_Visualization/Check_False_Positives.ipynb). This code helped me and Daan to check the first false positives of the sprint code. After checking these false positives we found almost 50% of the false positives were actually true positives. After adding these true positives to the dataset i started to evaluate the CNN and RNN models, while other team members continued with checking the false positives. 

For the evaluation of the models the Recall was most important because this was data i knew about it was correct. For both models i made confusion matrixes of the results of the test set. I used these confusion matrixes to compare the two neural network. In the table below are the Recall/Precision and Accuracy score for the dectection of a sprint.

*All Data*
| Models | Recall  | Precision  | Accuracy |
| :---:   | :-: | :-: | :-: |
| CNN | 0.86 | 0.78| 0.83 |
| RNN | 0.89 | 0.83| 0.88 |

*Just True Positives and False Positives*
| Models | Recall  | Precision  | Accuracy |
| :---:   | :-: | :-: | :-: |
| CNN | 0.94 | 0.59| 0.73 |
| RNN | 0.92 | 0.69| 0.79 |

From the tables above you can see the RNN model got the best results in 'All Data' for all the testing parameters. For the table with 'Just True Positives and False Positives' the CNN scored a better Recall but way worse accuracy and precision, from this informaton i choose to go futher with the RNN model because the average score of the RNN fits better for our purpose.

### Visualizing the outcome of the Model
The results of both models is visualized in the code by plotting the Accuracy and Loss of both the train and test set. Also for both models the Recall and Precision for the test set is plotted, see [CNN](Predictive_Models/1D_CNN_Dataloader.ipynb) and [RNN](Predictive_Models/RNN_Overlap_Dataloader.ipynb). To visualize the results more for both models the confusion matrix was plotted. This showed clearly how the models were preforming. 

## Domain knowledge

### Introduction of the subject field

### Literature research

### Explanation of the Terminology, jargon and definitions



## Data preproccessing 

### Data Exploration
Before i could start with coding i needed to understand the dataset i was going to work with. The dataset of the wheel project consists of sensordata of two IMU sensors on a wheelchair, one on the frame and one on the right wheel. This Sensordata was from a specific player that played a specific match. The action of this player were tagged in a seperate file with the hand of video data. Before i could explore the data, the actions needed to be alligned with the sensordata. This was done by the whole project group. 

To understand this alligned data i have written a visualization code that will visualize all IMU sensorsdata for the tagged wheelchair basketball action like Fast Defencen. see [Detecting Patterns](Data_Visualization/Timestampfastdefence.ipynb). This visualization code helped me to understand the correlations between specific action like Fast Defence and the output of the IMU sensors. With this information i made a hypothesis that the variables wheelRotationalSpeedX and framRotationSpeedY would work the best to detect Fast defences.

After visualizing the dataset, the features that i found most sutable where used for the training of a K-Nearest Neighbor model to detect sprinting behavior. When fitting the data to the model i ran into a problem. The model didn't have great results because the dataset wasn't balanced. The balance of the dataset was visualized to see how bad it was see: [Balancing](Predictive_Models/1D_CNN_Dataloader.ipynb). The dataset was more inbalanced then i thought, with this information i balanced the dataset. After balancing the dataset the results of the model improved.

### Data Cleansing
When the group received the dataset, the dataset needed to be cleaned. The dataset had two big problem, the first problem was that the dataset consisted of many NaN values. The second problem was that the dataset had a lot of unneccesary data, for example the match of a specific player had a duration of 45 minutes but the dataset of this player had data for 60 minutes. 

To solve these two problems i wrote a simple program that would replace all the NaN in the dataset with the value 0 and would delete all the data where the player wasn't actively playing. The program that realized this is [Clean Data](Data_Preparation/Clean_Data.ipynb).

### Data Preparation

#### Balancing Dataset
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#### Transforming Dataset
To Transform the dataset a lowpass filter was fitted to the model. This lowpass filter filtered out the high frequencies in the dataset. This was helpfull because now the dataset is more smooth and most of the big outliers are removed, this improved the results of our predictive models. After the lowpass filter a differential equation was fitted to the dataset. This was done to identify if a player is accelerating. This was done through a differential equation of the wheelRotatationSpeedX. This resulted in the acceleration of the wheel.

The code that i wrote for the differential equation and lowpassfilter can be seen in: [Random Forrest Classifier](Predictive_Models/RandomForrestCLassifier.ipynb). 

### Data explanation
The sensor dataset consisted of 2 IMU sensors with both 3 Axis (XYZ), this resulted into a total of 6 features in the dataset. The dataset was expanded with a few proccesed features like frameRotation, a timeLine and frameAngle. In total the sensor dataset consisted of 16 features that could be used for detection specific actions. All features had a samplefrequency of 100Hz, this meant that for every second there were 100 datapoint for each feature. 

The Action that correspondent with sensor dataset where tagged by a human using vidoe material. These action and there timestamps where noted in a seperate dataset. 

### Data visualization
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

## Communication

### Presentations

#### Internal/External 
During the minor i have given multiple internal/external presentations. These presentations showed the progress of my project group during the minor and what we were going to do in the next scrum sprint. Also giving the presentations gave me a opportunity to improve my english presentations skills and ask question to the class about problems i got with my project. The links to the presentations are: [Internal 06-11](Presentations/Wheels_06-11.pdf), [Internal 24-10](Presentations/Project_Wheels_24-10.pdf), [External 10-12](Presentations/ExternalWheels-9-12.pdf).

#### Learning Lab
I gave a learning lab during the minor about data preparation for Sensors data, see: [Learning Lab](Presentations/LearningLab.pdf). This Learning lab was focust on teaching the class the importance of data preparation. During the Learning lab i also gave the class a challenge to win a beer, sadly only one person submitted a jupyter notebook. The challengs was to improve the results of a dataset with just data preparation see: [Challenge](Presentations/LearningLab.ipynb)

### Writing Paper
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

## Reflection and evaluation

### Reflection on own contribution on the project 
### Reflection on own learning objectives.
### Evaluation on the group project as a whole
