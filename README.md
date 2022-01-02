Name: Martijn Kok

Studentnumber: 18094627

Teachers: J. Vuurens &  R. Vermeij & T. Andrioli

Project: Wheels

# Portfolio
Individual portfolio of the contribution to the project, my abilities and what i have learned

## Introduction
My name is Martijn Kok and i am a fourth year electrical engineering student that has done Applied Data Science as a minor. During de Applied Data Science minor i learned a lot about data and what can be achieved with just data. Through out the whole minor i have improved my abilities like Project Communication and Data Visualization, but i also learned a lot of new abilities like Predictive Models, Data preparation and Evaluation. This portfolio will be proof of what i have learned during this minor.


## Datacamp online course
The main coding language in this minor is python. To get familiar with this coding language i began a online coding tutorial named Datacamp. In Datacamp you have a multiple courses with different goals. The main subject are: handling with pandas dataframes, preparing data, applying models and validating models. 

After finishing 100% of the assigned datacamp courses, i have learned a lot about machine learning and python. Through Datacamp i improved my knowledge about how to use python with lage datasets and improve these datasets for machine learning. Futhermore datacamp tought me what machine learning is and how you can implement different models like KNearestNeighbor, Linear Regression and DecisionTree with python. 
My process of Datacamp can be seen through this link: [Datacamp progress](Datacamp/Assignments-DataCampLearn.pdf), [Datacamp proof](Datacamp/MartijnKok_DataCampProfile_Overview.pdf)




## Research project

### Task definition

### Evaluation

### Conclusions

### Planning



## Predictive Analytics

### Selecting a Model

#### CNN
The wheels team got a large time series dataset of multiple gyroscope datapoints. I researched into models that would accept a time series as a input and that are used in classifying movements from humans with sensordata. After researching different models i came across a 1D-Convolutional Neural Network used for human activity recognition through sensor data: https://www.sciencedirect.com/science/article/pii/S2666307421000140. In this research paper they also use gyroscope data to recognize movements, this was done with a 1D-CNN Model, a KNN, a SVM, etc. In this researchpaper the 1D-CNN significantly outperforms traditional approaches. This persuaded me to try a 1D-CNN model on my project.

#### RNN
When i researched for models that can be used on IMU (gyroscope data) i found a interesting research paper about detection specific motion from IMU data with LSTM (Long Short-Term Memory) and a RNN (Recurrent Neural Network), link: [Research Paper LSTM&RNN](Predictive_Models/sensors-21-01264-v2.pdf). This research paper showed a lot of potential. But i wasn't sure about it so i researched more about which methods are already used to recognize activities. I found another paper that used LTSM RNN model for recognizing activities of humans through gyroscopes mounted on the wrist https://www.scitepress.org/Papers/2018/65852/65852.pdf. This showed a accuracy of 96.63% for detecting if a person was walking, idle, run, swinging or crouching. This whole paper had a lot in common with my project and showed great results with a LSTM RNN model so i decided that i would give a RNN with LSTM a go to see if the results would be beter then the CNN model. 

#### Conclusion
After researching the 1D-CNN and the RNN, i found out that both models are widely used in recognizing human activity with IMU data and show great results. This has a strong resembles with the classification i want to do for the project. Therefore the models could also have great results on my dataset. To see which model would preform best, i made both models and compared the recall, precision and accuracy.

### Configuring a Model
#### CNN
After deciding that i would use a 1D-CNN model i needed to configure a model. This took a long time because there aren't that many 1D-CNN models online. After trail and error i made a CNN model see link: [CNN](Predictive_Models/1D_CNN_Dataloader.ipynb). This model consist of a multiple convolution and linear layers. The CNN model is configured to receive tensors that were split into windows of 1 second with a overlap of 0.5 seconds, this was done so the results of could easily be compared to models used by other team members. For every 1 second window the model will decide if there is a sprint or not. The hyperparameters of this model (like learning rate and linear layer size) were tuned during the training of this model.

#### RNN
The RNN model is a basic RNN model with a hidden layer and a LSTM classifier. This model has the same input as the CNN model so most parts of the CNN code can be used for the RNN model, link: [RNN](Predictive_Models/RNN_Overlap_Dataloader.ipynb). The model was based on a RNN model that J. Vuurens showed during a lecture. The hyperparameters of this model (like learning reate and hidden layer size) were tuned during the training of this model. This model is also configured to receive tensors that were split into windows of 1 second with a overlap of 0.5 seconds. This was done to make the RNN en CNN model mostly the same so the data preparation steps aren't that different.


### Training a Model
After the two models were made i trained them with the main dataset of the project. I splitted the dataset into two parts, the train part and valid part. The train part was 75% of the data en the valid part was 25%. The first time i trained the model i set the epochs on 3000, ofcourse this overfitted the model. To ensure there was no overfitting i visualized the accuracy, loss, recall and precision over the amount of epochs, see: [CNN](Predictive_Models/1D_CNN_Dataloader.ipynb) and [RNN](Predictive_Models/RNN_Overlap_Dataloader.ipynb). After looking at the visualizations i saw that both models were overfitting and the learning rate and epochs of the models were to high. To fix the overfitting i used a dataloader on both models also i played with the learning rate and the epochs. This dataloader trained the model with a batch size of 64. For the CNN i also tuned the size of the convolutional layers and the linear layers. I tuned the RNN with different hidden layer sizes. For both model i played around and looked at the results for every change and finaly tuned the model to achieve the highest recall with a acceptable precision.

### Evaluation a Model
After training the model the results needed to be evaluated. This was easilier said then done, because the dataset that was provided by our problem owner wasn't completed. This meant the the results of the models couldn't be evaluated on the normal way. To evaluate the models i started with looking at the false positives of the models, this was done to check if the false positives are really wrong or if the dataset was wrong. This was done with visualization code that i wrote: [False Positives Visualization](Data_Visualization/Check_False_Positives.ipynb). This code helped me and Daan to check the first false positives of the sprint code. After checking these false positives we found almost 50% of the false positives were actually true positives. After adding these true positives to the dataset i started to evaluate the CNN and RNN models, while other team members continued with checking the false positives. 

For the evaluation of the models the Recall was most important because this that I knew was correct. For both models i made confusion matrixes of the results of the valid datasset. I used these confusion matrixes to compare the two neural network. In the table below are the Recall/Precision and Accuracy score for the dectection of a sprint.

*All Data*
| Models | Recall  | Precision  | Accuracy |
| :---:   | :-: | :-: | :-: |
| CNN | 0.86 | 0.78| 0.83 |
| RNN | 0.91 | 0.86| 0.90 |

*Just True Positives and False Positives*
| Models | Recall  | Precision  | Accuracy |
| :---:   | :-: | :-: | :-: |
| CNN | 0.94 | 0.59| 0.73 |
| RNN | 0.92 | 0.74| 0.82 |

From the tables above you can see the RNN model got the best results in 'All Data' for all the testing parameters. For the table with 'Just True Positives and False Positives' the CNN scored a better Recall but way worse accuracy and precision, from this informaton i choose to go futher with the RNN model because the average score of the RNN fits better for our purpose then the CNN.

### Visualizing the outcome of the Model
The results of both models is visualized in the code by plotting the Accuracy and Loss of both the train and test set. Also for both models the Recall and Precision for the test set is plotted, see [CNN](Predictive_Models/1D_CNN_Dataloader.ipynb) and [RNN](Predictive_Models/RNN_Overlap_Dataloader.ipynb). To visualize the results more for both models the confusion matrix was plotted. This showed clearly how the models were preforming. 

## Domain knowledge

### Introduction of the subject field
This project 
Sensor data 
IMU
Wheelchair basketball


### Literature research
Before I could begin with making machine learning models for my project I needed to research literature to know what alreaby has been done for the problem my project faces. The first step was to understand what exactly the problem of my project was. My project was to use machine learning to recognize specific wheelchar basketball movements from IMU recordings. While researching i found a researchpaper that goes deeper into the problem of my project (https://dl.acm.org/doi/pdf/10.1145/2700648.2809845), this research paper explains the problems that wheelchair athletes have with tracking their activities in sports like basketball, rugby and tennis. This researchpaper made me understand the problem of my project better.

Now I knew what the exact problem is of my project, I wanted to know if there are any people that already tried to solve this problem. I found a few research papers that already tried to recognize specific movements like sitting and walking with the help of IMU recordings. But all of these research papers were for people that weren't in a wheelchair, but the approach of these researches could also be used for my project. A few examples of these research papers are: https://www.sciencedirect.com/science/article/pii/S2666307421000140, https://www.scitepress.org/Papers/2018/65852/65852.pdf, https://www.mdpi.com/1424-8220/21/4/1264/htm and http://www.ijpmbs.com/uploadfile/2017/1227/20171227050020234.pdf.

All of above research papers showed me that different kinds of (NN) neural networks are commonly used to recognize activities from IMU data. But i had no idee how a neural network worked, to understand neural network I started to research how you can make a NN from scratch. While researching I found a great tutorial that showed me how you can make a NN and what everything meant https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/. This tutorial helped me to make a basic NN and understand how it is made. But a basic NN neural network wasn't great for recognizing activities, most people used a 1D-CNN (Convolutional Neural Network) or RNN (Recurrent Neural Network) model. Theser NN networks were more complicated then a simple NN network. To understand how these models worked I researched into simple visualizations of these models https://towardsdatascience.com/understanding-1d-and-3d-convolution-neural-network-keras-9d8f76e29610, https://aditi-mittal.medium.com/understanding-rnn-and-lstm-f7cdf6dfc14e. These visualization helped me to understand how the models worked and know I could implement these models. For the 1D-CNN I researched a lot because there weren't many tutorial about how you can implement a 1D-CNN model with Pytorch, but i found a tutorial that helped me to make a basic 1D-CNN, https://wandb.ai/authors/ayusht/reports/Implementing-and-Tracking-the-Performance-of-a-CNN-in-Pytorch-An-Example--VmlldzoxNjEyMDU. For the RNN I used a example from a lecture that was given by Jeroen Vuurens


### Explanation of the Terminology, jargon and definitions
* IMU, During my project there was a lot about IMU's (Inertial Measuring unit). A IMU is a sensor that is able to record accelerations in 3 Axis (XYZ), With these measurements you can see small movements. 

* Classification/Regression, In machine learning your have multiple kinds of machine learning. During this minor i mainly focussed on two kind of machine learning, Classification and Regression. 
 * Classification, The first kind of machine learning is Classification, classification is a kind of machine learning that can classify data. For example when you have 20 pictures of dogs and there are 4 different breeds, a classification model will be able to predict which of the 4 breed each dog is. 
 * Regression, When you don't just want to classify something but you want to predict a specific value, you can use regression. A regression model is able to predict exact values, for example how much the sales of a company will be next year. 

* Pre proccesing 
  * Balancing
  
* NN
  * Epochs
  * Learningrate
  * linear layers
  * Overfitting
  * Underfitting
  
* CNN
  * Tensors
  
* RNN
  * LSTM
  
* Evaluation
  * Cross Validation/ N-fold Cross Validation 
  * Confusion Matrix



## Data preproccessing 

### Data Exploration
Before i could start with coding i needed to understand the dataset i was going to work with. The dataset of the wheel project consists of sensordata of two IMU sensors on a wheelchair, one on the frame and one on the right wheel. This Sensordata was from a specific player that played a specific match. The action of this player were tagged in a seperate file with the hand of video material. Before i could explore the data, the actions needed to be alligned with the sensordata. This was done by the whole project group. 

To understand this alligned data i have written a visualization code that will visualize all IMU sensorsdata for the tagged wheelchair basketball action like Fast Defences see [Detecting Patterns](Data_Visualization/Timestampfastdefence.ipynb). This visualization code helped me to understand the correlations between specific action like Fast Defence and the output of the IMU sensors. A example is that through the visualization code I saw the pattern of a fast defence, this fast defence consisted of a fast turn before a fast sprint. This was best seen with the wheelRotationalSpeedX and frameRotationalSpeedY of the IMU's. With this information i made a hypothesis that the variables wheelRotationalSpeedX and framRotationSpeedY would work the best to detect Fast defences.

After visualizing the dataset, the features that i found most sutable where used for the training of a K-Nearest Neighbor model to detect sprinting behavior. When fitting the data to the model i ran into a problem. The model didn't have great results because the dataset wasn't balanced. The balance of the dataset was visualized to see how bad it was see: [Balancing](Predictive_Models/1D_CNN_Dataloader.ipynb). The dataset was more inbalanced then i thought, with this information i balanced the dataset. After balancing the dataset the results of the model improved. The balancing of the dataset was done through oversampling of the postivie resutls. This was done 2 times for the [CNN](Predictive_Models/1D_CNN_Dataloader.ipynb) and [RNN](Predictive_Models/RNN_Overlap_Dataloader.ipynb) model, because the balancing process of each model was a little different because of the input of each model was different.

### Data Cleansing
When the group received the dataset, the dataset needed to be cleaned. The dataset had two big problem, the first problem was that the dataset consisted of many NaN values. The second problem was that the dataset had a lot of unneccesary data, for example the match of a specific player had a duration of 45 minutes but the dataset of this player had data for 60 minutes. 

To solve these two problems i wrote a simple program that would replace all the NaN in the dataset with the value 0. To fix the problem of unneccesary data I used a document were all the start and stop timestamps of the video were linked to the dataset. For example the video started at the 1500 point of the dataset and ended at the 3500 point, the code would delete all the data before the 1500 point and everything after the 3500 point. The program that realized this is [Clean Data](Data_Preparation/Clean_Data.ipynb).

### Data Preparation

#### Transforming Dataset with math
To Transform the dataset a lowpass filter was fitted to the model. A lowpass filter was choosen because the IMU's had a sample frequency of 100Hz and some really small movements like a turn of 0.1 degree would be seen by the IMU. For the project we only needed to look at the big movements like turning and sprinting. This meant the small movements didn't matter to us and could only affect our results. The lowpass filter deleted all the small movements and made the dataset more smooth, this improved the results of our predictive models with a small percentage off +1% acurracy, +1% recall and +1% precision. After the lowpass filter a differential equation was fitted to the dataset. This was done to identify if a player is accelerating and could help the model with detecting sprints. This was done through a differential equation of the wheelRotatationSpeedX. This resulted in the acceleration of the wheel. The code that i wrote for the differential equation and lowpassfilter can be seen in: [Random Forrest Classifier](Predictive_Models/RandomForrestCLassifier.ipynb). The differential of the wheelspeed resulted in a increase of 1% accuracy, 3.5% recall and 4% precision.

#### Transforming Dataset through comparison of false positives
A big problem during the project was that the received dataset didn't have all true positives tagged. Therefore when a model is going to classify movements there will be many false positives. To expand the dataset i wrote code that will help with expanding the dataset. The goal of the code is to compare the timestamps of false negatives of two machine learning models to conclude if a false positives is actually false or true. If both models predicted a false positives on the same timestamps i assumed that that the models were right and added a sprint to the dataset at this specific timestamp see [Comparison of FP](Data_Preparation/Compare_Results_Models.ipynb).

### Data explanation
The sensor dataset consisted of 2 IMU sensors with both 3 Axis (XYZ), this resulted into a total of 6 features in the dataset. The dataset was expanded with a few proccesed features like frameRotation, a timeLine and frameAngle. In total the sensor dataset consisted of 16 features that could be used for detection specific actions. All features had a samplefrequency of 100Hz, this meant that for every second there were 100 datapoint for each feature. 

The Action that correspondent with sensor dataset where tagged by a human using video material. These action and there timestamps where noted in a seperate dataset. These two datasets needed to be combined before it can be used for machine learning. 

### Data visualization
Before i could start with training predictive models in needed to understand the data. To understand this data i have written a visualization code that will visualize all IMU sensors for wheelchair basketball action in the dataset like a fast defence, see [Detecting Patterns](Data_Visualization/Timestampfastdefence.ipynb). This visualization code helped me understand the importance of visualization/understanding your data before you can train a model. Through the visualization the best features for the model can be choosen, like the wheelspeed of the wheelchair and the frame rotation speed. 

Throughout the whole data preparation, training, tuning and validating of the predictive models data visualization was used. A example is the visualization of the balance of the data as seen at data preparation. Furtermore visualization was used when creating the CNN and RNN models as seen in training and validating the models.

## Communication

### Presentations

#### Internal/External 
During the minor i have given multiple internal/external presentations. These presentations showed the progress of my project group during the minor and what we were going to do in the next scrum sprint. Also giving the presentations gave me a opportunity to improve my English presentations skills and ask question to the class about problems i got with my project. The links to the presentations are: [Internal 06-11](Presentations/Wheels_06-11.pdf), [Internal 24-10](Presentations/Project_Wheels_24-10.pdf), [External 10-12](Presentations/ExternalWheels-9-12.pdf).

#### Learning Lab
I gave a learning lab during the minor about data preparation for Sensors data, see: [Learning Lab](Presentations/LearningLab.pdf). This Learning lab was focust on teaching the class the importance of data preparation. During the Learning lab i also gave the class a challenge to win a beer, sadly only one person submitted a jupyter notebook. The challengs was to improve the results of a dataset with just data preparation see: [Challenge](Presentations/LearningLab.ipynb).

### Writing Paper
For the research paper i have written particulaire parts like the part about recurrent neural networks, validation, and results. Also i helped a team member with question about the content of the researchpaper for example information about validation and data preperation. Furthermore i worked together with my whole team to improve the paper by peer reviewing eachothers work multiple times. I was also responsible for the layout of the whole paper, this included the references to images and
sources. 

## Reflection and evaluation

### Reflection on own contribution on the project
*STARR*
| | |
| :---:   | :-: |
| Situation | During the minor I was part of the Wheels project group. This project was about how you can use IMU data to classify specific wheelchair basketball movements. |
| Task | During the project I wanted to make a good contribution to my project team and didn't want to lack in work ethic. Furthermore I wanted to share my knowledge about specific subject I learned from school to the rest of my project team. |
| Action | In the project I started with learning the basics of machine learning through a online programming course named Datacamp. When this was finished I began with focussing on what the possibilities were with the given dataset and started with writing code to improve the given dataset. When this was done is started to dive into predictive models like neural networks through research and trial and error, to understand how it all worked and how I could be implemented into my project.  In between all of this I gave internal and external presentations to the class and external people|
| Results | I finished the online programming course Datacamp and wrote code that worked pretty well to fix a few issues that we had with the dataset like cleaning and finding more ground truths. After that I finished multiple Neural Network models like a 1D Convolution Neural Network and a Recurrent Neural Network with Long-Short Term Memory that were heavily used during the project. In these models and models of my teammates I implemented many features that helped me and my teammates to have better results on our models, a few examples are data loaders, data cleaners and validation programs. I also helped my teammates with coding their parts so that they wouldn’t be stuck on problems. Furthermore I wrote parts of the Research paper that I knew most off and kept the whole class updated on my project through internal and external presentations. Also I gave a learning lab to the whole class about the importance of data preparation of sensor data and how you could implement it. |
| Reflection| My coding contribution to this project was really good. I did a lot of the coding for my project and helped my teammates with coding when it was needed. For the research paper I had a good but not excellent contribution, I wrote what in needed to write and i reviewed everbody's work but i did just as much work as my teammates. I gave a lot of presentations during the project and I kept the class updated about my project, I also gave a learning lab about data preparation for sensor data. This was a great contribution to the project and the whole class.|

### Reflection on own learning objectives
*STARR*
| | |
| :---:   | :-: |
| Situation | I started with the Minor Applied Data Science to learn more about the use of machine learning models and how you can use data. |
| Task | During the minor I wanted to expand my knowledge about machine learning and learn about the possibilities you have with data. Also I wanted to work with student from different educational backgrounds then me to find out how much they know about coding and learn form the knowledge they have that I don’t have.  |
| Action | In the minor I started with a project that had four people with different educational background then me. In this project I worked with the scrum method to plan the whole project. During this project I also worked with many machine learning models and with different datasets. I also followed every lecture the teachers gave about machine learning and other subject like data visualization and research during the minor. |
| Results |  The thing I learned could be split into two parts, applied data science and teamwork. In the minor I learned a lot about applied data science and what the possibilities are of data. I also learned how you can implement different models like KNearestNeighbour, 1D-CNN and RNN. For these model I learned how you can tune models and what is important to do before you start working with big datasets like cleaning/preparing your data and understanding your data. For the teamwork aspect of this minor I learned how you can work together with people with different educational background then you and have a better end result because of it. I also used a planning method I would never use on my own and learned how you can organize a project through scrum.|
| Reflection| I learned a lot from the whole minor and my project, it was a great learning experience. I have learned a lot about applied data science and what is means to be a data scientist. It was fun to work with data and see what you can do with is, but it’s not something I want to continue with in my career because I miss making something physicals and don’t want to sit behind a computer all day. For the teamwork I maybe learned the most during this minor. I was great to work with people with different educational backgrounds then me because they have different views on specific problems. The way different kind of people work together and how you can use everybody’s talent for specific tasks was really interesting for me. |

### Evaluation on the group project as a whole
*STARR*
| | |
| :---:   | :-: |
| Situation |  In the minor I worked together with 4 other persons to make a machine learning script that can analyse IMU data and classify specific wheelchair basketball movements with it. |
| Task | The goal of the project was to work together to research and make a machine learning model that will classify specific wheelchair basketball movements. This was done for the Dutch female Paralympic team. For this project I needed to work close with my team and use everybody’s strong point to achieve a great and result |
| Action |  In the project me and my team started with exploring our project and what we needed to do. With the knowledge about this project, we wrote a plan of approach for us and our problem owner. To organize our project, we used the scrum method, this included a stand up every morning at 9:30 and a retrospective every two weeks to see what we can improve in the project. After the plan of approach, we started to work in duos to try machine learning and to explore our data. When we knew more about machine learning and our data we started to work on our final machine learning model that would detect sprint in IMU data. Our findings of our dataset and machine learning models were documented into a research paper and a personal portfolio.|
| Results | The first result of our project was our plan of approach, this helped us to tell our problem owner and ourselves what we were going to do during the project. During the whole project scrum helped us with knowing what everybody was doing, and this gave us the opportunity to intervene if somebody was doing something wrong. With the retrospective we have really improved the workplace and the motivation of every project member.  We as a project group made two predictive models that can detect sprints out of IMU data. We also made a program that uses the two predictive models to compare results and improve our dataset. We documented this all in a research paper.|
| Reflection| We as a project group worked great together and everybody of my project group wanted to help each other and would always make time for each other. This resulted in a good working environment and motivated me to work harder. The scrum method also helped me to know what everybody in my team was doing. But we also had some problems in the group, in the first 10 weeks we had an extra team member that wasn’t doing much. We tried multiple ways to get him to worked but with no success. This resulted that the person quitted the minor because he didn’t do much. The remaining person in the project had only one problem. That problem was we had two person that were quite dominant in what the wanted, it was hard to change their mind (me included). But this was solved by great communication form the other member of the project group. In general, this project group was one of the best groups I had during a project, because of the diversity of everybody’s talents. |

