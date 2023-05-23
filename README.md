# Detection of violence against children in videos
## AFEKA - Tel-Aviv Academic College Of Engineering<br/>Department: Intelligent Systems<br/>Course: Final project

**Language:** Python 3.9 Libraries: keras/tensorflow, pandas, matplotlib, scikit-learn, numpy, opencv-python.<br/>
**Deep learning models:** Convolution Neural Network (CNN), Residual Neural Network (ResNet) and Long Short-Term Memory (LSTM)<br/>
**Development environments:** Spider 3.3.6, PyCharm 2021.3.1.<br/>
**Cloud platforms:** Google Colaboratory, Kaggle.<br/>

Violence against children is a world spread phenomenon - it occurs around the world, in every country and every society; 
all too often it happens within the family and in many different forms such as neglection, physical and sexual abuse etc. Violence against children can have dire consequences for the child's physical, emotional and psychosocial development. In this paper we focused on physical violence in kindergarten and its detection. When violence occurs in a Kindergarten it often remains unknown since the child either afraid to tell his parents about it or simply cannot speak yet due to his/her age.

This project proposes a physical violence detecting method based on indoor surveillance cameras. The cameras capture video streams, these streams are
classified by using a deep-learning Convolution Neural Network (CNN) and Long Short-Term Memory (LSTM) based approach for violence detection by learning the
detailed features in the videos. CNN used for feature extraction while LSTM is used to classify the video, based on those features. 
As a CNN feature extraction we use ResNet152V2 pre-trained model.

The novelty of this project is synthesized data. In this project we focused on specific type of violence of adults against children. There are no available data sets match this situation and not enough relevant videos on the Internet, and the existing ones are very low quality. Thus, we decided to use reborn dolls which are very similar to real children.

For the testing purpose we used videos which created under similar conditions as
the videos from the training dataset. We have reached various classification
accuracies up to 70% for “violent” frames of the tested videos. The results of the tests were relatively good, but only for those videos similar to the ones from the dataset. We need to improve the generalization and for this purpose, we have to
increase the training dataset.

Our main task is that the model will learn from the human movement only. To meet this target during each video session, we created videos containing both violent and non-violent movements. The rest of the parameters (background, clothes, distance etc.) remained the same.
Another important goal is to achieve generalization. The trained model must learn to
recognize humans (reborn dolls) on the video and classify their movement under
different conditions: poses, clothes, gender, hairstyle etc. 

After numerous experiments we came to the conclusion that in order to train the model so it can generalize well - there is a need for thousands of videos. The dataset must contain not only synthetically generated videos but also real ones. To continue this work, we need to increase the dataset. 


**Project files:**<br/>
* *settings.py*<br/>
In this module are defined global parameters that are used in several modules.<br/>
List of classification classes, height/width of the processed images etc.<br/>
* *utils.py*<br/>
This module contains some utilities. The function that sets CPU/GPU mode for the model<br/>
training and the function that returns date/time string.<br/>
* *frames_processing.py*<br/>
Implementation of video and frames processing for further training, testing, classification.<br/>
Getting video parameters, loading frames from video files, writing frames to video files,<br/>
writing frames to image files, resizing, padding frames, extracting certain frames, extracting<br/>
specific region from the frames, creating blank frames, creating padded frames sequence –<br/>
adding blank frames to the existing ones, flipping, blurring, reversing, rotating video files for<br/>
data augmentation purpose.<br/>
* *classification.py*<br/>
Implementation of video stream classification.<br/>
Frames sequence classification, windowing, models testing.<br/>
* *conv2d_lstm.py*<br/>
Creating/training a model based on combination of the architectures “Conv2D” and “LSTM”.<br/>
* *Training_run_results.txt*<br/>
Summary of the model training.<br/>
* *Testing_run_results.txt*<br/>
Output of the reference videos testing.<br/>

The code for training is found in the file conv2d_lstm.py. The train dataset must be located in<br/>
the same directory as this file. The dataset directory is called "video_data" and must be found in<br/>
the same directory as the code files. Open the development environment and run this file.<br/>
The code for classification is found in the file classification.py. Be sure that the model file<br/>
conv2d_lstm_model.h5 is located in the same directory. In the function testing set the correct<br/>
path to the directory containing the tested video files. In the “main” function set the lines <br/>
***lst = [i for i in range(1, 120 + 1)] <br/>
testing(model, lst, False)*** <br/>
uncommented. Open the development environment and run this file.<br/>
The dataset presented in the repository has only several examples. The full dataset has about 8000 videos.<br/>

See *"Final Project Presentation.pdf"* for the details.
