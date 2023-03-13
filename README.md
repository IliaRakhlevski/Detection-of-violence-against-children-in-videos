# Detection of violence against children in videos
## AFEKA - Tel-Aviv Academic College Of Engineering<br/>Department: Intelligent Systems<br/>Course: Final project

**Language:** Python 3.9 Libraries: keras/tensorflow, pandas, matplotlib, scikit-learn, numpy, opencv-python.<br/>
**Development environments:** Spider 3.3.6, PyCharm 2021.3.1.<br/>
**Cloud platforms:** Google Colaboratory, Kaggle.<br/>

This project proposes a physical violence detecting method based on indoor surveillance cameras.<br/> 
The cameras capture video streams, these streams are classified by using a deep learning Convolutional 
Neural Network (CNN) and Long Short-Term Memory (LSTM) based approach<br/> 
for violence detection by learning the detailed features in videos.<br/> 
The CNN is used for feature extraction and LSTM is used to classify video based on those features.<br/>
As a CNN feature extraction we use ResNet152V2 pre-trained model.<br/>
The dataset presented in the repository has only several examples. The full dataset has about 8000 videos.

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

See *"Final Project Presentation.pdf"* for the details.
