# Detection of violence against children in videos
## AFEKA - Tel-Aviv Academic College Of Engineering<br/>Department: Intelligent Systems<br/>Course: Final project

**Language:** Python 3.9 Libraries: keras/tensorflow, pandas, matplotlib, scikit-learn, numpy, opencv-python.
**Development environments:** Spider 3.3.6, PyCharm 2021.3.1.
**Cloud platforms:** Google Colaboratory, Kaggle.

This project proposes a physical violence detecting method based on indoor surveillance cameras.<br/> 
The cameras capture video streams, these streams are classified by using a deep learning Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM)<br/> 
based approach for violence detection by learning the detailed features in videos.<br/> 
The CNN is used for feature extraction and LSTM is used to classify video based on those features.<br/>
As a CNN feature extraction we use ResNet152V2 pre-trained model.<br/>
The dataset presented in the repository has only several examples. The full dataset has about 8000 videos.
