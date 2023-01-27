# Creation and training of the model

import keras
from keras.layers import Dense 
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.recurrent import LSTM
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
import datetime
import settings
import utils
import shutil
import frames_processing
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import math



# Training configuration
BATCH_SIZE = 16                 # 16 - my computer, 32 - Google Colab, 64 - Kaggle
NO_EPOCHS = 40                  # number epochs in training

LEARNING_RATE = 0.0001          # basic value - 0.0001, this value is being changed during training
TEST_SIZE = settings.TEST_SIZE
VALIDATION_SPLIT = settings.VALIDATION_SIZE



###########################  CNN + LSTM  #################################

from keras.applications.resnet_v2 import ResNet152V2    


from keras.models import Model
from keras.layers import Input
from keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.layers import Bidirectional


# function that returns the base model for features extraction
def create_resnet152_v2_cnn_base():
    model_path = 'model_data\\resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
    cnn_base = ResNet152V2(input_shape=(settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.CHANNELS),
                     weights=model_path,
                     include_top=False)
    return cnn_base


# model creation
# classes - target (classification) classes
# cnn_base_func - function that returns the base model for features extraction
def create_model(classes, cnn_base_func):
    
    reg_lambda_l1 = 0.000001
    reg_lambda_l2 = 0.000001
    
    video = Input(shape=(settings.SEQ_LEN,
                     settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.CHANNELS))
       
    cnn_base = cnn_base_func()
    
    cnn_out = GlobalAveragePooling2D()(cnn_base.output)
    cnn = Model(inputs=cnn_base.input, outputs=cnn_out)
    cnn.trainable = False
    encoded_frames = TimeDistributed(cnn)(video)

    encoded_sequence = Bidirectional(LSTM(4,

                                      kernel_regularizer=regularizers.l1_l2(l1=reg_lambda_l1, l2=reg_lambda_l2),
                                      recurrent_regularizer=regularizers.l1_l2(l1=reg_lambda_l1, l2=reg_lambda_l2),
                                      bias_regularizer=regularizers.l1_l2(l1=reg_lambda_l1, l2=reg_lambda_l2),
                                      activity_regularizer=regularizers.l1_l2(l1=reg_lambda_l1, l2=reg_lambda_l2)
                                      ))(encoded_frames)

    outputs = Dense(units=1, activation="sigmoid")(encoded_sequence)
    model = Model([video], outputs)
       
    optimizer = Adam(learning_rate=LEARNING_RATE)
    
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["binary_accuracy"])
    
    model.summary()    
    plot_model(model, show_shapes=True, show_layer_names=True)
   
    return model




##########################  data training with training/validation generators ###########################

# custom callback
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, basic_learning_rate):
        super(CustomCallback, self).__init__()
        
        # set learning mode
        self.basic_learning_rate = basic_learning_rate

    # end of the epoch
    def on_epoch_end(self, epoch, logs=None): 
        # save current model after end of each epoch
        filename = 'Models/conv2d_lstm_model_{0}.h5'.format(epoch+1)    
        self.model.save(filename)
        print("\n")

    # begin of the epoch       
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        
        # set learning rate before begining of each epoch         
        keras.backend.set_value(self.model.optimizer.learning_rate, self.basic_learning_rate)

        print("Learning rate is {}".format(self.basic_learning_rate))


# data training using generators
# data_dir - directory containing video data for training
def data_training_with_generators(data_dir):
    
    #  use CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    classes = settings.CLASSES
    
    print("\n\n===================  Loading data  ===================  ")
    
    files_names, targets = frames_processing.get_data(data_dir, classes)
    
#    print("Data:\n")
#    frames_processing.print_data_sequences(files_names, targets)
#    print("---------------------------------------------------------------------------------------------")
    
    
    files_names_testing, targets_testing, files_names_training, targets_training = frames_processing.split_data(files_names, targets, TEST_SIZE)
    
    
    files_names_validation, targets_validation, files_names_training, targets_training = frames_processing.split_data(files_names_training, targets_training, VALIDATION_SPLIT)
    
    
    training_sequence_generator = frames_processing.generate_training_sequences(files_names_training, targets_training, 
                                                                 BATCH_SIZE, 
                                                                 settings.IMG_WIDTH, settings.IMG_HEIGHT, 
                                                                 settings.SEQ_LEN, True, 
                                                                 settings.NUM_SKIPPED_FRAMES)
        
    validation_sequence_generator = frames_processing.generate_validation_sequences(files_names_validation, targets_validation, 
                                                                 BATCH_SIZE,                   
                                                                 settings.IMG_WIDTH, settings.IMG_HEIGHT, 
                                                                 settings.SEQ_LEN, True, 
                                                                 settings.NUM_SKIPPED_FRAMES)
    
    print("\n-----------------------------------------------------------------------------------------------------")
    print("\nTraining:\n")
    frames_processing.print_data_sequences(files_names_training, targets_training)
    print("-------------------------------------------------------------------------------------------------------")
    print("\nValidation:\n" )
    frames_processing.print_data_sequences(files_names_validation, targets_validation)
    print("-------------------------------------------------------------------------------------------------------")
    print("\nTesting:\n")
    frames_processing.print_data_sequences(files_names_testing, targets_testing)  
    print("-----------------------------------------------------------------------------------------------------\n")
    
    
    
    # create new model
    print("\n\n===================  Creating model  ===================  "\n\n")
    model = create_model(classes, create_resnet152_v2_cnn_base)
        
    # load existing model   
#    print("\n\n===================  Loading model  ===================  "\n\n")
#    model = keras.models.load_model('conv2d_lstm_model.h5')
    
    
    custom_callback = CustomCallback(LEARNING_RATE)
    
    callbacks = [custom_callback]
    
    
    print("\n\n===================  Training parameters ============================  "\n\n")
    
    
    NUM_VIOLENCE_SMPLS = frames_processing.get_number_violent_samples(targets_training)
    NUM_NO_VIOLENCE_SMPLS = len(targets_training) - NUM_VIOLENCE_SMPLS
    print("\nViolent training samples:", NUM_VIOLENCE_SMPLS)
    print("\nNon-violent training samples:", NUM_NO_VIOLENCE_SMPLS)
    num_smpls = NUM_VIOLENCE_SMPLS + NUM_NO_VIOLENCE_SMPLS
    violence_class_weights = num_smpls / (NUM_VIOLENCE_SMPLS * len(classes))
    no_violence_class_weights = num_smpls / (NUM_NO_VIOLENCE_SMPLS * len(classes))

    
    class_weights = {0: violence_class_weights, 
                     1: no_violence_class_weights}
           
    print(f'\nviolence_class_weights: {violence_class_weights} / no_violence_class_weights: {no_violence_class_weights}\n')       
    print("Batch size:", BATCH_SIZE)    
    print("\nSteps per epoch:", math.ceil(len(targets_training) / BATCH_SIZE))
    print("\nValidation samples:", len(targets_validation))
    print("\nValidation steps:", math.ceil(len(targets_validation) / BATCH_SIZE))
    print("\nTest samples:", len(targets_testing), "\n")
  
    

    print("\n\n===================  Training ============================  "\n\n")
    
    # it is used in Kaggle only
    #if os.path.exists('Models'):
    #    shutil.rmtree('Models')
    #os.makedirs('Models')
    
    # Fit data to model
    history = model.fit_generator(generator = training_sequence_generator, 
                        validation_data = validation_sequence_generator,
                        steps_per_epoch = math.ceil(len(targets_training) / BATCH_SIZE), 
                        validation_steps = math.ceil(len(targets_validation) / BATCH_SIZE),
                        epochs=NO_EPOCHS,
                        class_weight=class_weights,
                        callbacks=callbacks)
  
    

    print("\n\n===================  Saving model ============================  "\n\n")
    
    #  save the model  
    ts = datetime.datetime.now().timestamp()
    filename = 'Models\\conv2d_lstm_model_{0}.h5'.format(ts)
    model.save(filename)
    
    

    print("\n\n===================  Accuracy and Loss graphs ============================  "\n\n")

    fig = plt.figure()

     # list all data in history
    #print(history.history.keys())
    # summarize history for accuracy
    subplot1 = fig.add_subplot(2, 1, 1)
    subplot1.plot(history.history['binary_accuracy'])
    subplot1.plot(history.history['val_binary_accuracy'])
    subplot1.set_title('model accuracy')
    subplot1.set_ylabel('accuracy')
    subplot1.set_xlabel('epoch')
    subplot1.legend(['train', 'validation'], loc='upper left')

#    plt.show()
    # summarize history for loss
    subplot2 = fig.add_subplot(2, 1, 2)
    subplot2.plot(history.history['loss'])
    subplot2.plot(history.history['val_loss'])
    subplot2.set_title('model loss')
    subplot2.set_ylabel('loss')
    subplot2.set_xlabel('epoch')
    subplot2.legend(['train', 'validation'], loc='upper left')

    fig.tight_layout(pad=3.0)

    plt.show()

    fig.savefig('accuracy_loss.png')


    print("\n\n===================  Loading test samples  ============================  "\n\n")
    
    X_test, y_test = frames_processing.create_testing_sequences(files_names_testing, targets_testing,
                                                                    settings.IMG_WIDTH, settings.IMG_HEIGHT, 
                                                                    settings.SEQ_LEN, True, 
                                                                    settings.NUM_SKIPPED_FRAMES)
    

    print("\n\n===================  Evaluation  ============================  "\n\n")
    
    # Generate generalization metrics
    score = model.evaluate(X_test, y_test, verbose=1)
    print(f'\nTest loss: {score[0]} / Test accuracy: {score[1]}')
    
    
    print("\n\n===================  Prediction  ============================  "\n\n")

    y_pred = model.predict(X_test, verbose=1)
    
    y_pred = [1 * (x[0]>=0.5) for x in y_pred] 
     
    print('\nClassification Report:\n')

    print(classification_report(y_test, y_pred, target_names=classes))
    
    print('\nConfusion Matrix:\n')
    cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    
    cmtx = pd.DataFrame(cm, 
    index=['actual: WithViolence', 'actual: NoViolence'], 
    columns=['predicted: WithViolence', 'predicted: NoViolence']
    )
    print(cmtx)
    print('\n\n')
    
    
    
# train model with generators
def train_data_with_generators():
    
    data_dir = "video_data"
    
    data_training_with_generators(data_dir)
    


    
if __name__ == '__main__':
    
    train_data_with_generators()
    
    