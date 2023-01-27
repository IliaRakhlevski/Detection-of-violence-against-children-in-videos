# Loading/writing frames from/into videos
# Processing frames

from keras.applications.resnet_v2 import preprocess_input


import cv2
import imutils
import os
import numpy as np
import settings
import random

BLACK = [0,0,0]     # black color
PAD_COLOR = BLACK   # color of the added pads
ROT_ANGLE = 15      # frame rotation angle
BLUR_VAL = (50, 50) # blurring value



# Add pads to image in order it to have the given size (img_width, img_height)
# image - image (frame) object
# img_width - final image width
# img_height - final image height
def image_padding(image, img_width, img_height):
    im_cur_height = image.shape[0]
    im_cur_width = image.shape[1]
    
    top = 0
    bottom = 0
    left = 0
    right = 0
    
    if im_cur_height < img_height:
        deltha = (int)((img_height - im_cur_height) // 2)
        top = deltha
        bottom = deltha
        if (top + bottom + im_cur_height) < img_height:
            top += 1
        
    if im_cur_width < img_width:
        deltha = (int)((img_width - im_cur_width) // 2)
        left = deltha
        right = deltha
        if (left + right + im_cur_width) < img_width:
            left += 1
    
    pad_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_COLOR)
    
    return pad_image



# Change the image in order it to have the given size (img_width, img_height)
# image - image (frame) object
# img_width - final image width
# img_height - final image height
def frame_preprocessing(image, img_width, img_height):
    im_cur_height = image.shape[0]
    im_cur_width = image.shape[1]
    
    height_factor = im_cur_height / img_height
    width_factor  = im_cur_width  / img_width
    
    factor = width_factor
    if height_factor > width_factor:
        factor = height_factor
        
    im_cur_height = int(im_cur_height / factor)
    im_cur_width  = int(im_cur_width  / factor)
    
    image = cv2.resize(image, (im_cur_width, im_cur_height))
    
    image = image_padding(image, img_width, img_height)
    
    image = preprocess_input(image)
    
    return image



# Extract frames from the given video, preprocess them and put into list
# video_path - path to the video
# img_width - final frame width (if 'is_preprocess' is True)
# img_height - final frame height (if 'is_preprocess' is True)
# is_preprocess - is the frames must be preprocessed
# skipped_frames - how many frames must be skipped between frames that are 
#                 preprocessed and put into the list
def frames_extraction(video_path, img_width, img_height, is_preprocess = True, skipped_frames = 0):
    frames_list = []
     
    vidObj = cv2.VideoCapture(video_path)
    if not vidObj.isOpened():
        raise IOError("Couldn't open video stream")
    
    # number of frames in the video
    num_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    #print("Number of frames - {0}".format(num_frames))
    
    FramesCount = 0             # counter of the frames
    SkippedFramesCount = 0      # counter of the skipped frames
 
    while FramesCount < num_frames: 
         
        success, image = vidObj.read() 
        if success:
            
            if SkippedFramesCount % (skipped_frames + 1) == 0:
                
                SkippedFramesCount = 0
            
                if is_preprocess == True:
                    image = frame_preprocessing(image, img_width, img_height)
                
                # write frame to file - for DEBUG only!!!!
                #cv2.imwrite('frames_output/image_out_' + str(FramesCount) + '.jpg', image)
                    
                frames_list.append(image)
                
            FramesCount += 1
            SkippedFramesCount += 1
        else:
            print("Defected frame")
            break
            
    vidObj.release()
         
    return frames_list



# Resize frames, skip some frames from the given frames list
# frames_list - list of the frames
# img_width - final frame width (if 'is_preprocess' is True)
# img_height - final frame height (if 'is_preprocess' is True)
# skipped_frames - how many frames must be skipped between frames that are 
#                 preprocessed and put into the list
def frames_preprocessing(frames_list, img_width, img_height, skipped_frames):
    new_frames_list = []

    FramesCount = 0             # counter of the frames
    SkippedFramesCount = 0      # counter of the skipped frames
    
    for frame in frames_list:     
        if SkippedFramesCount % (skipped_frames + 1) == 0:
            
            SkippedFramesCount = 0
            image = frame_preprocessing(frame, img_width, img_height)
            
            # write frame to file - for DEBUG only!!!!
            #cv2.imwrite('frames_output/image_out_' + str(FramesCount) + '.jpg', image)
                
            new_frames_list.append(image)
            
        FramesCount += 1
        SkippedFramesCount += 1
    
    return new_frames_list



# get some video properties:
# frame width/height, frames per second, number of frames
# video_path - path to the video
def get_video_properties(video_path):
    vidObj = cv2.VideoCapture(video_path)
    if not vidObj.isOpened():
        raise IOError("Couldn't open video stream")
        
    # frames width   
    width  = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # frames height
    height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # number of frames in the video
    num_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
       
    # frames per second
    fps = int(round(vidObj.get(cv2.CAP_PROP_FPS)))  
    
    vidObj.release()
    
    return width, height, num_frames, fps 



# write frames to files (one frame to one file)
# frames_list - list of the frames
# path - path of created files
def write_frames_to_files(frames_list, path):   
    FramesCount = 0
    for frame in frames_list:
        cv2.imwrite(path + '/image_out_' + str(FramesCount) + '.jpg', frame)
        FramesCount += 1
 
 
    
# write frames to video  
# frames_list - list of the frames
# path - path of the video
# fps - frames per second
def write_frames_to_video(frames_list, path, fps):
    if len(frames_list) == 0:
        return
    height = frames_list[0].shape[0]
    width = frames_list[0].shape[1]
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video=cv2.VideoWriter(path, fourcc, fps, (width,height))
    for frame in frames_list:
        video.write(frame)
    video.release()



# extract new frames with the given region from the given frames list
# frames_list - original list of the frame
# x - start of the region in x-axe
# y - start of the region in y-axe
# width - width of the region
# height - width of the region
def get_region(frames_list, x, y, width, height):
    if len(frames_list) == 0:
        return
    new_frames = []
    for frame in frames_list:
        ROI = frame[y:(y+height), x:(x+width)].copy()
        new_frames.append(ROI)
    return new_frames



# create blank image with the given parameters
# width - frame width 
# height - frame height
# rgb_color - color of all the pixels
def create_blank_image(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image



# create blank images sequence with the given parameters
# num_images - number of images
# width - frame width 
# height - frame height
# rgb_color - color of all the pixels
def create_blank_images(num_images, width, height, rgb_color=(0, 0, 0)):
    new_frames = []
    
    for i in range(num_images):
        image = create_blank_image(width, height, rgb_color)
        new_frames.append(image)
        
    return new_frames



# create padded frames sequences
# frames - list of frames
# width - frame width
# height -frame height
# seq_len - length of sequence
def create_padded_sequences(frames, width, height, seq_len):
    pad_seq = []
    
    if len(frames) < seq_len:
        num_seq = seq_len // len(frames) + 1
        step = (seq_len - len(frames)) // (num_seq - 1)
        
        for i in range(0, num_seq):
            start = step * i
            end = start + len(frames)
            
            head = create_blank_images(start, width, height)
            tail = create_blank_images((seq_len - end), width, height)
            
            seq = head + frames + tail
            pad_seq.append(seq)
    
    return pad_seq



# create sequences of padded videos with the same length
# video_path - path to the video
def create_sequence_padded_videos(video_path):

    width, height, num_frames, fps = get_video_properties(video_path)
    
    frames_list = frames_extraction(video_path, 
                                  settings.IMG_WIDTH, 
                                  settings.IMG_HEIGHT, 
                                  False, 0)
    
    padded_seq = create_padded_sequences(frames_list, width, height, settings.VID_LEN)
    
    i = 0
    for seq in padded_seq:
        ch = "_padded_{0}.mp4".format(i)
        new_path = video_path.replace(".mp4", ch)
        write_frames_to_video(seq, new_path, fps)
        i += 1



# extract frames from the given video and create a new video from these frames
# video_path - path to the original video
# intervals_list - list of intervals (start frame : end frame)
def extract_video_from_video(video_path, intervals_list):
    width, height, num_frames, fps = get_video_properties(video_path)
    
    frames_list = frames_extraction(video_path, width, height, False, 0)
    
    i = 0
    for interval in intervals_list:
        start = interval[0]
        end = interval[1]
     
        new_video_frames = frames_list[start : end + 1]
        ch = "_extracted_{0}_{1}.mp4".format(start, end)
        new_path = video_path.replace(".mp4", ch)
        write_frames_to_video(new_video_frames, new_path, fps)
        i += 1



# blur a video
# video_path - path to the video
def video_blur(video_path):
    frames_list = [] # list of flipped frames
    
    vidObj = cv2.VideoCapture(video_path)
    if not vidObj.isOpened():
        raise IOError("Couldn't open video stream")
    
    # number of frames in the video
    num_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    	
    # frames per second
    fps = int(round(vidObj.get(cv2.CAP_PROP_FPS)))  
	
    FramesCount = 0             # counter of the frames
     
    while FramesCount < num_frames: 
         
        success, image = vidObj.read() 
        if success:
    			
            #blur the image
            blured_image = cv2.blur(image, BLUR_VAL)
            
            # add flipped image to the list
            frames_list.append(blured_image)
            
            FramesCount += 1
     
        else:
            print("Defected frame")
            break
            
    vidObj.release()
    
    # create name of new video file
    video_path = video_path.replace(".mp4", "_b.mp4")
    # write new frames to video file	
    write_frames_to_video(frames_list, video_path, fps)



# rotate a video
# video_path - path to the video
def video_rotate(video_path, angle = 0):
    frames_list = [] # list of flipped frames
    
    vidObj = cv2.VideoCapture(video_path)
    if not vidObj.isOpened():
        raise IOError("Couldn't open video stream")
    
    # number of frames in the video
    num_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    	
    # frames per second
    fps = int(round(vidObj.get(cv2.CAP_PROP_FPS)))  
	
    FramesCount = 0             # counter of the frames
     
    while FramesCount < num_frames: 
         
        success, image = vidObj.read() 
        if success:
    			
            # horizontal flip of the image
            rotated_image = imutils.rotate(image, angle)
            
            # add flipped image to the list
            frames_list.append(rotated_image)
            
            FramesCount += 1
     
        else:
            print("Defected frame")
            break
            
    vidObj.release()
    
    # create name of new video file
    if angle >= 0:
        video_path = video_path.replace(".mp4", "_rccw.mp4")
    else:
        video_path = video_path.replace(".mp4", "_rcw.mp4")    
    # write new frames to video file	
    write_frames_to_video(frames_list, video_path, fps)



# reverse video
# video_path - path to the video
def video_reverse(video_path):
    frames_list = [] # list of flipped frames
    
    vidObj = cv2.VideoCapture(video_path)
    if not vidObj.isOpened():
        raise IOError("Couldn't open video stream")
    
    # number of frames in the video
    num_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    	
    # frames per second
    fps = int(round(vidObj.get(cv2.CAP_PROP_FPS)))  
	
    FramesCount = 0             # counter of the frames
     
    while FramesCount < num_frames: 
         
        success, image = vidObj.read() 
        if success:
    			
            # add flipped image to the list
            frames_list.append(image)
            
            FramesCount += 1
     
        else:
            print("Defected frame")
            break
            
    vidObj.release()
    
    # create name of new video file
    video_path = video_path.replace(".mp4", "_r.mp4")
    
    # reverse frames
    frames_list.reverse()
    
    # write new frames to video file	
    write_frames_to_video(frames_list, video_path, fps)
    
  


# horizontal flip of a video
# video_path - path to the video
def video_flip(video_path):
    frames_list = [] # list of flipped frames
    
    vidObj = cv2.VideoCapture(video_path)
    if not vidObj.isOpened():
        raise IOError("Couldn't open video stream")
    
    # number of frames in the video
    num_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    	
    # frames per second
    fps = int(round(vidObj.get(cv2.CAP_PROP_FPS)))  
	
    FramesCount = 0             # counter of the frames
     
    while FramesCount < num_frames: 
         
        success, image = vidObj.read() 
        if success:
    			
            # horizontal flip of the image
            flipped_image = cv2.flip(image, 1)
            # add flipped image to the list
            frames_list.append(flipped_image)
            
            FramesCount += 1
     
        else:
            print("Defected frame")
            break
            
    vidObj.release()
    
    # create name of new video file
    video_path = video_path.replace(".mp4", "_f.mp4")
    # write new frames to video file	
    write_frames_to_video(frames_list, video_path, fps)
    
 

 
# reverse of all the videos in the given directory
# input_dir - directory containing videos
def reverse_video_dir(input_dir):
    files_list = os.listdir(input_dir)

    for f in files_list:
        file_path = os.path.join(input_dir, f)
        video_reverse(file_path)
        

    
# horizontal flip of all the videos in the given directory
# input_dir - directory containing videos
def flip_video_dir(input_dir):
    files_list = os.listdir(input_dir)

    for f in files_list:
        file_path = os.path.join(input_dir, f)
        video_flip(file_path)
        
  
  
# blur of all the videos in the given directory
# input_dir - directory containing videos
def blur_video_dir(input_dir):
    files_list = os.listdir(input_dir)

    for f in files_list:
        file_path = os.path.join(input_dir, f)
        video_blur(file_path)

 
 
# rotate of all the videos in the given directory
# input_dir - directory containing videos
def rotate_video_dir(input_dir):
    files_list = os.listdir(input_dir)

    for f in files_list:
        file_path = os.path.join(input_dir, f)
        video_rotate(file_path, 15)
        video_rotate(file_path, -15)
      
 
 
# rename all the file in the given directory
# input_dir - directory containing videos
# base_name - base name of the files
def rename_files_video_dir(input_dir, base_name):
    
    new_file = ""
    i = 0
    files_list = os.listdir(input_dir)
    random.shuffle(files_list)
    
    for f in files_list:
        old_file_path = os.path.join(input_dir, f)
        if len(base_name) > 0 :
            new_file = "{0}_{1}.mp4".format(base_name, i)
        else:
            new_file = "{0}.mp4".format(i)
        new_file_path = os.path.join(input_dir, new_file)
        os.rename(old_file_path, new_file_path)
        i += 1
        
        
        
# rename all the file in the given directory
# input_dir - directory containing videos
# source_name - part of the file name must be changed
# dest_name - new string that is set instead of the "source_name" part
def rename_part_files_video_dir(input_dir, source_name, dest_name ):
    
    new_file = ""

    files_list = os.listdir(input_dir)
    random.shuffle(files_list)
    
    for f in files_list:
        old_file_path = os.path.join(input_dir, f)
        new_file = f.replace(source_name, dest_name)
        new_file_path = os.path.join(input_dir, new_file)
        os.rename(old_file_path, new_file_path)

       

# create data for training/testing
# input_dir - directory containing video data
# classes - list of classes for classification
# img_width - final frame width (if 'is_preprocess' is True)
# img_height - final frame height (if 'is_preprocess' is True)
# seq_len - length of sequence
# is_preprocess - is the frames must be preprocessed
# skipped_frames - how many frames must be skipped between frames that are 
#                 preprocessed and put into the list    
def create_data(input_dir, classes, img_width, img_height, seq_len, is_preprocess = False, skipped_frames = 0):
    X = []  # video data - list of video sequences
    Y = []  # list of classes for classification
     
    #classes_list = os.listdir(input_dir)
    
    for c in classes:
        print(c)
        files_list = os.listdir(os.path.join(input_dir, c))
        #random.seed(0)
        random.shuffle(files_list)
        print(files_list)
        for f in files_list:
            frames = frames_extraction(os.path.join(os.path.join(input_dir, c), f), img_width, img_height, is_preprocess, skipped_frames)
            
            # if number of frames larger than defined length of sequence => truncate number of frames
            if len(frames) > seq_len:
                frames = frames[0:seq_len]
            # if number of frames smaller than defined length of sequence => add blank frames
            elif len(frames) < seq_len:
                blank_frames = create_blank_images((seq_len - len(frames)), img_width, img_height)
                frames.extend(blank_frames)
            # add frames to the video data
            if len(frames) == seq_len:
                X.append(frames)             
                y = [0]*len(classes)
                y[classes.index(c)] = 1
                Y.append(y)

    return X, Y


# get list of files and their targets for each class from the input directory
# input_dir - directory containing video data
# classes - list of classes for classification
# shuffle - if the lists must be shuffled
def get_data(input_dir, classes, shuffle = True):
    
    files_names = []
    targets = []
    
    for c in classes:
        #print(c)
        files_list = os.listdir(os.path.join(input_dir, c))
        
        if shuffle == True:
            random.shuffle(files_list)
        #print(files_list)
        
        for f in files_list:
            files_names.append(os.path.join(os.path.join(input_dir, c), f))             
            y = classes.index(c)
            targets.append(y)
    
    if shuffle == True:
        temp = list(zip(files_names, targets))
        random.shuffle(temp)
        files_names, targets = zip(*temp)
        files_names = list(files_names)
        targets = list(targets)

    return files_names, targets


# split the lists of files (and targets) into two parts
# files_names - list of files
# targets - list of targets for the list 'files_names'
# factor - size of the new lists (factor/1-factor) relative to the original one
def split_data(files_names, targets, factor = 0.1):
    
    data_len = len(files_names)
    ind = int(data_len * factor)
    
    files_names_first_half = files_names[:ind].copy()
    files_names_second_half = files_names[ind:].copy()
    targets_first_half = targets[:ind].copy()
    targets_second_half = targets[ind:].copy()
    
    return files_names_first_half, targets_first_half, files_names_second_half, targets_second_half

  
  
# generate training frames sequences from the list of files names
# files_names - list of files
# targets - list of targets for the list 'files_names' 
# batch_size - size of batch during training
# img_width - image width
# img_height - image height
# seq_len - length of one sequence
# is_preprocess - if must be preprocessed (some action like: resizing, rescaling etc.)
# skipped_frames - number of of frames that must be skipped during sequence creation
def generate_training_sequences(files_names, targets, 
                                batch_size, 
                                img_width, img_height, 
                                seq_len, is_preprocess = False, 
                                skipped_frames = 0):
  
    X = []  # video data - list of video sequences
    Y = []  # list of classes for classification
    count = 0
    
    while True:
        for i in range(len(files_names)):
            
#            print("Train -----------------------------------------------------------------------------------------------------")
#            print(i, files_names[i], targets[i])    
                   
            frames = frames_extraction(files_names[i], 
                                       img_width, img_height, 
                                       is_preprocess, skipped_frames)
            
             # if number of frames larger than defined length of sequence => truncate number of frames
            if len(frames) > seq_len:
                frames = frames[0:seq_len]
            # if number of frames smaller than defined length of sequence => add blank frames
            elif len(frames) < seq_len:
                blank_frames = create_blank_images((seq_len - len(frames)), img_width, img_height)
                frames.extend(blank_frames)
                
            # add frames to the video data
            if len(frames) == seq_len:
                X.append(frames) 
                Y.append(targets[i])
            
            count += 1
                
            if count == batch_size or i == len(files_names)-1:

                X = np.asarray(X)
                Y = np.asarray(Y)
#                print("Train batch -----------------------------------------------------------------------------------------------------\n")
    
                yield X, Y

                count = 0
                X = [] 
                Y = [] 
       
        
# generate validation frames sequences from the list of files names
# files_names - list of files
# targets - list of targets for the list 'files_names' 
# batch_size - size of batch during validation
# img_width - image width
# img_height - image height
# seq_len - length of one sequence
# is_preprocess - if must be preprocessed (some action like: resizing, rescaling etc.)
# skipped_frames - number of of frames that must be skipped during sequence creation        
def generate_validation_sequences(files_names, targets, 
                                    batch_size,
                                    img_width, img_height, 
                                    seq_len, is_preprocess = False, 
                                    skipped_frames = 0):
  
    X = []  # video data - list of video sequences
    Y = []  # list of classes for classification
    count = 0
    
    while True:
        for i in range(len(files_names)):
            
#            print("Valid -----------------------------------------------------------------------------------------------------")
#            print(i, files_names[i], targets[i])      
            frames = frames_extraction(files_names[i], 
                                       img_width, img_height, 
                                       is_preprocess, skipped_frames)
            
             # if number of frames larger than defined length of sequence => truncate number of frames
            if len(frames) > seq_len:
                frames = frames[0:seq_len]
            # if number of frames smaller than defined length of sequence => add blank frames
            elif len(frames) < seq_len:
                blank_frames = create_blank_images((seq_len - len(frames)), img_width, img_height)
                frames.extend(blank_frames)
                
            # add frames to the video data
            if len(frames) == seq_len:
                X.append(frames) 
                Y.append(targets[i])
            
            count += 1
                
            if count == batch_size or i == len(files_names)-1:

                X = np.asarray(X)
                Y = np.asarray(Y)
 #               print("Valid batch -----------------------------------------------------------------------------------------------------\n")
    
                yield X, Y

                count = 0
                X = [] 
                Y = [] 
 


# generate testing frames sequences from the list of files names
# files_names - list of files
# targets - list of targets for the list 'files_names' 
# img_width - image width
# img_height - image height
# seq_len - length of one sequence
# is_preprocess - if must be preprocessed (some action like: resizing, rescaling etc.)
# skipped_frames - number of of frames that must be skipped during sequence creation   
def create_testing_sequences(files_names, targets, 
                            img_width, img_height, 
                            seq_len, is_preprocess = False, 
                            skipped_frames = 0):
  
    X = []  # video data - list of video sequences
    Y = []  # list of classes for classification
    
    for i in range(len(files_names)):
               
        frames = frames_extraction(files_names[i], 
                                   img_width, img_height, 
                                   is_preprocess, skipped_frames)
        
        # if number of frames larger than defined length of sequence => truncate number of frames
        if len(frames) > seq_len:
            frames = frames[0:seq_len]
        # if number of frames smaller than defined length of sequence => add blank frames
        elif len(frames) < seq_len:
            blank_frames = create_blank_images((seq_len - len(frames)), img_width, img_height)
            frames.extend(blank_frames)
            
        # add frames to the video data
        if len(frames) == seq_len:
            X.append(frames) 
            Y.append(targets[i])
        
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y 



# print data sequence
# files_names - full paths list
# targets - violent/non-violent
def print_data_sequences(files_names, targets): 
    
    if len(files_names) != len(targets):
        print("sizes of the files_names and of the targets are different")
        return
    
    print("Number of items:", len(files_names), "\n")
    
    for i in range(len(files_names)):
        print(files_names[i], "-", targets[i])
    print("\n") 



# count number of "violent" samples in the "targets" list
# targets - list of targets (each value 0(WithViolence)/1(NoViolence))
def get_number_violent_samples(targets):

    count = 0
    
    for i in range(len(targets)):
        val = targets[i]
        if settings.CLASSES[val] == "WithViolence":
            count += 1
    
    return count
                
 
 
# loading frame from video and their preparation for classification
# video_path - path to the video
# img_width - final frame width (if 'is_preprocess' is True)
# img_height - final frame height (if 'is_preprocess' is True)
# seq_len - length of sequence
# is_preprocess - is the frames must be preprocessed
# skipped_frames - how many frames must be skipped between frames that are 
#                 preprocessed and put into the list  
def create_data_from_video(video_path, 
                           img_width, img_height, 
                           seq_len, 
                           is_preprocess = False, 
                           skipped_frames = 0):  
    
    frames = frames_extraction(video_path, img_width, img_height, is_preprocess, skipped_frames)
    if len(frames) > seq_len:
        frames = frames[0:seq_len]
    # if number of frames smaller than defined length of sequence => add blank frames
    elif len(frames) < seq_len:
        blank_frames = create_blank_images((seq_len - len(frames)), img_width, img_height)
        frames.extend(blank_frames)       
        
    return frames
 


# extract videos from videos with constant length and constant step
# these parameters are defined in the file "settings"
# after extraction they are written to a new video
def extract_videos_from_videos_with_len_and_step(video_name, fps=30):
    
    original_frames_list = frames_extraction(video_name, 0, 0, False, 0)
    
    win_pos = 0     # window position
    
    while win_pos < len(original_frames_list):
        
        if (win_pos + settings.SEQ_LEN) < len(original_frames_list):
            frames = original_frames_list[win_pos:(win_pos + settings.SEQ_LEN)]
        else:
            frames = original_frames_list[win_pos:]
            blank_frames = create_blank_images((settings.SEQ_LEN - len(frames)), 
                                               settings.IMG_WIDTH, settings.IMG_HEIGHT)
            frames.extend(blank_frames)
         
        start = win_pos * (settings.NUM_SKIPPED_FRAMES + 1)
        end = (win_pos + settings.SEQ_LEN) * (settings.NUM_SKIPPED_FRAMES + 1)
        if end > len(original_frames_list):
            end = len(original_frames_list)
            
        ch = "___{0}_{1}.mp4".format(start, end)
        vid_name = video_name.replace(".mp4", ch)        
        write_frames_to_video(frames, vid_name, fps / (settings.NUM_SKIPPED_FRAMES + 1))
            
        if (win_pos + settings.SEQ_LEN) >= len(original_frames_list):
            break
        
        win_pos += settings.WIN_STEP

 

# extract videos from videos with constant length and constant step  
# this function is used to divide a video file into many small videos
# with with constant length and constant step
def temp_create_videos_with_len_and_step():
    
    video_name = '1.mp4'           
    path = 'C:/Users/rakhl/Documents/Afeka/Final_Project/Videos/Creation/Current/'
#    path = 'C:/Users/rakhl/Documents/Afeka/Final_Project/Videos/Frames/'
    
    extract_videos_from_videos_with_len_and_step(path + video_name)
        

        


if __name__ == '__main__':

# run it to divide a file into many small videos with with constant length and constant step
# these parameter are defined in the file "settings"
    #temp_create_videos_with_len_and_step()
    
 
# rename, flip, blur, reverse files in certain directories
#    rename_files_video_dir('C:/Users/rakhl/Documents/Afeka/Final_Project/Videos/video_data/NoViolence', "video")
#    rename_files_video_dir('C:/Users/rakhl/Documents/Afeka/Final_Project/Videos/video_data/WithViolence', "video")
#    flip_video_dir('C:/Users/rakhl/Documents/Afeka/Final_Project/Videos/video_data/NoViolence')
#    flip_video_dir('C:/Users/rakhl/Documents/Afeka/Final_Project/Videos/video_data/WithViolence')
#    reverse_video_dir('C:/Users/rakhl/Documents/Afeka/Final_Project/Videos/video_data/NoViolence')
    


    

    
    