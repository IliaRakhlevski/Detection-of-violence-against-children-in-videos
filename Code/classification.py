# Videos and frames sequences classification

import settings
import frames_processing
import keras
import os
import numpy as np

WIN_CLASS_RES_PRINT = True  # flag indicates if to print the result
FILE_DESC_WRITE = False     # flag indicates if to write into file

file_desc = None            # file descriptor to open file for writing

with_violence_actual = 0
with_violence_correct = 0
with_violence_incorrect = 0
no_violence_actual = 0
no_violence_correct = 0
no_violence_incorrect = 0


# merge intervals
# intervals - list of intervals
def merge_intervals(intervals):
    starts = intervals[:,0]
    ends = np.maximum.accumulate(intervals[:,1])
    valid = np.zeros(len(intervals) + 1, dtype=np.bool)
    valid[0] = True
    valid[-1] = True
    valid[1:-1] = starts[1:] >= ends[:-1]
    return np.vstack((starts[:][valid[:-1]], ends[:][valid[1:]])).T


# classify frames
# frames_list - list of frames to be classified
# model - model is used for frames classification
def frames_classification(frames_list, model):
    frames = np.asarray([frames_list])
    predictions = model.predict(frames)
    predictions = [1 * (x[0]>=0.5) for x in predictions] 
    return predictions[0]
    

# classify video with constant length defined in the "settings" file
# video_name - path to the classified video
# model - model is used for video classification
def const_len_video_classification(video_name, model):
    frames_list = frames_processing.create_data_from_video(video_name, 
                                                      settings.IMG_WIDTH, 
                                                      settings.IMG_HEIGHT,
                                                      settings.SEQ_LEN,
                                                      True, settings.NUM_SKIPPED_FRAMES)
    
    return frames_classification(frames_list, model)


# classify video with windowing
# video_name - path to the classified video
# model - model is used for video classification
# fps - frames per second, for debug purpose
def get_frames_with_violence(video_name, model, fps=30):

    global file_desc
    global FILE_DESC_WRITE

    # original frames from video
    original_frames_list = frames_processing.frames_extraction(video_name, 0, 0, False, 0)
    # preprocessed frames received from the original ones
    preprocessed_frames = frames_processing.frames_preprocessing(original_frames_list, 
                                                                 settings.IMG_WIDTH, 
                                                                 settings.IMG_HEIGHT, 
                                                                 settings.NUM_SKIPPED_FRAMES)
    
    win_pos = 0     # window position
    intervals = []  # list of intervals of frames contain violence
    violence_frames = []

    if FILE_DESC_WRITE == True and file_desc is not None:
        file_desc.write('\n')
    
    # windowing
    while win_pos < len(preprocessed_frames):
        
        # get frames from current window position
        if (win_pos + settings.SEQ_LEN) < len(preprocessed_frames):
            frames = preprocessed_frames[win_pos:(win_pos + settings.SEQ_LEN)]
        else:
            frames = preprocessed_frames[win_pos:]
            blank_frames = frames_processing.create_blank_images((settings.SEQ_LEN - len(frames)), settings.IMG_WIDTH, settings.IMG_HEIGHT)
            frames.extend(blank_frames)
            
        # classify the frames               
        class_ind = frames_classification(frames, model)
        
        
        # for debug only, do not delete
        start = win_pos * (settings.NUM_SKIPPED_FRAMES + 1)
        end = (win_pos + settings.SEQ_LEN) * (settings.NUM_SKIPPED_FRAMES + 1)
        if end > len(original_frames_list):
            end = len(original_frames_list)
        if  WIN_CLASS_RES_PRINT == True:
            win_class_res = "Start - {0}, End - {1}, Class - {2}".format(start, end, settings.CLASSES[class_ind])
            print(win_class_res)
            if FILE_DESC_WRITE == True and file_desc is not None:
                file_desc.write(win_class_res + '\n')

# it is used for debug
#        ch = "___{0}_{1}.mp4".format(start, end)
#        vid_name = video_name.replace(".mp4", ch)        
#        frames_processing.write_frames_to_video(frames, vid_name, fps / (settings.NUM_SKIPPED_FRAMES + 1))
        
     
        # if frames contain violence add their numbers as interval
        if settings.CLASSES[class_ind] == "WithViolence":
            start = win_pos * (settings.NUM_SKIPPED_FRAMES + 1)
            end = (win_pos + settings.SEQ_LEN) * (settings.NUM_SKIPPED_FRAMES + 1)
            if end > len(original_frames_list):
                end = len(original_frames_list)
            intervals.append([start, end])
         #   frames_processing.write_frames_to_files(frames, 'C:/Users/rakhl/Documents/Afeka/Final_Project/Projects/Final/frames_output/test')
 
        if (win_pos + settings.SEQ_LEN) >= len(preprocessed_frames):
            break
        
        win_pos += settings.WIN_STEP
        
    if len(intervals) == 0:
        return violence_frames, intervals
    
    # interval of frames containing violence    
    intervals = np.array(intervals)
    # merge overlapped intervals
    merged_intervals = merge_intervals(intervals)
    
    # create list of frames containing violence taken from original video
    for ind in range(len(merged_intervals)):
        inter = merged_intervals[ind]
        frames = original_frames_list[inter[0]:inter[1]]
        violence_frames.append(frames)
        
    return violence_frames, merged_intervals


# video classification
# video_name - path to the classified video
# model - model is used for video classification
# print_files - indicates if to write "violent" frames to videos
def video_classification(video_name, model, print_files = True):

    global file_desc
    global FILE_DESC_WRITE

    width, height, num_frames, fps = frames_processing.get_video_properties(video_name)
    
    print("\n" + video_name)
    if FILE_DESC_WRITE == True:
        file_desc.write("\n\n\n" + video_name + '\n')

    print("\n --- Width of frames - {0}".format(width))
    print(" --- Height of frames - {0}".format(height))
    print(" --- Number of frames - {0}".format(num_frames))
    print(" --- Frames per second - {0}\n".format(fps))  
    
    # get list of frames containing violence
    violence_frames, intervals = get_frames_with_violence(video_name, model, fps) 
    # write frames to videos
    if print_files == True:
        for ind in range(len(violence_frames)): 
            ch = "_{0}.mp4".format(ind)
            path = video_name.replace(".mp4", ch)
            frames_processing.write_frames_to_video(violence_frames[ind], path, fps)
  
    # list of predicted frames values 0 - non-violent, 1 - violent     
    fr_class_pred = [0] * num_frames
    for ind in range(len(intervals)):
        inter = intervals[ind]
        start = int(inter[0])
        end = int(inter[1])
#        fr_class_pred[start:end+1]
        if end == num_frames:
            end = end - 1
        if start <= end and end < num_frames:
            fr_class_pred[start:end+1] = [1] * ((end + 1) - start)
    
    # list of real frames values 0 - non-violent, 1 - violent 
    fr_class_real = get_frames_classes(video_name, num_frames)
    
    if fr_class_real is not None:
        # calculate IoU for predicted and real values
        res, iou = get_iou(fr_class_pred, fr_class_real)
        
        if res == 0 or res == 1:
            print('IoU: ', round(iou, 2))
            if FILE_DESC_WRITE == True:
                file_desc.write('\nIoU: ' + str(round(iou, 2)))

        test_frames_statistics(fr_class_pred, fr_class_real)



# classify video in the specific directory
# input_dir - target directory
# model - pretrained model
# is_train_videos - if true then it is talked about the videos used for training, their size is 20 frames
def dir_video_classification(input_dir, model, is_train_videos = True):
    files_list = os.listdir(input_dir)

    class_res = [0, 0]
    for f in files_list:
        file_path = os.path.join(input_dir, f)
        print(file_path)
        if is_train_videos == True:
            pred_ind = const_len_video_classification(file_path, model)
            print(settings.CLASSES[pred_ind])
            class_res[pred_ind] += 1
        else:
            video_classification(file_path, model, False)
    print(settings.CLASSES[0], "-", class_res[0], " | ", settings.CLASSES[1], "-", class_res[1])



# read from the file frames values:
# ranges of frames that are "violent"
# and create a list in which each cell
# contains 0 ("non-violent" frame) or ("violent" frame)
def get_frames_classes(video_name, num_frames):
    
    fr_class_val = [0] * num_frames
    ch = "_frames.txt"
    path = video_name.replace(".mp4", ch)
    
    try:
        f = open(path, 'r')
    except IOError:
        print('\nCannot open file:', path)
        return None
             
    Lines = f.readlines()
    
    print('\n\nActual "violent" frames:')
    for line in Lines:
        print("Line: {}".format(line.strip()))
        numbers = line.split()
        if len(numbers) == 2:
            start = int(numbers[0])
            end = int(numbers[1])
            if start <= end and end < num_frames:
                fr_class_val[start:end+1] = [1] * ((end + 1) - start)
    print('\n')
    f.close()
    return fr_class_val


# get intersection over union for "violent" frames
# fr_class_val1 - first list of frames values
# fr_class_val2 - second list of frames values
# in a list each frame has the value:
#   0 - no violence
#   1 - with violence
def get_iou(fr_class_val1, fr_class_val2):
    
    # sizes of two list are different
    if (len(fr_class_val1) != len(fr_class_val2)):
        return (2, 0.0)
    
    # both lists do not contain "1"
    if (1 not in fr_class_val1) and (1 not in fr_class_val2):
        return (1, 0.0)
    
    # find overlap violence frames
    overlap = [0] * len(fr_class_val1)
    
    for i in range(len(fr_class_val1)):
        if fr_class_val1[i] == 1 and fr_class_val2[i] == 1:
            overlap[i] = 1
      
    # find union of violence frames
    union = [0] * len(fr_class_val1)
    
    for i in range(len(fr_class_val1)):
        if fr_class_val1[i] == 1 or fr_class_val2[i] == 1:
            union[i] = 1
            
    overlap_1 = overlap.count(1)
    union_1 = union.count(1)
    
    # IoU = Area of Overlap / Area of Union
    
    return (0, float(overlap_1/union_1))



# get IoU between two videos
# video_name1 - first video
# video_name2 - second video
def test_frames_classes(video_name1, video_name2):
    
    fr_class_val1 = get_frames_classes(video_name1, 10)
    fr_class_val2 = get_frames_classes(video_name2, 10)

    res, iou = get_iou(fr_class_val1, fr_class_val2)
    
    print(res, iou)


# get statistics in comparing between two lists containing
# information about frames types (violen/non-violent) 
# frames_class_pred - predicted frames values
# frames_class_real - real frames values 
def test_frames_statistics(frames_class_pred, frames_class_real):

    global file_desc
    global FILE_DESC_WRITE
    global with_violence_actual
    global with_violence_correct
    global with_violence_incorrect
    global no_violence_actual
    global no_violence_correct
    global no_violence_incorrect

    # sizes of two list are different
    if (len(frames_class_pred) != len(frames_class_real)):
        return

    with_viol_actual = 0
    with_viol_true = 0
    with_viol_false = 0

    no_viol_actual = 0
    no_viol_true = 0
    no_viol_false = 0

    for i in range(len(frames_class_pred)):

        if frames_class_real[i] == 1:
            with_viol_actual += 1

        if frames_class_real[i] == 0:
            no_viol_actual += 1

        if frames_class_pred[i] == 1 and frames_class_real[i] == 1:
            with_viol_true += 1
        elif frames_class_pred[i] == 0 and frames_class_real[i] == 0:
            no_viol_true += 1
        elif frames_class_pred[i] == 1 and frames_class_real[i] == 0:
            no_viol_false += 1
        elif frames_class_pred[i] == 0 and frames_class_real[i] == 1:
            with_viol_false += 1

    print("\nWith Violence (actual):", with_viol_actual)
    if with_viol_actual > 0:
        print("Correct predicted: ", with_viol_true, "(", round(with_viol_true / with_viol_actual * 100, 2), "%) | Incorrect predicted:",
          with_viol_false, "(", round(with_viol_false / with_viol_actual * 100, 2), "%)")
    else:
        print("Correct predicted: ", with_viol_true, " (100%) | Incorrect predicted:", with_viol_false, "(0%)")

    print("\nNo Violence (actual):", no_viol_actual)
    if no_viol_actual > 0:
        print("Correct predicted: ", no_viol_true, "(", round(no_viol_true / no_viol_actual * 100, 2), "%) | Incorrect predicted:",
          no_viol_false, "(", round(no_viol_false / no_viol_actual * 100, 2), "%)")
    else:
        print("Correct predicted: ", no_viol_true, " (100%) | Incorrect predicted:", no_viol_false, "(0%)")

    with_violence_actual += with_viol_actual
    with_violence_correct += with_viol_true
    with_violence_incorrect += with_viol_false
    no_violence_actual += no_viol_actual
    no_violence_correct += no_viol_true
    no_violence_incorrect += no_viol_false

    if FILE_DESC_WRITE == True:
        file_desc.write("\n\nWith Violence (actual): " + str(with_viol_actual))
        if with_viol_actual > 0:
            file_desc.write("\n     Correct predicted: " + str(with_viol_true) + " (" + str(round(with_viol_true / with_viol_actual * 100, 2)) + " %)")
            file_desc.write("\n     Incorrect predicted: " + str(with_viol_false) + " (" + str(round(with_viol_false / with_viol_actual * 100, 2)) + " %)")
        else:
            file_desc.write("\n     Correct predicted: " + str(with_viol_true) + " (100 %)")
            file_desc.write("\n     Incorrect predicted: " + str(with_viol_false) + " (0 %)")

        file_desc.write("\n\nNo Violence (actual): " + str(no_viol_actual))
        if no_viol_actual > 0:
            file_desc.write("\n     Correct predicted: " + str(no_viol_true) + " (" + str(round(no_viol_true / no_viol_actual * 100, 2)) + " %)")
            file_desc.write("\n     Incorrect predicted: " + str(no_viol_false) + " (" + str(round(no_viol_false / no_viol_actual * 100, 2)) + " %)")
        else:
            file_desc.write("\n     Correct predicted: " + str(no_viol_true) + " (100 %)")
            file_desc.write("\n     Incorrect predicted: " + str(no_viol_false) + " (0 %)")


# perform testing
# model - the loaded model
# lts - list of tests (files names)
# path - path to the directory containing the tests
# print_files - indicates if to write "violent" frames to videos
def testing(model, lst, path, print_files = True):

    global file_desc
    global FILE_DESC_WRITE
    global with_violence_actual
    global with_violence_correct
    global with_violence_incorrect
    global no_violence_actual
    global no_violence_correct
    global no_violence_incorrect


    FILE_DESC_WRITE = True
    file_desc = open('tests_results.txt', 'w')

    for i in lst:
        video_name = path + '/Test_{0}.mp4'.format(i)
        video_classification(video_name, model, print_files)
        print("\n\n")

    file_desc.write("\n\n\nSUMMARY:")
    file_desc.write("\n\nWith Violence (actual): " + str(with_violence_actual))
    if with_violence_actual > 0:
        file_desc.write("\n     Correct predicted: " + str(with_violence_correct) + " (" + str(
            round(with_violence_correct / with_violence_actual * 100, 2)) + " %)")
        file_desc.write("\n     Incorrect predicted: " + str(with_violence_incorrect) + " (" + str(
            round(with_violence_incorrect / with_violence_actual * 100, 2)) + " %)")
    else:
        file_desc.write("\n     Correct predicted: " + str(with_violence_correct) + " (100 %)")
        file_desc.write("\n     Incorrect predicted: " + str(with_violence_incorrect) + " (0 %)")

    file_desc.write("\n\nNo Violence (actual): " + str(no_violence_actual))
    if no_violence_actual > 0:
        file_desc.write("\n     Correct predicted: " + str(no_violence_correct) + " (" + str(
            round(no_violence_correct / no_violence_actual * 100, 2)) + " %)")
        file_desc.write("\n     Incorrect predicted: " + str(no_violence_incorrect) + " (" + str(
            round(no_violence_incorrect / no_violence_actual * 100, 2)) + " %)")
    else:
        file_desc.write("\n     Correct predicted: " + str(no_violence_correct) + " (100 %)")
        file_desc.write("\n     Incorrect predicted: " + str(no_violence_incorrect) + " (0 %)")

    file_desc.close()
    FILE_DESC_WRITE = False



if __name__ == '__main__':
# used for testing

#  use CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load the model if exists
    model = keras.models.load_model('conv2d_lstm_model.h5')

# all tests
    lst = [i for i in range(1, 120 + 1)]
# synthetically created
#    lst = [i for i in range(1, 5 + 1)] + [39, 48] + [i for i in range(85, 110 + 1)]
# real with violence
#    lst = [i for i in range(6, 13 + 1)] + [46, 47, 49, 50] + [i for i in range(65, 76 + 1)]
# no humans
#    lst = [57, 58, 59]
# real no violence
#    lst = [i for i in range(14, 38 + 1)] + [i for i in range(40, 45 + 1)] + [i for i in range(51, 56 + 1)] + \
#          [i for i in range(60, 64 + 1)] + [i for i in range(77, 84 + 1)] + [i for i in range(111, 120 + 1)]

# testing
    path = 'C:/Users/rakhl/Documents/Afeka/Final_Project/Videos/Test'
    testing(model, lst, path, False)

# example of specific file testing with creation of videos containing recognized "violent" scenes
#    video_name = 'C:/Users/rakhl/Documents/Afeka/Final_Project/Videos/Test/Test_8.mp4'
#    video_classification(video_name, model, True)


# example of classification of the files found in the specific directory
#    inp_dir = 'C:/Users/rakhl/PycharmProjects/FinalProject/video_data/WithViolence'
#    inp_dir = 'C:/Users/rakhl/PycharmProjects/FinalProject/video_data/NoViolence'
#    dir_video_classification(inp_dir, model)


    
  
    
    
    
    
    
    
    
    