# Global parameters


# size of processed images (frames) extracted from the video
IMG_HEIGHT = int(224) 
IMG_WIDTH  = int(224)

# test size
TEST_SIZE = 0.01

# validation size
VALIDATION_SIZE = 0.01

# number of channels
CHANNELS = 3

# Classification classes
CLASSES = ["WithViolence", "NoViolence"]

# number of skipped frames
NUM_SKIPPED_FRAMES = 0

# sequence lenght 
SEQ_LEN = 20

# windowing step
# 20 - for training set creation
# 5  - for classification
WIN_STEP = 5

# Video lenght
VID_LEN = SEQ_LEN * (NUM_SKIPPED_FRAMES + 1)
