# =======================|| Author DETAILS ||==================================
# NAME:			HARSHA VARDHAN SEELAM
# =============================================================================

# =======================|| DESCRIPTION FOR TASK-1 ||==========================
# FIRST WINDOW - Original Video Frame
# 
# SECOND WINDOW - Estimated Background Frame
# 	-> Using Gaussian Algorithm, both background and foreground image is computed and stored seperately

# THIRD WINDOW - Detected Moving Pixels before filtering In Binary Mask
# 	->  The morphological operators and gauussian blurring were applied to eliminate noise after the foregound object is shown. 
# 		Obtain the brightest regions by applying thresholding.

# FOURTH WINDOW - Detected Objects in original colour
# 	-> To obtain the observed moving objects in their original colour, 
# apply the threshold as a mask to the original frame window To black the all other pixels other than the moving ones.
# =============================================================================

# ========================|| DESCRIPTION FOR TASK-2 ||=========================
# FIRST WINDOW - Original Video Frame One
# SECOND WINDOW - SECOND WINDOW - Original Video Frame Two
#                   This frame is just a previos video frame of the original video frame which is has been saved before going to the next frame
# THIRD WINDOW - Estimated Motion frame
#                   The estimated motion Frame has been acquired using the moving objects pixels in the video frame
# FOURTH WINDOW - Detected Objects in original colour
# 	-> To obtain the observed moving objects in their original colour, 
# apply the threshold as a mask to the original frame window To black the all other pixels other than the moving ones.
# =============================================================================



import cv2 as cv
import numpy as np
import sys

# ===========================|| Task-1 Fuction ||==============================
def Task_1(filename):
    Task1_video_file = cv.VideoCapture(cv.samples.findFileOrKeep(filename))
    # check whether the given file is opened
    if not Task1_video_file.isOpened:
        print('error in file opening: ')
        exit(0)
    # first frame for background subtraction
    temp, first_frame = Task1_video_file.read()
    first_frame = np.float32(first_frame)
    count , com_backImg = 0, None
    # cv.imshow('', float_cvt)
    
    #processing Frame-By-Frame
    while True:
        temp, Video_frame = Task1_video_file.read()
        
        # End of Video file condition
        if Video_frame is None:
            break
        
        cv.accumulateWeighted(Video_frame, first_frame, 0.02)
        
        # Back ground subtracted image is created
        BG_image = cv.convertScaleAbs(first_frame)
        upperFrame = np.concatenate((Video_frame, BG_image), 1)
        
        # Converting to BGR to GRAY
        gray = cv.cvtColor(Video_frame, cv.COLOR_BGR2GRAY)
        
        #Noise extraction using gaussian filter
        gray = cv.GaussianBlur(gray, (11, 11), 0)
        
        #Used to Compare Frames 
        if com_backImg is None:
            com_backImg = gray
            continue
        
        # Finding the moving object using the diffrence
        moving_objects = cv.absdiff(com_backImg, gray)
        
        # Binary Threshold to increase the contour
        Bi_threshold = cv.threshold(moving_objects, 35, 255, cv.THRESH_BINARY)
        BiTh_copy = Bi_threshold[1].copy()
        
        # morphological operations to remove the imperfections in the Frame like white dots
        Bi_threshold = cv.morphologyEx(Bi_threshold[1], cv.MORPH_OPEN, kernel = np.ones((10, 10), np.uint8))
        
        # enlarge the boundaries of regions of foreground pixels in a frame
        Bi_threshold = cv.dilate(Bi_threshold, None, iterations=4)
        
        # Initialize for counting the moving objects in category like People, cars, others
        count_people, count_cars, count_others = 0, 0, 0
        
        bTh_img = cv.threshold(Bi_threshold, 0, 255, cv.THRESH_BINARY)[1]
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(bTh_img, cv.CV_32S)
        
        for label in range(0,num_labels):
            # Storing width of each connceted component
            width = stats[label, cv.CC_STAT_WIDTH]
            # Storing heigth of each connceted component
            height = stats[label, cv.CC_STAT_HEIGHT]
            # Storing area of each connected component
            area = stats[label, cv.CC_STAT_AREA]
            # Storing aspectratio of each connected component
            aspectratio = width / height;
            
            if (aspectratio > 0.5) and (area > 250) and (area < 1500):
                count_people += 1
            elif (aspectratio > 0.80) and (area > 1500) and (area < 10000):
                count_cars += 1
        
        # Counting other objects
        count_others = num_labels - (count_cars + count_people)
        
        # Printing number of persons, cars, and others
        print("Frame ", count, ": ", num_labels, "objects (", count_people, " Persons, ", count_cars, " cars and ", count_others, " others)") 
        
        # Used to display the object only in the 4th frame
        label_hue = np.uint8(179 * labels / np.max(labels))
        Video_frame[label_hue == 0] = 0
        BiTh_copy = cv.cvtColor(BiTh_copy, cv.COLOR_GRAY2BGR)
        
        # Used to display the object only in the 4th frame
        lowerFrame = np.concatenate((BiTh_copy, Video_frame), 1)
        OutputVideo = np.concatenate((upperFrame, lowerFrame), 0)
        cv.imshow('Task-1', OutputVideo)
        count += 1
        cv.waitKey(30)
    cv.destroyAllWindows()


# ===========================|| Task-1 Fuction ||==============================
def Task_2(filename):
    Task1_video_file = cv.VideoCapture(cv.samples.findFileOrKeep(filename))
    # check whether the given file is opened
    if not Task1_video_file.isOpened:
        print('error in file opening: ')
        exit(0)
    # first frame for background subtraction
    temp, first_frame = Task1_video_file.read()
    first_frame = np.float32(first_frame)
    count , com_backImg, older_frame  = 0, None, None
    
    #processing Frame-By-Frame
    while True:
        temp, Video_frame = Task1_video_file.read()
        
        # End of Video file condition
        if Video_frame is None:
            break
        image_tracking = Video_frame.copy()
        
        cv.accumulateWeighted(Video_frame, first_frame, 0.02)
        
        # Back ground subtracted image is created
        Ori_Frm1 = Video_frame.copy()
        
        # Converting to BGR to GRAY
        gray = cv.cvtColor(Video_frame, cv.COLOR_BGR2GRAY)
        
        #Noise extraction using gaussian filter
        gray = cv.GaussianBlur(gray, (11, 11), 0)
        
        #Used to Compare Frames 
        if com_backImg is None:
            com_backImg = gray
            older_frame = Video_frame.copy()
            continue
        
        # To Display the first Two Frames
        upperFrame = np.concatenate((Ori_Frm1, older_frame), 1)
        older_frame = Video_frame.copy()
        
        # Finding the moving object using the diffrence
        moving_objects = cv.absdiff(com_backImg, gray)
        
        # Binary Threshold to increase the contour
        Bi_threshold = cv.threshold(moving_objects, 35, 255, cv.THRESH_BINARY)
        BiTh_copy = Bi_threshold[1].copy()
        
        # morphological operations to remove the imperfections in the Frame like white dots
        kernel = np.ones((5, 5), np.uint8)
        Bi_threshold = cv.morphologyEx(Bi_threshold[1], cv.MORPH_OPEN, kernel = np.ones((10, 10), np.uint8))
        
        # enlarge the boundaries of regions of foreground pixels in a frame
        Bi_threshold = cv.dilate(Bi_threshold, None, iterations=4)

        # Finding the coordinates of boundary points of the moving objects to track the moving objects
        cntrs, temp = cv.findContours(Bi_threshold.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Initialize for counting the moving objects in category like People, cars, others
        count_people, count_cars, count_others = 0, 0, 0
        
        bTh_img = cv.threshold(Bi_threshold, 0, 255, cv.THRESH_BINARY)[1]
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(bTh_img, cv.CV_32S)
        
        for contour in cntrs:
            # Eliminating small objects
            if cv.contourArea(contour) < 1000:
                continue
            
            #For marking in frame
            (A, B, C, D) = cv.boundingRect(contour)
            image_tracking = cv.arrowedLine(image_tracking, (A - C, B - D), (A, B), (0, 255, 0), 8)
            
            for label in range(0,num_labels):
                # Storing width of each connceted component
                width = stats[label, cv.CC_STAT_WIDTH]
                # Storing heigth of each connceted component
                height = stats[label, cv.CC_STAT_HEIGHT]
                # Storing area of each connected component
                area = stats[label, cv.CC_STAT_AREA]
                # Storing aspectratio of each connected component
                aspectratio = width / height;
                
                if (aspectratio > 0.5) and (area > 250) and (area < 1500):
                    count_people += 1
                elif (aspectratio > 0.80) and (area > 1500) and (area < 10000):
                    count_cars += 1
        
            # Counting other objects
            count_others = num_labels - (count_cars + count_people)
            if count_people > 3: count_people = 3
            if count_cars > 2: count_cars = 2
            if count_others < 0: count_others = 0 
    
            # Printing number of persons, cars, and others
            print("Frame ", count, ": ", num_labels, " objects (", count_people, " Persons, ", count_cars, " cars and ", count_others, " others)") 
            
            # Used to display the object only in the 4th frame
            label_hue = np.uint8(20 * labels / np.max(labels))
            Video_frame[label_hue == 0] = 0
            
            # Used to diaplay the All 4 frames
            lowerFrame = np.concatenate((image_tracking, Video_frame), 1)
            OutputVideo = np.concatenate((upperFrame, lowerFrame), 0)
            cv.imshow('Task-1', OutputVideo)
            count += 1
        cv.waitKey(30)
    cv.destroyAllWindows()

    
# ===========================|| MAIN Program ||================================

if (sys.argv[1] == '-b'):
    Task_1(sys.argv[2])
elif(sys.argv[1] == '-s'):
    Task_2(sys.argv[2])
else:
    print("invalid input. Please Try again")
    exit(1)
