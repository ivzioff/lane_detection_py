'''
lane detection 
based on edge detection

'''
import cv2 # opencv 사용
import numpy as np

YELLOW_DETECT_FRAME_LIMIT = 5

yellow_left_detected = False
yelloe_right_deetected = False

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0,0,255)
FONT_THICKNESS = 1

yellow_left = 'not_detected'
yellow_right = 'not_detected'

white_left = 'not_detected'
white_right = 'not_detected'

yellow_right_detected = False
yellow_left_detected = False

left_count = 0
right_count= 0

yellow_left_fit_line = []
yellow_right_fit_line = []
white_lines = []

def roi(orig_frame, vertices):
    mask = np.zeros_like(orig_frame)
    
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(orig_frame, mask)
    return masked

def get_fitline(img, f_lines):   
    lines = np.squeeze(f_lines)
    lines = lines.reshape(lines.shape[0]*2,2)
    rows,cols = img.shape[:2]
    output = cv2.fitLine(lines,cv2.DIST_L2,0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0]-1)-y)/vy*vx + x) , img.shape[0]-1
    x2, y2 = int(((img.shape[0]/2+50)-y)/vy*vx + x) , int(img.shape[0]/2+50)
    
    result = [x1,y1,x2,y2]
    return result
video_name = 'challenge.mp4'
#video_name = 'test.mp4'
#video_name = 'test2.mp4'
#video_name = 'test3.mp4'
video = cv2.VideoCapture(video_name)

if video_name == 'test.mp4':
    road_type = 'urban'
if video_name == 'challenge.mp4':
    road_type = 'highway'

while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture(video_name)
        continue 
    frame_total = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = video.get(cv2.CAP_PROP_POS_FRAMES)
    
    height, width = orig_frame.shape[:2]
    height = int(height / 2)
    width = int(width / 2)
    redim = (width, height)
    orig_frame = cv2.resize(orig_frame, redim)
    
    y, Cr, Cb = cv2.split(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2YCrCb))        

    x_margin = int(0.1 * width)
    if road_type == 'urban':
        x_margin2 = int(0.2 * width)
    elif road_type == 'highway':
        x_margin2 = int(0.05 * width)
    y_margin = int(0.1 * height)
    
    vertices = np.array([[(x_margin,height),(width/2 - x_margin2, height /2 + y_margin),
                          (width/2 + x_margin2, height / 2 + y_margin),(width - x_margin, height)]], dtype= np.int32)    
    
    # 1. lane detection: yellow lane
    # yellow lanes are extracted from Cb-channel
    
    yellow_result = orig_frame.copy()
    white_result = orig_frame.copy()  
    
    yellow_gray_img = Cb
    yellow_blur_img = cv2.GaussianBlur(yellow_gray_img, (3,3), 0)    
    ret, yellow_blur_img = cv2.threshold(yellow_blur_img, 100, 255, cv2.THRESH_BINARY)
    yellow_canny_img = cv2.Canny(yellow_blur_img, 70, 210)
    
    yellow_ROI_img = roi(yellow_canny_img, vertices) 
    
    yellow_lines = cv2.HoughLinesP(yellow_ROI_img, 1, 1 * np.pi/180, 30, np.array([]), 5, 10)
    yellow_lines = np.squeeze(yellow_lines)
#    print(len(yellow_lines))
#    temp = np.zeros_like(yellow_ROI_img)
#    for l in yellow_lines:
#        cv2.line(temp, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 1)
#    yellow_slope_degree = (np.arctan2(yellow_lines[:,1] - yellow_lines[:,3], yellow_lines[:,0] - yellow_lines[:,2]) * 180) / np.pi
#    test_lines = cv2.HoughLinesP(yellow_ROI_img, 1, 1*np.pi/180, 30, np.array([]), 5, 10)
#    test_lines = np.squeeze(test_lines)
#    test_lines
#    cv2.imshow('test', temp)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
    
    if len(yellow_lines.shape) <= 1:
        yellow_left = 'not_detected'
        yellow_right = 'not_detected'
        left_count += 1
        right_count += 1
        
        # if the yellow line is not detected in last 5 Frames, then the prgm shows only the original frame
        # if the yellow line is detected in last maximum 4 Frames, but not in this Frame, prgm show the previous result 
        if left_count >= YELLOW_DETECT_FRAME_LIMIT:
            left_count = YELLOW_DETECT_FRAME_LIMIT
        else:
            if len(yellow_left_fit_line) > 0:
                cv2.line(yellow_result, (yellow_left_fit_line[0], yellow_left_fit_line[1]), (yellow_left_fit_line[2], yellow_left_fit_line[3]), [0,255,0], 2)        
                cv2.line(orig_frame, (yellow_left_fit_line[0], yellow_left_fit_line[1]), (yellow_left_fit_line[2], yellow_left_fit_line[3]), [0,255,0], 2)
                
        if right_count >= YELLOW_DETECT_FRAME_LIMIT:
            right_count = YELLOW_DETECT_FRAME_LIMIT
        else:
            if len(yellow_right_fit_line) > 0:
                cv2.line(yellow_result, (yellow_right_fit_line[0], yellow_right_fit_line[1]), (yellow_right_fit_line[2], yellow_left_fit_line[3]), [0,255,0], 2)        
                cv2.line(orig_frame, (yellow_right_fit_line[0], yellow_right_fit_line[1]), (yellow_right_fit_line[2], yellow_left_fit_line[3]), [0,255,0], 2)          
                
    else:
        yellow_slope_degree = (np.arctan2(yellow_lines[:,1] - yellow_lines[:,3], yellow_lines[:,0] - yellow_lines[:,2]) * 180) / np.pi
        
        yellow_lines = yellow_lines[np.abs(yellow_slope_degree)<160]
        yellow_slope_degree = yellow_slope_degree[np.abs(yellow_slope_degree)<160]
        
        yellow_lines = yellow_lines[np.abs(yellow_slope_degree)>100]
        yellow_slope_degree = yellow_slope_degree[np.abs(yellow_slope_degree)>100]
        
        yellow_L_lines, yellow_R_lines = yellow_lines[(yellow_slope_degree>0),:], yellow_lines[(yellow_slope_degree<0),:]
        temp = np.zeros((orig_frame.shape[0], orig_frame.shape[1], 3), dtype=np.uint8)
        yellow_L_lines, yellow_R_lines = yellow_L_lines[:,None], yellow_R_lines[:,None]
#        print(len(L_lines))
#        print(len(R_lines))
        left_count = 0
        if len(yellow_L_lines)>2:
            yellow_left_fit_line = get_fitline(orig_frame,yellow_L_lines)    
            yellow_left_detected = True
        else:
            if left_count < YELLOW_DETECT_FRAME_LIMIT:
                left_count += 1
            elif left_count >= YELLOW_DETECT_FRAME_LIMIT:
                left_count = YELLOW_DETECT_FRAME_LIMIT 
                yellow_left_detected = False
        
        right_count = 0    
        if len(yellow_R_lines)>2:
            yellow_right_fit_line = get_fitline(orig_frame,yellow_R_lines)
            yellow_right_detected = True
        else :
            if right_count < YELLOW_DETECT_FRAME_LIMIT:
                right_count += 1
            elif left_count >= YELLOW_DETECT_FRAME_LIMIT:
                right_count = YELLOW_DETECT_FRAME_LIMIT
                yellow_right_detected = False
        
        if yellow_left_detected:
            cv2.line(yellow_result, (yellow_left_fit_line[0], yellow_left_fit_line[1]), (yellow_left_fit_line[2], yellow_left_fit_line[3]), [0,255,0], 2)        
            cv2.line(orig_frame, (yellow_left_fit_line[0], yellow_left_fit_line[1]), (yellow_left_fit_line[2], yellow_left_fit_line[3]), [0,255,0], 2)        
            yellow_left = 'detected'
        elif yellow_left_detected != True and left_count < YELLOW_DETECT_FRAME_LIMIT and len(yellow_left_fit_line) > 0:
            cv2.line(yellow_result, (yellow_left_fit_line[0], yellow_left_fit_line[1]), (yellow_left_fit_line[2], yellow_left_fit_line[3]), [0,255,0], 2)        
            cv2.line(orig_frame, (yellow_left_fit_line[0], yellow_left_fit_line[1]), (yellow_left_fit_line[2], yellow_left_fit_line[3]), [0,255,0], 2)        
            yellow_left = 'not_detected'
        else:
            yellow_left = 'not_detected'
            yellow_left_fit_line = []
            
            
        if yellow_right_detected:
            cv2.line(yellow_result, (yellow_right_fit_line[0], yellow_right_fit_line[1]), (yellow_right_fit_line[2], yellow_right_fit_line[3]), [0,255,0], 2)    
            cv2.line(orig_frame, (yellow_right_fit_line[0], yellow_right_fit_line[1]), (yellow_right_fit_line[2], yellow_right_fit_line[3]), [0,255,0], 2)    
            yellow_right = 'detected'
        elif yellow_right_detected != True and right_count < YELLOW_DETECT_FRAME_LIMIT and len(yellow_right_fit_line) > 0:
            cv2.line(yellow_result, (yellow_right_fit_line[0], yellow_right_fit_line[1]), (yellow_right_fit_line[2], yellow_right_fit_line[3]), [0,255,0], 2)    
            cv2.line(orig_frame, (yellow_right_fit_line[0], yellow_right_fit_line[1]), (yellow_right_fit_line[2], yellow_right_fit_line[3]), [0,255,0], 2)    
            yellow_right = 'not_detected'
        else:
            yellow_right = 'not_detected'
            yellow_right_fit_line = []
            
    '''
    2. lane detection: white lane
    white lanes are extracted from y-channel
    '''
    
    
    white_gray_img = y
#    white_gray_img = cv2.cvtColor(white_result, cv2.COLOR_BGR2GRAY)
#    white_gray_img = cv2.addWeighted(white_gray_img_1, 1.5, white_gray_img_2, 1.5, -200)
    
    blur_img = cv2.GaussianBlur(white_gray_img, (9,9), 0)    
#    _, blur_img = cv2.threshold(blur_img, 200,255, cv2.THRESH_BINARY) # there's no detected line
    canny_img = cv2.Canny(blur_img, 70, 210)
    
    white_ROI_img = roi(canny_img, vertices) 
    
    
    white_lines = cv2.HoughLinesP(white_ROI_img, 1, 1 * np.pi/180, 30, 10, 2)

    if white_lines.__class__ == None.__class__:   
        pass
    else:
        if len(white_lines.shape)>1:
            white_lines = np.squeeze(white_lines)
    
    ###
    tmp = np.zeros_like(white_ROI_img)
#    if white_lines.__class__ != None.__class__:
#        if len(white_lines.shape) >1 :
#            for l in white_lines:
#                cv2.line(tmp, (l[0], l[1]), (l[2], l[3]), (255,255,255), 1)
        ###
    if white_lines.__class__ == None.__class__:
        white_left = 'not_detected'
        white_right = 'not_detected'
        
    elif len(white_lines.shape) <= 1:
        white_left = 'not_detected'
        white_right = 'not_detected'
        
    else:
        white_slope_degree = (np.arctan2(white_lines[:,1] - white_lines[:,3], white_lines[:,0] - white_lines[:,2]) * 180) / np.pi
        
        white_lines = white_lines[np.abs(white_slope_degree)<150]
        white_slope_degree = white_slope_degree[np.abs(white_slope_degree)<150]
        
        white_lines = white_lines[np.abs(white_slope_degree)>120]
        white_slope_degree = white_slope_degree[np.abs(white_slope_degree)>120]

        white_L_lines, white_R_lines = white_lines[(white_slope_degree>0),:], white_lines[(white_slope_degree<0),:]
#        temp = np.zeros((orig_frame.shape[0], orig_frame.shape[1], 3), dtype=np.uint8)
        white_L_lines, white_R_lines = white_L_lines[:,None], white_R_lines[:,None]
#        print(len(L_lines))
#        print(len(R_lines))
        if len(white_L_lines)>2:
            white_left_fit_line = get_fitline(orig_frame,white_L_lines)
        else:
            pass

        if len(white_R_lines)>2:
            white_right_fit_line = get_fitline(orig_frame,white_R_lines)
        else:
            pass
        
        if 'white_left_fit_line' in locals().keys() or 'white_left_fit_line' in globals().keys(): # if white_lane on the left half is detected
            if len(white_left_fit_line) == 4:
                white_left = 'detected'
            else:
                white_left = 'not detected'
                
            if yellow_left == 'not_detected' and len(yellow_L_lines) == 0 and white_left == 'detected': # if the yellow line on the left half are nat detected 
                cv2.line(white_result, (white_left_fit_line[0], white_left_fit_line[1]), (white_left_fit_line[2], white_left_fit_line[3]), [255,0,0], 2)    
                cv2.line(orig_frame, (white_left_fit_line[0], white_left_fit_line[1]), (white_left_fit_line[2], white_left_fit_line[3]), [255,0,0], 2)    
                for l in white_L_lines:
                    cv2.line(tmp, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (255,255,255), 1)
            elif left_count == YELLOW_DETECT_FRAME_LIMIT and white_left == 'detected':
                cv2.line(white_result, (white_left_fit_line[0], white_left_fit_line[1]), (white_left_fit_line[2], white_left_fit_line[3]), [255,0,0], 2)    
                cv2.line(orig_frame, (white_left_fit_line[0], white_left_fit_line[1]), (white_left_fit_line[2], white_left_fit_line[3]), [255,0,0], 2)    
                for l in white_L_lines:
                    cv2.line(tmp, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (255,255,255), 1)
                
        if 'white_right_fit_line' in locals().keys() or 'white_right_fit_line' in globals().keys(): # if white_lane on the right half is detected
            if len(white_right_fit_line) == 4:
                white_right = 'detected'
            else:
                white_right = 'not_detected'
                
            if yellow_right == 'not_detected' and len(yellow_R_lines) == 0 and white_right == 'detected': # if the yellow line on the right half are not detected
                cv2.line(white_result, (white_right_fit_line[0], white_right_fit_line[1]), (white_right_fit_line[2], white_right_fit_line[3]), [255,0,0], 2)    
                cv2.line(orig_frame, (white_right_fit_line[0], white_right_fit_line[1]), (white_right_fit_line[2], white_right_fit_line[3]), [255,0,0], 2)    
                for l in white_R_lines:
                    cv2.line(tmp, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (255,255,255), 1)
            elif right_count == YELLOW_DETECT_FRAME_LIMIT and white_right == 'detected':
                cv2.line(white_result, (white_right_fit_line[0], white_right_fit_line[1]), (white_right_fit_line[2], white_right_fit_line[3]), [255,0,0], 2)    
                cv2.line(orig_frame, (white_right_fit_line[0], white_right_fit_line[1]), (white_right_fit_line[2], white_right_fit_line[3]), [255,0,0], 2)    
                for l in white_R_lines:
                    cv2.line(tmp, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (255,255,255), 1)
         
    cv2.putText(orig_frame, 'white_lane_left : ' + white_left, (10,20), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    cv2.putText(orig_frame, 'white_lane_right: ' + white_right, (10,40), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    cv2.putText(orig_frame, 'yellow_lane_left : ' + yellow_left, (10,60), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    cv2.putText(orig_frame, 'yellow_lane_right: ' + yellow_right, (10,80), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    cv2.putText(orig_frame, 'Frame: ' + str(frame_count) + ' / ' + str(frame_total), (10,100), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)    
    
    cv2.imshow('Original Frame', orig_frame)
    cv2.imshow('result: white lanes', white_result)
    cv2.imshow('result: yellow lanes', yellow_result)
    cv2.imshow('y_edges', yellow_ROI_img)
    cv2.imshow('w_edges', white_ROI_img)
    cv2.imshow('white-gray-image', white_gray_img)
    cv2.imshow('yellow_thresh', yellow_blur_img)
    cv2.imshow('yellow-gray-img', Cb)
    cv2.imshow('w-lines', tmp)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break;
    
video.release()
cv2.destroyAllWindows()


