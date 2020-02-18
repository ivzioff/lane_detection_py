'''
lane detection 
based on edge detection

'''
import cv2 # opencv 사용
import numpy as np

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
video = cv2.VideoCapture(video_name)

while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture(video_name)
        continue 
#    
#    frame_total = video.get(cv2.CAP_PROP_FRAME_COUNT)
#    frame_count = video.get(cv2.CAP_PROP_POS_FRAMES)
    
    height, width = orig_frame.shape[:2]
    height = int(height / 2)
    width = int(width / 2)
    redim = (width, height)
    orig_frame = cv2.resize(orig_frame, redim)
    
#    hsv = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2HSV)
#    h,s,v = cv2.split(hsv)
#    v = cv2.equalizeHist(v)
#    hsv = cv2.merge((h,s,v))
#    enhenced_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    

    gray_img = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
#    clahe = cv2.createCLAHE(clipLimit = 4.0, tileGridSize = (8,8))
#    enhenced_frame = clahe.apply(gray_img)
    gray2 = -gray_img
    blur_img = cv2.GaussianBlur(gray_img, (3,3), 0)    
    
    canny_img = cv2.Canny(blur_img, 70, 210)
    
    x_margin = int(0.05 * width)
    x_margin2 = int(0.05 * width)
    y_margin = int(0.05 * height)
    
    vertices = np.array([[(x_margin,height),(width/2 - x_margin2, height /2 + y_margin),
                          (width/2 + x_margin2, height / 2 + y_margin),(width - x_margin, height)]], dtype= np.int32)
    ROI_img = roi(canny_img, vertices) 
    
    lines = cv2.HoughLinesP(ROI_img, 1, 1 * np.pi/180, 30, 10, 20)
    lines = np.squeeze(lines)
#    print(frame_count)
    if len(lines.shape) == 1:
        pass
    else:
        slope_degree = (np.arctan2(lines[:,1] - lines[:,3], lines[:,0] - lines[:,2]) * 180) / np.pi
        
        lines = lines[np.abs(slope_degree)<160]
        slope_degree = slope_degree[np.abs(slope_degree)<160]
        
        lines = lines[np.abs(slope_degree)>105]
        slope_degree = slope_degree[np.abs(slope_degree)>105]
        
        L_lines, R_lines = lines[(slope_degree>0),:], lines[(slope_degree<0),:]
        temp = np.zeros((orig_frame.shape[0], orig_frame.shape[1], 3), dtype=np.uint8)
        L_lines, R_lines = L_lines[:,None], R_lines[:,None]
#        print(len(L_lines))
#        print(len(R_lines))
        if len(L_lines)>2:
            left_fit_line = get_fitline(orig_frame,L_lines)
            
        else:
            pass
        if len(R_lines)>2:
            right_fit_line = get_fitline(orig_frame,R_lines)
            
        cv2.line(orig_frame, (left_fit_line[0], left_fit_line[1]), (left_fit_line[2], left_fit_line[3]), [255,0,0], 2)
        cv2.line(orig_frame, (right_fit_line[0], right_fit_line[1]), (right_fit_line[2], right_fit_line[3]), [255,0,0], 2)
        
        result = cv2.addWeighted(orig_frame, 1., temp, 1., 0.)

    result = orig_frame
    cv2.imshow('result',result)
    cv2.imshow('gray', gray_img)
    cv2.imshow('inverse', gray2)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break;
    
video.release()
cv2.destroyAllWindows()