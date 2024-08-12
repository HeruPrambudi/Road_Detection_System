import cv2
import numpy as np
import csv
import time
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

cv2.namedWindow("Normal", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Normal", 640, 480)

# cv2.namedWindow("Warped", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Warped", 640, 480)

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 640, 480)

end_time = None
start_time = None

class Road_Detection:
    def __init__(self, frame):
        self.frame              = frame
        self.threshold_frame    = None
        self.warped_frame       = None
        self.result_frame       = None
        self.slide_previous_window_frame = None
        self.sliding_window_frame = None
        self.frame_sliding_window = None
        self.out_of_lane_decision = None

        self.drawROI = None
        self.laneTolerance = 50 #dalam cm
        self.widthOfLane = 250
        self.alert = 0
        self.outOfLane = 0
        self.fps                = 0
        self.brightness         = 0
        self.percentage_brightness  = 0
        self.frame_size         = (self.frame.shape[1], self.frame.shape[0])
        self.width              = 640
        self.height             = 480
        self.roi_points         = self.getRoi("roi_points.csv")
        self.src_roi            = np.float32(self.roi_points)
        # self.src_roi            = np.float32([  ((45/100)*self.width, (72/100)*self.height), # Top-left corner
        #                                         ((14/100)*self.width, self.height), # Bottom-left corner            
        #                                         ((90/100)*self.width, self.height), # Bottom-right corner
        #                                         ((57/100)*self.width, (72/100)*self.height) ])
        self.dst_roi            = np.float32([  (0, 0), # Top-left corner
                                                (0, self.height), # Bottom-left corner            
                                                (self.width, self.height), # Bottom-right corner
                                                (self.width, 0) ])
        self.transformation_matrix      = cv2.getPerspectiveTransform(self.src_roi, self.dst_roi)
        self.inv_transformation_matrix  = cv2.getPerspectiveTransform(self.dst_roi, self.src_roi)
        self.min_threshold, self.max_threshold = self.getThresholdingValue('threshold_value.csv')
        self.histogram                  = None
        self.no_of_windows              = 10
        self.margin                     = int((1/12) * self.width)  
        self.minpix                     = int((1/24) * self.width)              
        self.left_fit                   = None
        self.right_fit                  = None
        self.left_lane_inds             = None
        self.right_lane_inds            = None
        self.ploty                      = None
        self.left_fitx                  = None
        self.right_fitx                 = None
        self.leftx                      = None
        self.rightx                     = None
        self.lefty                      = None
        self.righty                     = None
        self.YM_PER_PIX                 = 10.0 / 1000 # meters per pixel in y dimension
        self.XM_PER_PIX                 = 3.5 / 400 # meters per pixel in x dimension
        self.left_curvem                = None
        self.right_curvem               = None
        self.ave_curvem                 = None
        self.center_offset              = None
        self.decision                   = None
        self.leftx_base                 = None
        self.rightx_base                = None
        self.symbol_size                = 50
        self.straight_symbol            = cv2.imread("straight.png", cv2.IMREAD_UNCHANGED)
        self.turn_left_symbol           = cv2.imread("turn-left-rbg.png", cv2.IMREAD_UNCHANGED)
        self.turn_right_symbol          = cv2.imread("turn-right-rbg.png", cv2.IMREAD_UNCHANGED)
        self.straight_symbol            = cv2.resize(self.straight_symbol, (self.symbol_size, self.symbol_size))
        self.turn_left_symbol           = cv2.resize(self.turn_left_symbol, (self.symbol_size, self.symbol_size))
        self.turn_right_symbol          = cv2.resize(self.turn_right_symbol, (self.symbol_size, self.symbol_size))
        self.symbol_position            = ((int((12/100)*self.width)),int((1/100)*self.height))
        self.roi_symbol                 = self.frame[self.symbol_position[1]:self.symbol_position[1] + self.symbol_size, 
                                            self.symbol_position[0]:self.symbol_position[0] + self.symbol_size]
        
        self.array_zero = False
        # self.plot_sliding_window        = 0
        # self.plot_slide_previous_window = 0

    def draw_rotating_steering(self, img, center, radius, angle):
        # Hitung posisi titik-titik ujung garis di sekitar lingkaran
        points = []
        for i in range(3):
            x = int(center[0] + radius * np.cos(angle + i * 2 * np.pi / 3))
            y = int(center[1] + radius * np.sin(angle + i * 2 * np.pi / 3))
            points.append((x, y))
        
        # Gambar lingkaran
        cv2.circle(img, center, radius, (0, 255, 0), 10)
        
        # Gambar tiga garis dari pusat ke titik-titik ujung
        for point in points:
            cv2.line(img, center, point, (0, 0, 255), 15)


    def getRoi(self, csv_filename):
        points = []
        with open(csv_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                x, y = map(int, row)
                points.append((x, y))
        return points
        
    def getThresholdingValue(self, csv_filename): 
        with open(csv_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            min_threshold = tuple(map(int, next(csv_reader)))
            max_threshold = tuple(map(int, next(csv_reader)))
        return min_threshold, max_threshold

    def setSymbols(self, symbol, frame):
        height, width = symbol.shape[:2]
        top_left_x = int(self.symbol_position[0])
        top_left_y = int(self.symbol_position[1])
        for y in range(height):
            for x in range(width):
                overlay_color = symbol[y, x, :3]  # first three elements are color (RGB)
                overlay_alpha = symbol[y, x, 3] / 255  # 4th element is the alpha channel, convert from 0-255 to 0.0-1.0
                background_color = frame[top_left_y + y, top_left_x + x]
                composite_color = background_color * (1 - overlay_alpha) + overlay_color * overlay_alpha
                frame[self.symbol_position[1] + y, self.symbol_position[0] + x] = composite_color
        return frame
    
    def thresholdingProcess(self):
        hsv_frame               = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_frame)

        # Hitung rata-rata kecerahan (nilai V)
        self.brightness = np.mean(v)
        # brightness = int(gray_frame.mean())
        self.percentage_brightness = (self.brightness/255)*100
        # print((brightness/255)*100)


        # self.threshold_frame    = cv2.inRange(hsv_frame, self.min_threshold, self.max_threshold)
        yellow_lane             = cv2.inRange(hsv_frame, (0, 31, 100), (61,173,255))#datasetnew2
        # yellow_lane             = cv2.inRange(hsv_frame, (0, 80, 142), (39,255,255))#malam
        #yellow_lane             = cv2.inRange(hsv_frame, (20, 40, 160), (179,255,255))#malam
        white_lane              = cv2.inRange(hsv_frame, (0, 0, 134), (179,183,235)) #bermarka
        # white_lane              = cv2.inRange(hsv_frame, (0, 0, 120), (179,35,235)) #dtasetnew1
        # white_lane              = cv2.inRange(hsv_frame, (40, 0, 133), (130,80,255))#tolsda
        # white_lane              = cv2.inRange(hsv_frame, (91, 0, 101), (179,52,247))#datasetnew2
        # white_lane              = cv2.inRange(hsv_frame, (0, 0, 147), (179,37,255))#malam new
        # white_lane              = cv2.inRange(hsv_frame, (0, 0, 110), (179,27,255))#malam
        self.threshold_frame    = cv2.bitwise_or(white_lane, yellow_lane)
        self.threshold_frame    = cv2.equalizeHist(self.threshold_frame) 
        return self.threshold_frame

    
    def perspective_transform(self):    
        self.warped_frame = cv2.warpPerspective(self.threshold_frame, self.transformation_matrix, self.frame_size, flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
        return self.warped_frame
        
    def calculate_histogram(self):
        self.histogram = np.sum(self.warped_frame[int(self.warped_frame.shape[0]/2):,:], axis=0)
        # Uncomment if you want to show plot
        # self.plot_histogram()
        return self.histogram

    def plot_histogram(self):
        figure, (ax1, ax2) = plt.subplots(2,1) # 2 row, 1 columns
        ax1.imshow(self.warped_frame, cmap='gray')
        # ax2.set_title("Warped Frame")
        ax2.plot(self.histogram)
        # ax2.set_title("Histogram Peaks")
        figure.canvas.draw()
        plot_img = np.array(figure.canvas.renderer.buffer_rgba())
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        # self.plot_histogram = plot_img
        cv2.imshow('Plot_sliding_window', plot_img)

    def histogram_peak(self):
        midpoint    = int(self.histogram.shape[0]/2)
        leftx_base  = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint
        print(rightx_base - leftx_base)
        if(rightx_base - leftx_base < self.widthOfLane):
            self.outOfLane = 1
        else:
            self.outOfLane = 0
        # max_width_road_frame = 650
        # min_width_road_frame = 450
        # if((rightx_base - leftx_base) > max_width_road_frame):
        #     if((rightx_base-midpoint)>(midpoint-leftx_base)):
        #         max_rightx_base = leftx_base+max_width_road_frame
        #         self.rightx_base = np.argmax(self.histogram[midpoint:max_rightx_base]) + midpoint
        #         self.leftx_base = leftx_base
        #     else:
        #         min_leftx_base = rightx_base-max_width_road_frame
        #         self.leftx_base = np.argmax(self.histogram[min_leftx_base:midpoint]) + min_leftx_base
        #         self.rightx_base = rightx_base
        #     print("ubah", rightx_base, leftx_base)
        # elif((rightx_base - leftx_base) < min_width_road_frame):
        #     self.leftx_base = self.leftx_base    
        #     self.rightx_base = self.rightx_base
        # else:
        #     self.leftx_base = leftx_base
        #     self.rightx_base = rightx_base
        # buat algoritma yang dekat dengan tengah
        return leftx_base, rightx_base

    def get_lane_line_indices_sliding_windows(self):
        margin                  = self.margin
        frame_sliding_window    = self.warped_frame.copy()
        window_height           = int(self.warped_frame.shape[0]/self.no_of_windows)       
        nonzero                 = self.warped_frame.nonzero()
        nonzeroy                = np.array(nonzero[0])
        nonzerox                = np.array(nonzero[1]) 
        left_lane_inds          = []
        right_lane_inds         = []
        leftx_base, rightx_base = self.histogram_peak()
        leftx_current           = leftx_base
        rightx_current          = rightx_base
        no_of_windows           = self.no_of_windows

        for window in range(no_of_windows):
            win_y_low           = self.warped_frame.shape[0] - (window + 1) * window_height
            win_y_high          = self.warped_frame.shape[0] - window * window_height
            win_xleft_low       = leftx_current - margin
            win_xleft_high      = leftx_current + margin
            win_xright_low      = rightx_current - margin
            win_xright_high     = rightx_current + margin
            cv2.rectangle(frame_sliding_window,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (255,255,255), 2)
            cv2.rectangle(frame_sliding_window,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (255,255,255), 2)
            good_left_inds      = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds     = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
                
            # If you found > minpix pixels, recenter next window on mean position
            minpix = self.minpix
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                        
        left_lane_inds  = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        if(left_lane_inds.size == 0 or right_lane_inds.size == 0 or self.outOfLane == 1):
            self.array_zero = True
            self.result_frame = self.frame
            cv2.putText(self.result_frame, 'OUT OF LANE', (int((370/600)*self.width), int((40/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                    (float((0.5/600)*self.width)),(0,0,255),2,cv2.LINE_AA)
            print("array kosong")
        else :
            self.array_zero = False
            leftx           = nonzerox[left_lane_inds]
            lefty           = nonzeroy[left_lane_inds] 
            rightx          = nonzerox[right_lane_inds] 
            righty          = nonzeroy[right_lane_inds]
            self.left_fit   = np.polyfit(lefty, leftx, 2)
            self.right_fit  = np.polyfit(righty, rightx, 2)  
            self.frame_sliding_window = frame_sliding_window

            # Uncomment if you want to show plot
            # self.plot_sliding_window()

        return self.left_fit, self.right_fit
    
    def get_lane_line_previous_window(self, left_fit, right_fit):
        margin      = self.margin
        nonzero     = self.warped_frame.nonzero()  
        nonzeroy    = np.array(nonzero[0])
        nonzerox    = np.array(nonzero[1])
        self.left_lane_inds     = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (
                                    nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        self.right_lane_inds    = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (
                                    nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
        self.leftx  = nonzerox[self.left_lane_inds]
        self.lefty  = nonzeroy[self.left_lane_inds] 
        self.rightx = nonzerox[self.right_lane_inds]
        self.righty = nonzeroy[self.right_lane_inds]          
        self.left_fit   = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit  = np.polyfit(self.righty, self.rightx, 2)
        self.ploty      = np.linspace(0, self.warped_frame.shape[0]-1, self.warped_frame.shape[0]) 
        self.left_fitx  = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        
        # Uncomment if you want to show plot
        # self.plot_slide_previous_window()

    def plot_sliding_window(self):
        nonzero     = self.warped_frame.nonzero()
        nonzeroy    = np.array(nonzero[0])
        nonzerox    = np.array(nonzero[1])
        frame_sliding_window = self.frame_sliding_window.copy()
        ploty = np.linspace(0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        out_img = np.dstack((
            frame_sliding_window, frame_sliding_window, (
            frame_sliding_window))) * 255
        out_img[nonzeroy[self.left_lane_inds], nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[self.right_lane_inds], nonzerox[self.right_lane_inds]] = [0, 0, 255]
        figure, (ax2, ax3) = plt.subplots(2,1) # 3 rows, 1 column
        ax2.imshow(frame_sliding_window, cmap='gray')
        ax3.imshow(out_img)
        ax3.plot(left_fitx, ploty, color='yellow')
        ax3.plot(right_fitx, ploty, color='yellow')
        figure.canvas.draw()
        plot_img = np.array(figure.canvas.renderer.buffer_rgba())
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        self.sliding_window = plot_img
        
        cv2.imshow('Plot_sliding_window', plot_img)
        

    def plot_slide_previous_window(self):
        nonzero     = self.warped_frame.nonzero()
        nonzeroy    = np.array(nonzero[0])
        nonzerox    = np.array(nonzero[1]) 
        out_img     = np.dstack((self.warped_frame, self.warped_frame, (
                        self.warped_frame)))*255
        window_img  = np.zeros_like(out_img)
        out_img[nonzeroy[self.left_lane_inds], nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[self.right_lane_inds], nonzerox[self.right_lane_inds]] = [0, 0, 255]
        margin = self.margin
        left_line_window1 = np.array([np.transpose(np.vstack([
                                    self.left_fitx-margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                        self.left_fitx+margin, self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([
                                        self.right_fitx-margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                        self.right_fitx+margin, self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        figure, (ax2, ax3) = plt.subplots(2,1) # 3 rows, 1 column
        ax2.imshow(self.warped_frame, cmap='gray')
        ax3.imshow(result)
        ax3.plot(self.left_fitx, self.ploty, color='yellow')
        ax3.plot(self.right_fitx, self.ploty, color='yellow') 
        figure.canvas.draw()
        plot_img = np.array(figure.canvas.renderer.buffer_rgba())
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        self.slide_previous_window_frame = plot_img
        cv2.imshow('plot_slide_previous_window', plot_img)
                  

    
    def calculate_curvature(self, frame):
        y_eval          = np.max(self.ploty)    
        left_fit_cr     = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * (self.XM_PER_PIX), 2)
        right_fit_cr    = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * (self.XM_PER_PIX), 2)
        left_curvem     = ((1 + (2*left_fit_cr[0]*y_eval*self.YM_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curvem    = ((1 + (2*right_fit_cr[0]*y_eval*self.YM_PER_PIX + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        self.left_curvem    = left_curvem
        self.right_curvem   = right_curvem
        self.ave_curvem     = (left_curvem+right_curvem)/2
        percentage_curve    = (1/self.ave_curvem) * 100
        
        print("percentage : ", percentage_curve)
        cv2.rectangle(frame, (0, 0), (int((35/100)*self.width), int((30/100)*self.height)), (50, 50, 50), -1)
        cv2.rectangle(frame, (int((70/100)*self.width), 0), (int((100/100)*self.width), int((30/100)*self.height)), (50, 50, 50), -1)

        if(left_fit_cr[0] > 0 or right_fit_cr[0] > 0):
            self.draw_rotating_steering(frame, (int((85/100)*self.width), int((12/100)*self.height)), 40, math.radians(30)+(percentage_curve/100)*math.radians(180))
            cv2.putText(frame, '=>', (int((90/100)*self.width), int((28/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        elif(left_fit_cr[0] < 0 or right_fit_cr[0] < 0):
            self.draw_rotating_steering(frame, (int((85/100)*self.width), int((12/100)*self.height)), 40, math.radians(30)-(percentage_curve/100)*math.radians(180))        
            cv2.putText(frame, '<=', (int((74/100)*self.width), int((28/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)       
        else:
            self.draw_rotating_steering(frame, (int((85/100)*self.width), int((12/100)*self.height)), 40, math.radians(30))        
            

        
        cv2.putText(frame, str(percentage_curve)[:4] + '%', (int((81/100)*self.width), int((28/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, 'Brightness : '+str(self.percentage_brightness)[:2]+'%', (int((1/100)*self.width), int((10/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
            0.4,(255,255,255),1,cv2.LINE_AA) 
   

        if(self.percentage_brightness < 50):
            cv2.putText(frame, '[LESS BRIGHT]', (int((19/100)*self.width), int((10/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
            0.4,(0,0,255),1,cv2.LINE_AA) 
        elif(self.percentage_brightness < 80):
            cv2.putText(frame, '[GOOD BRIGHT]', (int((19/100)*self.width), int((10/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
            0.4,(0,255,0),1,cv2.LINE_AA) 
        else:
            cv2.putText(frame, '[TOO BRIGHT]', (int((19/100)*self.width), int((10/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
            0.4,(0,0,255),1,cv2.LINE_AA) 

        cv2.putText(frame, 'Lane Keeping : ', (int((1/100)*self.width), int((20/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4,(255,255,255),1,cv2.LINE_AA)
        
        cv2.putText(frame, 'Upcoming Road : ', (int((1/100)*self.width), int((15/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4,(255,255,255),1,cv2.LINE_AA)

        if left_fit_cr[0] < 0 and right_fit_cr[0] < 0 and percentage_curve > 5 :
            cv2.putText(frame, 'Turn Left', (int((19/100)*self.width), int((15/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4,(0,255,255),1,cv2.LINE_AA)
            
        elif left_fit_cr[0] > 0 and right_fit_cr[0] > 0 and percentage_curve > 5 :
            cv2.putText(frame, 'Turn Right', (int((19/100)*self.width), int((15/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4,(0,255,255),1,cv2.LINE_AA)
            
        else:
            cv2.putText(frame, 'Stay Straight', (int((19/100)*self.width), int((15/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4,(0,255,255),1,cv2.LINE_AA)



        # if left_fit_cr[0] < 0 and right_fit_cr[0] < 0 and self.ave_curvem < 50:
        #     self.decision = 'Turn Left'
        #     self.roi_symbol = self.turn_left_symbol[:, :, :3]
        #     cv2.putText(frame,'Left Curve Ahead', (int(( 1/100)*self.width), 
        #         int((16/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1,cv2.LINE_AA)
        #     cv2.putText(frame,'Curvature = '+str(self.ave_curvem)[:7]+' m ('+str(percentage_curve)[:4]+'%)', (int(( 1/100)*self.width), 
        #         int((19/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1,cv2.LINE_AA)
        #     frame = self.setSymbols(self.turn_left_symbol, frame)
        # elif left_fit_cr[0] > 0 and right_fit_cr[0] > 0 and self.ave_curvem < 50:
        #     self.decision = 'Turn Right'
        #     self.roi_symbol = self.turn_right_symbol[:, :, :3]
        #     cv2.putText(frame,'Right Curve Ahead', (int(( 1/100)*self.width), 
        #         int((16/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1,cv2.LINE_AA)
        #     cv2.putText(frame,'Curvature = '+str(self.ave_curvem)[:7]+' m ('+str(percentage_curve)[:4]+'%)', (int(( 1/100)*self.width), 
        #         int((19/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1,cv2.LINE_AA)
        #     frame = self.setSymbols(self.turn_right_symbol, frame)
        # else:
        #     self.decision = 'Straight'
        #     cv2.putText(frame,'Keep Straight Ahead', (int(( 1/100)*self.width), 
        #         int((16/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1,cv2.LINE_AA)
        #     cv2.putText(frame,'Curvature = '+str(self.ave_curvem)[:7]+' m ('+str(percentage_curve)[:4]+'%)', (int(( 1/100)*self.width), 
        #         int((19/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1,cv2.LINE_AA)
        #     frame = self.setSymbols(self.straight_symbol, frame)
        return frame

    def calculate_car_position(self, frame):
        car_location    = (self.roi_points[2][0] - self.roi_points[1][0])/2 + self.roi_points[1][0]
        height          = self.frame.shape[0]
        bottom_left     = (self.left_fit[0]*height**2 + self.left_fit[1]*height + self.left_fit[2])/(self.width)*(self.roi_points[2][0] - self.roi_points[1][0]) + self.roi_points[1][0]
        bottom_right    = (self.right_fit[0]*height**2 + self.right_fit[1]*height + self.right_fit[2])/(self.width)*(self.roi_points[2][0] - self.roi_points[1][0]) + self.roi_points[1][0]
        center_lane     = (bottom_right - bottom_left)/2 + bottom_left
        center_offset   = (np.abs(car_location) - np.abs(center_lane)) * self.XM_PER_PIX * 100            
        self.center_offset = center_offset

        
        if(center_offset > self.laneTolerance):
            cv2.putText(frame, 'TOO RIGHT', (int((1/100)*self.width), int((24/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4,(0,0,255),1,cv2.LINE_AA)
            self.alert = 1
        elif(center_offset < -self.laneTolerance):
            cv2.putText(frame, 'TOO LEFT', (int((1/100)*self.width), int((24/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4,(0,0,255),1,cv2.LINE_AA)
            self.alert = 1
        else:
            cv2.putText(frame, 'IN LANE', (int((1/100)*self.width), int((24/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4,(0,255,0),1,cv2.LINE_AA)
            self.alert = 0
            print("Pas")
        cv2.circle(frame, (int(car_location), self.height), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(frame, (int(center_lane), self.height), 10, (0, 0, 255), cv2.FILLED)

        # cv2.putText(frame, 'CENTER OFFSET : '+str(center_offset)[:5]+' cm', (int((370/600)*self.width), int((40/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                    # (float((0.5/600)*self.width)),(255,0,0),2,cv2.LINE_AA)
        cv2.putText(frame, 'Vehicle is '+str(float(center_offset/100))[:5]+' m away from center', (int((1/100)*self.width), int((28/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
            0.3,(255,255,255),1,cv2.LINE_AA)       
        self.result_frame = frame
        return self.result_frame

    # def out_of_lane_decision(self, center_offset):
    def overlay_lane_lines(self):
        warp_zero   = np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp  = np.dstack((warp_zero, warp_zero, warp_zero))       
        pts_left    = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right   = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts         = np.hstack((pts_left, pts_right))
        if(self.alert == 1):
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))
        else:
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        newwarp     = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (
                                    self.frame.shape[1], self.frame.shape[0]))
        cv2.imshow('warp', newwarp)
        
        result      = cv2.addWeighted(self.frame, 1, newwarp, 0.3, 0)
        point_draw = np.array(self.roi_points)
        point_draw = point_draw.reshape((-1, 1, 2))
        fillColor = (255,0,0)
        draw_frame = self.frame.copy()
        cv2.fillPoly(draw_frame, [point_draw], fillColor)
        # cv2.imshow('roi', result)
        self.result_frame = cv2.addWeighted(result, 1, draw_frame, 0.2, 0)
        
        # return result


    def process(self):
        global start_time, end_time
        if start_time is None:
            start_time = time.time()

        end_time = time.time()
        elapsed_time = end_time - start_time
        start_time = time.time()
        self.thresholdingProcess()
        self.perspective_transform()
        self.calculate_histogram()
        # # self.plot_histogram()
        left_fit, right_fit = self.get_lane_line_indices_sliding_windows()
        if(self.array_zero == False):
            self.get_lane_line_previous_window(left_fit, right_fit)
            out = self.calculate_curvature(self.frame)
            out = self.calculate_car_position(out)
            self.overlay_lane_lines()

        try:
            fps = 1/elapsed_time
            cv2.putText(self.result_frame, 'FPS : '+str(fps)[:4], (int((1/100)*self.width), int((5/100)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4,(255,255,255),1,cv2.LINE_AA)    
            print("fps : ", fps)
        except ZeroDivisionError:
            print("Error: Division by zero is not allowed.")


        cv2.imshow("Normal", self.threshold_frame)
        cv2.imshow("Warped", self.warped_frame)
        cv2.imshow("Result", self.result_frame)
        
        cv2.waitKey(1) 
            

def main():
    # cap = cv2.VideoCapture(0) 
    cap = cv2.VideoCapture('data/Bermarka_lengkap.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Gunakan 'X264' jika opsi ini tidak bekerja
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

    # savevideo = cv2.VideoWriter("result1.avi",
    #                         cv2.VideoWriter_fourcc(*'MJPG'),
    #                         30, (1280,720))
    if(cap.isOpened()==0):
        print('Video tidak ada\n')
    i = 0
    # leftx_base = None
    # rigthx_base = None
    while(cap.isOpened()):
        ret, frame = cap.read()
        filename = './sample_frame/image' + str(i) + '.jpg'
        # print(filename)
        if(i % 10 == 0):
            cv2.imwrite(filename, frame)
        
        i+=1
        if ret == True:
            frame = cv2.resize(frame, (640, 480))
            road = Road_Detection(frame=frame)
            
            # cv2.imwrite('sample.jpg', road.frame)
            # print(i)
            # i+=1
            road.process()
            # leftx_base = road.leftx_base
            # rigthx_base = road.rightx_base
            # road.threshold_frame = cv2.cvtColor(road.threshold_frame, cv2.COLOR_GRAY2BGR)
            
            out.write(road.result_frame)
            # savevideo.write(road.result_frame)
        else:   
            # savevideo.release()
            break

    cap.release()
    out.release()

main()