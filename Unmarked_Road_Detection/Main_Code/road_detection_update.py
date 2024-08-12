import cv2
import numpy as np
import time
import math

YM_PER_PIX                 = 10.0 / 1000
XM_PER_PIX                 = 3.5 / 400
#Size of frame
width = 640 
height = 480

def bgr_to_hsv(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv_frame

def curvature_radius(coefficients, y_values):
    A = coefficients[0]
    B = coefficients[1]
    radius = (1 + (2 * A * y_values * YM_PER_PIX + B)**2)**(3/2) / (2 * np.abs(A))    
    return radius

#read symbol
straight_symbol            = cv2.imread("./res/lurus.png", cv2.IMREAD_UNCHANGED)
turn_left_symbol           = cv2.imread("./res/belok_kanan.png", cv2.IMREAD_UNCHANGED)
turn_right_symbol          = cv2.imread("./res/belok_kiri.png", cv2.IMREAD_UNCHANGED)

symbol_size = 50

straight_symbol            = cv2.resize(straight_symbol, (symbol_size, symbol_size))
turn_left_symbol           = cv2.resize(turn_left_symbol, (symbol_size, symbol_size))
turn_right_symbol          = cv2.resize(turn_right_symbol, (symbol_size, symbol_size))

symbol_position            = ((int((12/100)*width)),int((1/100)*height))

def setSymbols(symbol, frame):
    height, width = symbol.shape[:2]
    top_left_x = int(symbol_position[0])
    top_left_y = int(symbol_position[1])
    
    # Hitung overlay_alpha di luar loop karena nilainya konstan
    overlay_alpha = symbol[:, :, 3] / 255  # 4th element is the alpha channel, convert from 0-255 to 0.0-1.0
    
    for y in range(height):
        for x in range(width):
            overlay_color = symbol[y, x, :3]  # first three elements are color (RGB)
            background_color = frame[top_left_y + y, top_left_x + x]
            composite_color = background_color * (1 - overlay_alpha[y, x]) + overlay_color * overlay_alpha[y, x]
            frame[symbol_position[1] + y, symbol_position[0] + x] = composite_color
            
    return frame


def draw_rotating_steering(img, center, radius, angle):
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


#================================================================#
# main Program

# # Range HSV jalan4
min_h = 0
min_s = 0
min_v = 0
max_h = 179
max_s = 82
max_v = 151


# min_h = 75
# min_s = 0
# min_v = 0
# max_h = 179
# max_s = 135
# max_v = 106



#ROI untuk warpperspective (left, up)(left, below)(right, below)(right, up)
pts_src = np.array([[int(0.3*width), int(0.75*height)], 
                    [int(0*width), height], 
                    [int(1*width), height], 
                    [int(0.7*width), int(0.75*height)]])


pts_dst = np.array([[0, 0], 
                    [0, height], 
                    [width, height], 
                    [width, 0]])

pts_src = pts_src.astype(np.float32)
pts_dst = pts_dst.astype(np.float32)


#transformation matrix
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
inv_matrix = cv2.getPerspectiveTransform(pts_dst, pts_src)

pts_src = pts_src.astype(np.int32)
pts_dst = pts_dst.astype(np.int32)

#threshold hsv config
lower_thresh = np.array([min_h, min_s, min_v])
upper_thresh = np.array([max_h, max_s, max_v])

#open video
video_capture = cv2.VideoCapture('./data/update.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Gunakan 'X264' jika opsi ini tidak bekerja
out = cv2.VideoWriter('output_BARU.mp4', fourcc, 20.0, (640, 480))

#define list
list_x = []
road_size_in_warp_frame = 460

i = 0

start_time = None

while True:
    if start_time is None:
        start_time = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    start_time = time.time()
    # fps = 1/elapsed_time
    # print("fps : ", fps)
    # read frame
    ret, frame = video_capture.read()
    
    if not ret:
        break

    # Atur frame di 640 x 480
    frame = cv2.resize(frame, (640, 480))

    roi_symbol = frame[symbol_position[1]:symbol_position[1] + symbol_size, 
                        symbol_position[0]:symbol_position[0] + symbol_size]


    # Gambar roi
    fill_roi = np.zeros_like(frame)
    cv2.fillPoly(fill_roi, [pts_src], (255, 0, 0))

    # Mengubah nilai alpha (transparansi) poligon
    alpha = 0.2  # 20% transparansi
    filled_img = cv2.addWeighted(fill_roi, alpha, frame, 1 - alpha, 0)


    # if(i % 60 == 0):
    #     namefile  = './sample_frame_4/image' + str(i) + '.jpg'
    #     cv2.imwrite(namefile, frame)

    # konversi ke hsv
    hsv_frame = bgr_to_hsv(frame)
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, s, v = cv2.split(hsv_frame)

    # Hitung rata-rata kecerahan (nilai V)
    brightness = np.mean(v)
    # brightness = int(gray_frame.mean())
    percentage_brightness = (brightness/255)*100
    # print((brightness/255)*100)

    # Threshold hsv by min_max value
    mask = cv2.inRange(hsv_frame, lower_thresh, upper_thresh)
    
    list_point = []
    list_exist = 0

    warp = cv2.warpPerspective(mask, matrix, (width, height))

    # Menentukan kernel untuk erosi dan dilasi
    kernel = np.ones((5,5), np.uint8)

    # Erosi
    warp = cv2.erode(warp, kernel, iterations = 1)

    cv2.imshow("warp", warp)

    if(not list_x):
        list_exist = 0
    else:
        list_exist = 1

    x = np.array([])
    y = np.array([])
    len_nonzero = np.array([])

    for row_index in range(height - 1, 0, -1):
        if list_exist == 1:
            row_index = int(row_index)
            batas_kiri = int(list_x[row_index-1]) - (road_size_in_warp_frame/2)
            batas_kanan = int(list_x[row_index-1]) + (road_size_in_warp_frame/2)
            
            batas_kiri = max(0, batas_kiri)  # Hindari nilai negatif
            batas_kanan = min(width-1, batas_kanan)  # Hindari melebihi batas lebar gambar
            
            warp[row_index, :int(batas_kiri)] = 0
            warp[row_index, int(batas_kanan):] = 0

        row_pixels = warp[row_index] # Ambil tiap baris
        non_zero_indices = np.nonzero(row_pixels)[0] # Hitung yang bernilai tidak nol 
        average_pixel_location = np.mean(non_zero_indices)

        x = np.append(x, average_pixel_location)
        y = np.append(y, row_index)
        len_nonzero = np.append(len_nonzero, len(non_zero_indices))
    mean_nonzero = np.mean(len_nonzero)

    frame_warp = np.zeros((height, width), dtype=np.uint8)
    ploty      = np.linspace(0, warp.shape[0]-1, warp.shape[0]) 
    y_eval = np.max(ploty)
    coeff = np.polyfit(y, x, 2)

    curve = curvature_radius(coeff, y_eval)
    percentage_curve = (100/curve)*100

    list_x.clear()

    for row_index in range(height -1, 0, -1):
        x_ = coeff[0]*row_index**2 + coeff[1]*row_index + coeff[2]

        list_x.append(x_)
        
        if not np.isnan(x_) and not np.isnan(row_index):
            cv2.circle(frame_warp, (int(x_), int(row_index)), 2, 0, -1)   
            cv2.line(frame_warp, (int(x_ - 0.5*len_nonzero[height - row_index - 1]), int(row_index)), (int(x_ + 0.5*len_nonzero[height-row_index-1]), int(row_index)), 255, 1)

    bgr_frame = np.zeros((height, width, 3), dtype=np.uint8)
    bgr_frame[:, :, 1] = frame_warp
    rewarp = cv2.warpPerspective(bgr_frame, inv_matrix, (width, height))
    filled_img = cv2.addWeighted(filled_img, 1, rewarp, 0.6, 0)

    #status berkendara
    center_lane     = list_x[height-2] 
    car_location    = width/2
    center_offset   = (np.abs(car_location) - np.abs(center_lane)) * XM_PER_PIX * 100            

    cv2.circle(filled_img, (int(car_location), height-2), 10, (255, 0, 0), cv2.FILLED)
    cv2.circle(filled_img, (int(center_lane), height-2), 10, (0, 0, 255), cv2.FILLED)

    center_offset   = (np.abs(car_location) - np.abs(center_lane)) * XM_PER_PIX * 100     

    # display 
    cv2.rectangle(filled_img, (0, 0), (int((40/100)*width), int((30/100)*height)), (50, 50, 50), -1)
    cv2.rectangle(filled_img, (int((70/100)*width), 0), (int((100/100)*width), int((30/100)*height)), (50, 50, 50), -1)

    #steer
    if(coeff[0] > 0):
        draw_rotating_steering(filled_img, (int((85/100)*width), int((12/100)*height)), 40, math.radians(30)+(percentage_curve/100)*math.radians(180))
        cv2.putText(filled_img, '=>', (int((90/100)*width), int((28/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 1, cv2.LINE_AA)        
    else:
        draw_rotating_steering(filled_img, (int((85/100)*width), int((12/100)*height)), 40, math.radians(30)-(percentage_curve/100)*math.radians(180))        
        cv2.putText(filled_img, '<=', (int((74/100)*width), int((28/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(filled_img, str(percentage_curve)[:4] + '%', (int((81/100)*width), int((28/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if(center_offset > 50):
        cv2.putText(filled_img, 'TOO RIGHT', (int((1/100)*width), int((25/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
            0.4,(0,0,255),1,cv2.LINE_AA)  
    elif(center_offset < -50):
        cv2.putText(filled_img, 'TOO LEFT', (int((1/100)*width), int((25/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
            0.4,(0,0,255),1,cv2.LINE_AA)
    else:
        cv2.putText(filled_img, 'IN LANE', (int((1/100)*width), int((25/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
            0.4,(0,255,0),1,cv2.LINE_AA)             

    if(percentage_brightness < 30):
        cv2.putText(filled_img, '[LESS BRIGHT]', (int((19/100)*width), int((10/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
        0.4,(0,0,255),1,cv2.LINE_AA) 
    elif(percentage_brightness < 70):
        cv2.putText(filled_img, '[GOOD BRIGHT]', (int((19/100)*width), int((10/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
        0.4,(0,255,0),1,cv2.LINE_AA) 
    else:
        cv2.putText(filled_img, '[TOO BRIGHT]', (int((19/100)*width), int((10/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
        0.4,(0,0,255),1,cv2.LINE_AA) 

    cv2.putText(filled_img, 'Upcoming Road : ', (int((1/100)*width), int((15/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4,(255,255,255),1,cv2.LINE_AA)

    if(percentage_curve > 5) and coeff[0] < 0 :
        cv2.putText(filled_img, 'Turn Left', (int((19/100)*width), int((15/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4,(0,255,255),1,cv2.LINE_AA)
        
    elif(percentage_curve > 5) and coeff[0] > 0 :
        cv2.putText(filled_img, 'Turn Right', (int((19/100)*width), int((15/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4,(0,255,255),1,cv2.LINE_AA)
        
    else:
        cv2.putText(filled_img, 'Stay Straight', (int((19/100)*width), int((15/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4,(0,255,255),1,cv2.LINE_AA)

    cv2.putText(filled_img, 'Lane Keeping : ', (int((1/100)*width), int((20/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
            0.4,(255,255,255),1,cv2.LINE_AA) 
    cv2.putText(filled_img, 'Vehicle is '+str(float(center_offset/100))[:5]+' m away from center', (int((1/100)*width), int((28/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
            0.3,(255,255,255),1,cv2.LINE_AA)  
    cv2.putText(filled_img, 'Brightness : '+str(percentage_brightness)[:2]+'%', (int((1/100)*width), int((10/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
            0.4,(255,255,255),1,cv2.LINE_AA) 
    
    try:
        fps = 1/elapsed_time
        cv2.putText(filled_img, 'FPS : '+str(fps)[:5], (int((1/100)*width), int((5/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
            0.4,(255,255,255),1,cv2.LINE_AA)   
        print("fps : ", fps)
    except ZeroDivisionError:
        print("Error: Division by zero is not allowed.")
          
    # if(coeff[0] > 0 and curve < 5000):
    #     filled_img = setSymbols(turn_left_symbol, filled_img)
    #     cv2.putText(filled_img,'Right Curve Ahead', (int(( 1/100)*width), 
    #         int((16/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1,cv2.LINE_AA)
    #     cv2.putText(filled_img,'Curvature = '+str(curve)[:7]+' m', (int(( 1/100)*width), 
    #         int((19/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1,cv2.LINE_AA)
    # elif(coeff[0] < 0 and curve < 5000):
    #     filled_img = setSymbols(turn_right_symbol, filled_img)        
    #     cv2.putText(filled_img,'Left Curve Ahead', (int(( 1/100)*width), 
    #         int((16/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 ,(255,255,255),1,cv2.LINE_AA)
    #     cv2.putText(filled_img,'Curvature = '+str(curve)[:7]+' m', (int(( 1/100)*width), 
    #         int((19/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1,cv2.LINE_AA)
    # else:
    #     filled_img = setSymbols(straight_symbol, filled_img)
    #     cv2.putText(filled_img,'Keep Straight Ahead', (int(( 1/100)*width), 
    #         int((16/100)*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 ,(255,255,255),1,cv2.LINE_AA)




    # display end


    # cv2.putText(filled_img, f"FPS: {fps:.2f}",  (int((220/600)*width), int((20/338)*height)), cv2.FONT_HERSHEY_SIMPLEX, (float((0.5/600)*width)), (255, 0, 0), 2, cv2.LINE_AA)
    #             # int((70/338)*height)), cv2.FONT_HERSHEY_SIMPLEX, (float((0.5/600)*width)),(255,0,0),2,cv2.LINE_AA)
    # cv2.putText(filled_img, 'DETEKSI JALUR', (int((5/600)*width), int((20/338)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
    #             (float((0.5/600)*width)),(255,0,0),2,cv2.LINE_AA)
    # cv2.putText(filled_img, 'STATUS BERKENDARA', (int((380/600)*width), int((20/338)*height)), cv2.FONT_HERSHEY_SIMPLEX, 
    #             (float((0.5/600)*width)),(255,0,0),2,cv2.LINE_AA)
    


    cv2.imshow("hsv", filled_img)
    cv2.imshow("frame", warp)
    cv2.imshow("warp_frame", rewarp)

    out.write(filled_img)
    i+=1

    cv2.waitKey(1)

# Tutup video dan jendela tampilan
out.release()
video_capture.release()
cv2.destroyAllWindows()

