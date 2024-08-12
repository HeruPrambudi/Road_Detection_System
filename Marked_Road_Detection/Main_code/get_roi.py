import cv2
import csv

point_index = 0 
points = []

frame = cv2.imread('roi.jpg')
frame = cv2.resize(frame, (640, 480))

def draw_circle(event,x,y,flags,param):
    global point_index, points
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(frame,(x,y),2,(255,0,0),-1)
        points.append((x, y))
        point_index += 1
        print('x : ', x, 'y : ', y, '\n')


while(1):
    cv2.imshow('frame', frame)
    cv2.setMouseCallback("frame", draw_circle)
    if cv2.waitKey(1) & 0xFF == ord('q') or point_index == 4:
        break

save = input("Save ROI : ") 
if(save == 'y'):
    csv_filename = 'roi_points.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for point in points:
            csv_writer.writerow(point)
