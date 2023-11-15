import tkinter as tk
from tkinter import filedialog
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# global variables
WIDTH = 2560 *9//16
HEIGHT = 2560
FONT = cv.FONT_HERSHEY_COMPLEX
FONTSCALE = 1.5
THICKNESS = 3
time_int = 5 # interval in seconds
time_start = 0 # seconds

# Getting file
# root = tk.Tk()
# root.withdraw()

# file_path = filedialog.askopenfilename()
cap = cv.VideoCapture("VIDEOFILES/IMG_9590.MOV")

# get fps
fps = cap.get(cv.CAP_PROP_FPS)


area_measurements = np.array([[0,0]], dtype=object)

# Loop until the end of the video
frame_no = 0
while cap.isOpened():
 
    # Capture frame-by-frameV
    ret, frame = cap.read()
    if not ret:
        break
    
    # frame = cv.resize(frame, (WIDTH, HEIGHT), interpolation = cv.INTER_CUBIC)

 
    # conversion of BGR to grayscale is necessary to apply this operation
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
    # get contours
    _, thresh = cv.threshold(gray, 155, 156, cv.THRESH_BINARY)
    thresh = cv.GaussianBlur(thresh, (11,11), 0)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # isolate contour of balloon (assuming ballon contour has largest area)
    try:
        balloon_contour = sorted(contours, key=cv.contourArea, reverse=True)[0]
    except:
        break
    
    # balloon area
    balloon_area = cv.contourArea(balloon_contour)
    
    # adding balloon contour to image
    cv.drawContours(frame, [balloon_contour], -1, (0,255, 0), -1, cv.LINE_AA)
    
    # adding text (with balloon area) to image
    cv.putText(frame, "Area: {:.3}kpx^2".format(balloon_area/1e6), (50, 100), FONT, 
        FONTSCALE, (255,255,255), THICKNESS, cv.LINE_AA)
    
    
    # Display the resulting frame
    cv.namedWindow('Video of Balloon')
    cv.resizeWindow('Video of Balloon', WIDTH, HEIGHT)
    cv.imshow('Video of Balloon', frame)
 
    # define q as the exit button
    if cv.waitKey(1) == ord('q'):
        break
    
    time = cap.get(cv.CAP_PROP_POS_FRAMES)/fps - time_start
    
    area_measurements = np.concatenate((area_measurements, np.array([[time, balloon_area]])))
    
    frame_no += time_int * fps
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_no)
 
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv.destroyAllWindows()


# plotting
area_measurements = area_measurements[1:].transpose() #ignores the first element, since it was a placeholder (0,0)
plt.plot(area_measurements[0], area_measurements[1]/1e6)

plt.title('Balloon Area vs Time')
plt.ylabel('Area of Balloon [$kpx^2$]')
plt.xlabel('Time [$s$]')

plt.grid(True, which='major', alpha=.5)
plt.grid(True, which='minor', alpha=.2, linestyle=':')
plt.minorticks_on()

plt.savefig("Figures/Balloon area vs time -- 2nd run", bbox_inches='tight', dpi=300)
plt.show()

np.savetxt("CSVs/2nd area measurements.csv", area_measurements, delimiter=",")

# print(area_measurements)
