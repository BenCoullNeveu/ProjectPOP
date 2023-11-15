import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# global variables
WIDTH = 1280
HEIGHT = 853
WINDOW = "IMAGE"
THRESH_MAX = 150
MAX_VAL = 150

CONTOUR_COLOR = (255,0,0)
ELLIPSE_COLOR = (0,255,0)
LINE_COLOR = (0,0,255)

FONT = cv.FONT_HERSHEY_COMPLEX
FONTSCALE = 5
THICKNESS = 4

m_per_pixel = 0.001 # m/pixel


def open_image(filename):
    # opening image using PIL
    raw = Image.open(filename)
    # converting image to numpy.array for opencv (converting to BGR too)
    img = cv.cvtColor(np.array(raw), cv.COLOR_RGB2BGR)
    # grayscale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray, img

def get_contours(image, blurred_threshold=True, kernel=(11,11)):
    _, thresh = cv.threshold(image, 15, 100, cv.THRESH_BINARY)
    if blurred_threshold:
        thresh = cv.GaussianBlur(thresh, kernel, 0)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return contours

def drawLine(image, vx, vy, x, y, color=LINE_COLOR, thickness=5):
    left = int((-x * vy/vx) + y)
    right = int(((image.shape[1]-x) *vy/vx) + y)
    cv.line(image, (image.shape[1]-1, right), (0, left), color, thickness)
    
def drawNormal(image, vx, vy, x, y, color=LINE_COLOR, thickness=5):
    nx = vy
    ny = -vx
    left = int((-x*ny/nx) + y)
    right = int(((image.shape[1]-x) *ny/nx) + y)
    cv.line(image, (image.shape[1]-1, right), (0, left), color, thickness)

def plotview(image, window_name=WINDOW, filename=None, width=WIDTH, height=HEIGHT, flags=cv.WINDOW_NORMAL):
    cv.namedWindow(window_name, flags)
    cv.imshow(window_name, image)
    cv.resizeWindow(window_name, width, height)
    if filename is not None:
        cv.imwrite('Balloon with Contours and Fitted Ellipse.png', image)
        
def align_image(grayscale, image, iterations=1):
    if iterations > 1:
        cnts = sorted(get_contours(grayscale), key= cv.contourArea, reverse=True)
        line = cv.fitLine(cnts[0], cv.DIST_L2, 0, 0.01, 0.01)
        h, w = image.shape[:2]
        center = (w//2, h//2)
        vx, vy = line[:2]
        theta = np.degrees(np.arctan(vx/ vy))[0]
        if vx >= 0:
            theta *= -1
        M = cv.getRotationMatrix2D(center, theta, 1) # rotation matrix
        image = cv.warpAffine(image, M, (w, h))
        grayscale = cv.warpAffine(grayscale, M, (w, h))
        iterations -= 1
        grayscale, image = align_image(grayscale, image, iterations)
    return grayscale, image

# preparing images
gray_init, img_init = open_image('IMG_0367.CR2')

# aligning images
gray, img = align_image(gray_init, img_init, iterations=5) # sweetspot seems to be 5 iterations, though there are some problems (see image)

# getting the contours and sorting them
contours = sorted(get_contours(gray), key= cv.contourArea, reverse=True)

# getting balloon contour, its best-fitting ellipse and its best-fit line
balloon_contour = contours[0] #assuming the balloon has the largest contour area
balloon_ellipse = cv.fitEllipse(balloon_contour)
balloon_line = cv.fitLine(balloon_contour, cv.DIST_L2, 0, 0.01, 0.01) # [vx,vy,x,y]

# drawing the contour, ellipse and line on the image
img_overlay = img.copy()
cv.drawContours(img_overlay, [balloon_contour], 0, CONTOUR_COLOR, 5, lineType=cv.LINE_AA)
cv.ellipse(img_overlay, balloon_ellipse, ELLIPSE_COLOR, 5)
drawLine(img_overlay, *balloon_line)    
drawNormal(img_overlay, balloon_line[0], balloon_line[1], 3260, 2300)

# create blank mask
blank = np.zeros(img.shape[:2], np.uint8)

# create images out of the blanks and animating the intersection line
conts, lines = blank.copy(), blank.copy()
cv.drawContours(conts, contours, 0, 100, -1)
drawNormal(lines, *balloon_line, color=100, thickness=1)

# getting intersection
intersection = np.logical_and(conts, lines)
intersection = np.multiply(intersection, 1).astype(np.uint8) *254 # converting from bool to int

# finding the length of the intersection line
houghlines = cv.HoughLinesP(intersection,1, np.pi/180, 50, None, 50, 10)    
x0, y0, x1, y1 = houghlines[0][0] #starting and end points of the intersection line


for i in gray.shape[0]:
    # previous diameter (updated at every iteration)
    d_prev = 0
    
    # create blank mask
    blank = np.zeros(img.shape[:2], np.uint8)
    
    # create images out of the blanks and animating the intersection line
    conts, lines = blank.copy(), blank.copy()
    cv.drawContours(conts, contours, 0, 100, -1)
    drawNormal(lines, balloon_line[0], balloon_line[1], i, balloon_line[3], color=100, thickness=1)

    # getting intersection
    intersection = np.logical_and(conts, lines)
    intersection = np.multiply(intersection, 1).astype(np.uint8) *254 # converting from bool to int

    # finding the length of the intersection line
    houghlines = cv.HoughLinesP(intersection,1, np.pi/180, 50, None, 50, 10)
    x0, y0, x1, y1 = houghlines[0][0] #starting and end points of the intersection line
    
    d_curr = abs(x1-x0)*m_per_pixel # length of intersection line in meters, assuming the image has been rotated correctly s.t. the line is 1 pixel thick
    
    # calculating volume
    
    



# combining images
combined = (intersection + 1) * gray.copy()

# adding text
cv.putText(img_overlay, 'Contour', (4500,500), FONT, 
        FONTSCALE, CONTOUR_COLOR, THICKNESS, cv.LINE_AA) 
cv.putText(img_overlay, 'Fitting Ellipse', (4500,700), FONT, 
        FONTSCALE, ELLIPSE_COLOR, THICKNESS, cv.LINE_AA) 
cv.putText(img_overlay, 'Fitting Line', (4500,900), FONT, 
        FONTSCALE, LINE_COLOR, THICKNESS, cv.LINE_AA) 

# plotting the view
plotview(img_overlay)
plotview(combined, window_name='Intersection')
cv.waitKey(0)
cv.destroyAllWindows()

print(balloon_line[0], balloon_line[1])