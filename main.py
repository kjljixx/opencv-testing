import cv2
import numpy as np

# runPipeline() is called every frame by Limelight's backend.
def runPipeline(original_image, llrobot):
    largestContour = np.array([[]])
    llpython = [0,0,0,0,0,0,0,0]

    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HLS)
    filtered_image = image
    filtered_image = cv2.inRange(filtered_image, np.array([15, 0, 100]),
                                        np.array([30, 40, 255]))
    filtered_image = cv2.bitwise_and(filtered_image, filtered_image, mask = cv2.adaptiveThreshold(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,35,0))
    filtered_image = cv2.bitwise_and(image, image, mask = filtered_image)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    # filtered_image = cv2.erode(filtered_image, kernel, iterations=1)
    edges = cv2.Canny(image=filtered_image, threshold1=100, threshold2=200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.ximgproc.thinning(edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if len(contours) > 0:
        largestContour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largestContour)
        llpython = [1,x,y,w,h,9,8,7]

    sample_contours = []
    for i in contours:
        if(cv2.contourArea(i) > 1000):
            sample_contours.append(i)

    image = cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
    cv2.drawContours(image, sample_contours, -1, 255, 2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_HLS2BGR)
    image = image
       
    # make sure to return a contour,
    # an image to stream,
    # and optionally an array of up to 8 values for the "llpython"
    # networktables array
    return np.array([[]]), image, llpython
