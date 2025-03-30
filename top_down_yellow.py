import cv2
import numpy as np

# runPipeline() is called every frame by Limelight's backend.
def runPipeline(original_image, llrobot):
    largestContour = np.array([[]])
    llpython = [0,0,0,0,0,0,0,0]

    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HLS)
    filtered_image = image
    filtered_image = cv2.GaussianBlur(image, (29, 29), sigmaX=0, sigmaY=0)
    mask = cv2.inRange(filtered_image, np.array([10, 0, 100]),
                                        np.array([30, 90, 255]))
    filtered_image = cv2.bitwise_and(image, image, mask = mask)
    
    original_edges = cv2.Canny(image=cv2.split(filtered_image)[1], threshold1=20, threshold2=40)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    original_edges = cv2.dilate(original_edges, kernel)

    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(original_edges)
    sizes = stats[:, cv2.CC_STAT_AREA]
    min_size = 150

    # create empty output image with will contain only the biggest composents
    im_result = np.zeros_like(im_with_separated_blobs)

    # for every component in the image, keep it only if it's above min_size.
    # we start at 1 to avoid considering the background
    for index_blob in range(1, nb_blobs):
        if sizes[index_blob] >= min_size:
            im_result[im_with_separated_blobs == index_blob] = 255
    original_edges = im_result
    original_edges = original_edges.astype(np.uint8)

    edges = cv2.bitwise_not(original_edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.bitwise_and(edges, edges, mask = mask)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sample_contours = []
    if len(contours) > 0:
        largestContour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largestContour)
        llpython = [1,x,y,w,h,9,8,7]
    for i in contours:
        if(cv2.contourArea(i) > 5000):
            w_h_rat = cv2.minAreaRect(i)[1][0]/cv2.minAreaRect(i)[1][1]
            if((w_h_rat > 2.0 and w_h_rat < 3.0) or (w_h_rat > 0.33 and w_h_rat < 0.5)):
                sample_contours.append(i)
    boxed_sample_contours = []
    for i in sample_contours:
        rect = cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,(0,255,255),2)

    image = cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
    # cv2.drawContours(image, sample_contours, -1, 255, 2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_HLS2BGR)
    original_edges = cv2.cvtColor(original_edges, cv2.COLOR_GRAY2BGR)
    image = cv2.addWeighted(image, 1.0, original_edges, 0.0, 0)
       
    # make sure to return a contour,
    # an image to stream,
    # and optionally an array of up to 8 values for the "llpython"
    # networktables array
    return np.array([[]]), image, llpython
