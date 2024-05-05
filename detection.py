import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def detect_tumor(image_path):
    # Load image
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    X = img.shape[0] # image's height is used to scale the structuring elements utilized in morphology operations
    copy = np.copy(img) # it's a good practice to work on a copy rather than on the original image itself

    hist, bins = np.histogram(copy.flatten(),256,[0,256])
    # plt.stem(hist, use_line_collection=True)
    # plt.show()

    # First enhancement
    blur = cv.GaussianBlur(img, (5, 5), 2)
    enh = cv.add(img, (cv.add(blur, -100)))
    # plt.imshow(enh, cmap='gray', vmin=0, vmax=255)
    # plt.title("FIRST ENHANCEMENT")
    # plt.show()

    hist2, bins2 = np.histogram(enh.flatten(),256,[0,256])
    # plt.stem(hist2, use_line_collection=True)
    # plt.show()

    # Denoising
    median = cv.medianBlur(enh, 5)
    # plt.imshow(median, cmap='gray', vmin=0, vmax=255)
    # plt.title("DENOISED")
    # plt.show()

    # Morphological Gradient
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    gradient = cv.morphologyEx(median, cv.MORPH_GRADIENT, kernel)
    # plt.imshow(gradient, cmap='gray', vmin=0, vmax=255)
    # plt.title("MORPHOLOGICAL GRADIENT")
    # plt.show()

    # Second enhancement
    enh2 = cv.add(median, gradient)
    # plt.imshow(enh2, cmap='gray', vmin=0, vmax=255)
    # plt.title("SECOND ENHANCEMENT")
    # plt.show()

    hist3, bins3 = np.histogram(enh2.flatten(),256,[0,256])
    # plt.stem(hist3, use_line_collection=True)
    # plt.show()

    # First thresholding
    t = np.percentile(enh2, 85)
    ret, th = cv.threshold(enh2, t, 255, cv.THRESH_BINARY)
    # plt.imshow(th, cmap='gray', vmin=0, vmax=255)
    # plt.title("FIRST THRESHOLDING")
    # plt.show()

    # Morphology operations
    kernel_c = cv.getStructuringElement(cv.MORPH_ELLIPSE,(int((5*X)/100),int((5*X)/100))) #
    kernel_e = cv.getStructuringElement(cv.MORPH_ELLIPSE,(int((3*X)/100),int((3*X)/100))) #
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE,(int((7*X)/100),int((7*X)/100)))
    # plt.figure(figsize=(10,10),constrained_layout = True)

    opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel_e) # to eliminate small uninteresting structures
    # plt.subplot(221),plt.imshow(opening, cmap='gray', vmin=0, vmax=255)
    # plt.title("1. FIRST OPENING")

    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel_c) # to merge the remaining structures that may have been divided
    # plt.subplot(222),plt.imshow(closing, cmap='gray', vmin=0, vmax=255)
    # plt.title("2. CLOSING")

    erosion = cv.erode(closing,kernel_e,iterations = 1)
    # plt.subplot(223),plt.imshow(erosion, cmap='gray', vmin=0, vmax=255)
    # plt.title("3. FIRST EROSION")

    dilation = cv.dilate(erosion,kernel_e,iterations = 1)
    # plt.subplot(224),plt.imshow(dilation, cmap='gray', vmin=0, vmax=255)
    # plt.title("4. DILATION")

# Masking
    masked = cv.bitwise_and(copy, copy, mask=dilation)
    # plt.imshow(masked, cmap='gray', vmin=0, vmax=255)
    # plt.title("MASKED")
    # plt.show()

# Second round of morphology operations
    s_erosion = cv.erode(masked,kernel,iterations = 1)
    # plt.subplot(121),plt.imshow(s_erosion, cmap='gray', vmin=0, vmax=255)
    # plt.title("1. SECOND EROSION")

    final = cv.morphologyEx(s_erosion, cv.MORPH_OPEN, ker)
    # plt.subplot(122),plt.imshow(final, cmap='gray', vmin=0, vmax=255)
    # plt.title("2. SECOND OPENING")
    # plt.show()

# Third enhancement    
    blur3 = cv.GaussianBlur(final,(3,3),0)
    enh3 = cv.add(final,(cv.add(blur3,-100)))
    # plt.imshow(enh3, cmap='gray', vmin=0, vmax=255)
    # plt.title("THIRD ENHANCEMENT")
    # plt.show()

## Second thresholding
    upper = np.percentile(enh3,92)
    res = cv.inRange(enh3, 0, upper)
    # plt.imshow(res, cmap='gray', vmin=0, vmax=255)
    # plt.title("SECOND THRESHOLDING")
    # plt.show()

    # Final morphology step
    fin = cv.morphologyEx(res, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(int((7*X)/100),int((7*X)/100))))
    plt.imshow(fin, cmap='gray', vmin=0, vmax=255)
    # plt.title("LAST CLOSING")
    # plt.show()
    
# Contouring
    copy_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    contours, hierarchy = cv.findContours(fin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) > 1:
        cnt = contours[1]
        if len(contours) > 2:
            cv.drawContours(copy_rgb, contours, 2, (0, 0, 255), 3)
            plt.imshow(copy_rgb)
            plt.title("DETECTED TUMOR")
            plt.show()
        else:
            cv.drawContours(copy_rgb, contours, 1, (0, 0, 255), 3)
            plt.imshow(copy_rgb)
            plt.title("DETECTED TUMOR")
            plt.show()

        # area = int(cv.contourArea(cnt))
        # perimeter = int(cv.arcLength(cnt, True))

    #     print("Area:", area, "px")
    #     print("Perimeter:", perimeter, "px")
    # else:
    #     print("No tumor detected")