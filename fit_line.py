import cv2
import numpy as np
import logging
import os
import pickle
import time
import matplotlib.pyplot as plt


def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(gray)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255
    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 255
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)

    binary_output = np.zeros_like(grad_direction)
    binary_output[(grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 255
    return binary_output


def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]

    binary_output = np.zeros_like(S)  # placeholder line
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 255
    return binary_output


def unwarp_image(image):
    height, width = image.shape[:2]
    startTime = time.time()
    # crop src points
    xs1, ys1 = width * 0.20, height * 0.95
    xs2, ys2 = width * 0.47, height * 0.61
    xs3, ys3 = width * 0.53, height * 0.61
    xs4, ys4 = width * 0.84, height * 0.95
    src = np.float32([[xs1, ys1], [xs2, ys2], [xs3, ys3], [xs4, ys4]])
    # crop dst points
    xd1, yd1 = width * 0.35, height
    xd2, yd2 = width * 0.35, 0
    xd3, yd3 = width * 0.65, 0
    xd4, yd4 = width * 0.65, height
    dst = np.float32([[xd1, yd1], [xd2, yd2], [xd3, yd3], [xd4, yd4]])

    # transform perspective to bird's view
    M = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, (width, height))
    logging.info("unwarp_image took %f ms" % (round((time.time() - startTime) * 1000.0, 1)))
    return warped


def warp_image(image):
    height, width = image.shape[:2]
    startTime = time.time()
    # crop src points
    xs1, ys1 = width * 0.20, height * 0.95
    xs2, ys2 = width * 0.47, height * 0.61
    xs3, ys3 = width * 0.53, height * 0.61
    xs4, ys4 = width * 0.84, height * 0.95
    src = np.float32([[xs1, ys1], [xs2, ys2], [xs3, ys3], [xs4, ys4]])
    # crop dst points
    xd1, yd1 = width * 0.35, height
    xd2, yd2 = width * 0.35, 0
    xd3, yd3 = width * 0.65, 0
    xd4, yd4 = width * 0.65, height
    dst = np.float32([[xd1, yd1], [xd2, yd2], [xd3, yd3], [xd4, yd4]])

    # transform perspective to bird's view
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    logging.info("warp_image took %f ms" % (round((time.time() - startTime) * 1000.0, 1)))

    # for i in range(4):
    #     if i + 1 >= 4:
    #         (x2, y2) = src[0]
    #         (x1, y1) = src[3]
    #     else:
    #         (x2, y2) = src[i + 1]
    #         (x1, y1) = src[i]
    #     cv2.line(image, (x1, y1), (x2, y2), (0, 100, 0), 2)
    # for i in range(4):
    #     if i + 1 >= 4:
    #         (x2, y2) = dst[0]
    #         (x1, y1) = dst[3]
    #     else:
    #         (x2, y2) = dst[i + 1]
    #         (x1, y1) = dst[i]
    #     cv2.line(image, (x1, y1), (x2, y2), (200, 0, 0), 2)
    # cv2.imshow("lanes", image)
    return warped


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input image/video file")
    parser.add_argument("--calib", help="Calibration Matrix file (*.p) pickle file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    assert os.path.exists(args.input), "%s does not exist" % (args.input)
    assert os.path.exists(args.calib), "%s does not exist" % (args.calib)

    cap = cv2.VideoCapture(args.input)
    assert cap.isOpened(), "Failed to open %s" % (args.input)

    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with open(args.calib, mode='rb') as fp:
        calib = pickle.load(fp)

    mtx = calib['mtx']
    dist = calib['dist']

    for frameIdx in range(totalFrames):
        ret, frame = cap.read()
        if not ret:
            break

        startTime = time.time()

        undist = cv2.undistort(frame, mtx, dist)
        undist = cv2.GaussianBlur(undist, (3, 3), 0)
        cv2.imshow("undist", undist)

        # obtain bird's view
        warped_undist = warp_image(undist)
        cv2.imshow("warped", warped_undist)

        # color space - hls
        s_binary = hls_select(warped_undist, thresh=(10, 255))
        cv2.imshow("s_binary", s_binary)

        # gradient threshold
        sx_binary = abs_sobel_thresh(warped_undist, 'x', thresh=(20, 100))
        sy_binary = abs_sobel_thresh(warped_undist, 'y', thresh=(20, 100))
        cv2.imshow("sx_binary", sx_binary)
        cv2.imshow("sy_binary", sy_binary)

        # gradient direction
        dir_binary = dir_threshold(warped_undist, sobel_kernel=13, thresh=(0.4, 1.3))
        cv2.imshow("dir_binary", dir_binary)

        # magnitude gradient
        mag_binary = mag_thresh(warped_undist, sobel_kernel=9, thresh=(20, 200))
        cv2.imshow("mag_binary", mag_binary)

        # combine
        combine_binary = np.zeros_like(dir_binary)
        combine_binary[((sx_binary == 255) & (s_binary == 255)) | ((mag_binary == 255) & (dir_binary == 255))] = 255
        cv2.imshow("combine_binary", combine_binary)

        # Finding Lanes
        binary_warped = combine_binary.copy()
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        cv2.imshow("out_img", out_img)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(combine_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        # newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        newwarp = unwarp_image(color_warp)
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        cv2.imshow('Result', result)

        elapsedTime = time.time() - startTime
        cv2.putText(undist, "fps: %02d" % (int(1.0 / elapsedTime)), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2)

        key = cv2.waitKey(100)
        if key == ' ':
            key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    exit(0)
