import cv2
import numpy as np
import logging
import os
import glob
import pickle


def cameraCalibration(image_dir):

    assert os.path.exists(image_dir), "%s does not exist" % (image_dir)

    fileList = glob.glob("%s/*.jpg" % (image_dir))
    assert len(fileList) > 0, "No calibration images found"

    nx, ny = 9, 6

    pattern_points = np.zeros(shape=(nx * ny, 3), dtype=np.float32)
    pattern_points[:, :2] = np.indices(dimensions=(nx, ny)).T.reshape(-1, 2)

    objectPoints = []
    imagePoints = []
    for fileName in fileList:
        image = cv2.imread(fileName)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape[:2]
        found, corners = cv2.findChessboardCorners(gray, (nx, ny))
        if not found:
            continue

        vis = image.copy()
        cv2.drawChessboardCorners(vis, (nx, ny), corners, True)

        imagePoints.append(corners.reshape(-1, 2))
        objectPoints.append(pattern_points)

    assert len(objectPoints) > 0, "chessboard not found"
    assert len(imagePoints) > 0, "chessboard not found"

    rms, mtx, dist, rvec, tvec = cv2.calibrateCamera(objectPoints=objectPoints,
                                                     imagePoints=imagePoints,
                                                     imageSize=(w, h),
                                                     cameraMatrix=None,
                                                     distCoeffs=None)

    with open('camera_calib.p', mode='wb') as fp:
        pickle.dump({'rms': rms, 'mtx': mtx, 'dist': dist, 'rvec': rvec, 'tvec': tvec}, fp)

    with open("camera_calib.p", mode='rb') as fp:
        calib = pickle.load(fp)
        logging.info("RMS: %s" % (calib['rms']))
        logging.info("Camera Matrix: %s" % (calib['mtx']))
        logging.info("Distortion Coefficient: %s" % (calib['dist']))
        logging.info("RVEC: %s" % (calib['rvec']))
        logging.info("TVEC: %s" % (calib['tvec']))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="Directory containing calibration images")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        cameraCalibration(image_dir=args.image_dir)
    except Exception as e:
        logging.error(str(e))
