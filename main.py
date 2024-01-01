import cv2 as cv
from cv2 import aruco
import numpy as np

marker_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)

param_markers = aruco.DetectorParameters_create()

cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    if marker_corners:
        centrep=[]
        for ids, corners in zip(marker_IDs, marker_corners):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            centre=((top_right+bottom_left+bottom_right+top_left)/4)
            centrep.append(centre)
            if ids == 3 or 4:
                cv.line(frame,(int(centre[0]),int(centre[1])),(int(centrep[0][0]),int(centrep[0][1])),(0,255,0),2)
                #highlight the mid point of the line and print the pixel
                cv.circle(frame,(int(centre[0]),int(centre[1])),5,(0,0,255),-1)
                cv.circle(frame,(int(centrep[0][0]),int(centrep[0][1])),5,(0,0,255),-1)
                median=[(centre[0]+centrep[0][0])/2,(centre[1]+centrep[0][1])/2]
                
                cv.circle(frame,(int(median[0]),int(median[1])),5,(0,0,255),-1)
            cv.putText(
                frame,
                f"id: {ids[0]}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (200, 100, 0),
                2,
                cv.LINE_AA,
            )
            # print(ids, "  ", corners)
    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
