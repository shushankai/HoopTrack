import cv2
import numpy as np
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width



def draw_triangle(frame,bbox,color):
    """
    Draws a filled triangle on the given frame at the specified bounding box location.

    Args:
        frame (numpy.ndarray): The frame on which to draw the triangle.
        bbox (tuple): A tuple representing the bounding box (x, y, width, height).
        color (tuple): The color of the triangle in BGR format.

    Returns:
        numpy.ndarray: The frame with the triangle drawn on it.
    """
    y= int(bbox[1])
    x,_ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x,y],
        [x-10,y-20],
        [x+10,y-20],
    ])
    cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

    return frame




def draw_eclipse(frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center ,_ = get_center_of_bbox(bbox)

    width = get_bbox_width(bbox)

    cv2.ellipse(
        img= frame, 
        center= (x_center, y2), 
        axes= (width, int(0.35*width)),
        angle= 0, 
        startAngle= -45, 
        endAngle= 235, 
        color= color,
        thickness= 2, 
        lineType=cv2.LINE_4
    )

    # for putting tracking id 
    rectangle_width = 40 
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + 15
    y2_rect = (y2 + rectangle_height // 2) + 15


    if track_id is not None:
       cv2.rectangle(frame,
                        (int(x1_rect),int(y1_rect) ),
                        (int(x2_rect),int(y2_rect)),
                        color,
                        cv2.FILLED)


       x1_text = x1_rect + 15 # co ordinate to put the text 
       if track_id> 99: # if the track_id is 3 digit then move the co-ordinate for the text 10 pixel down
           x1_text -=10
       
       cv2.putText(img= frame,
            text = f'{track_id}',
            org=(x1_text, int(y1_rect + 15)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale= 0.6, 
            color=(0,0,0),
            thickness=2)
    return frame