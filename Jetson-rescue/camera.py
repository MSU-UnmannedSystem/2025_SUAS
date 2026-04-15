import cv2
import numpy as np
import math

def main():
    cap = cv2.VideoCapture(0)

    if (not cap.isOpened()):
        print("Error: Could not open webcam.")
        exit()

    print("Vision pipeline active. Searching for 5-meter red bullseye...")

    while True:
        ret, frame = cap.read()
        if (not ret):
            print("Failed to grab frame.")
            break

        #Converting BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #Creating a Red Mask (Two ranges needed for red in HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # Combine=ing the masks
        red_mask = mask1 + mask2

        # Cleaning up the mask using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        #Finding Contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if (area < 500): 
                continue
            perimeter = cv2.arcLength(contour, True)
            if (perimeter == 0):
                continue
            
            # Circularity formula: 4 * pi * (Area / Perimeter^2)
            circularity = 4 * math.pi * (area / (perimeter * perimeter))

            # Allow some tolerance (0.7 to 1.2) for distortion
            if (0.7 < circularity < 1.2):
                
                # 5. Find the exact (X, Y) center using Image Moments
                M = cv2.moments(contour)
                if (M["m00"] != 0): 
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    print(f"Bullseye acquired at Target Pixel: ({cX}, {cY})")

                    # Draw the red bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    # Drawing a dot at the center
                    cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
                    
                    # Labeling the center coordinate
                    cv2.putText(frame, f"({cX}, {cY})", (cX - 25, cY - 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Displaying the frame
        cv2.imshow('Webcam Stream', frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
