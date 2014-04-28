import numpy as np
import cv2
import cv

# This code identifies the landing pad in a video and prints out the x-y coordinates (origin in the center of the frame)
# of the landing pad within the FOV.

def nothing(x):
	pass

# NOTE: Insert 0 as argument into VideoCapture to use video feed from camera feed
# NOTE: Insert absolute path to filename to read/use a video file instead of camera feed
cap = cv2.VideoCapture(0)

# Create trackbar window
cv2.namedWindow('Trackbars')

# Create Hue, Saturation, and Value trackbars
cv2.createTrackbar('Upper Hue slider', 'Trackbars', 244, 255, nothing)
cv2.createTrackbar('Lower Hue slider', 'Trackbars', 50, 255, nothing)
cv2.createTrackbar('Upper Saturation slider', 'Trackbars', 220, 255, nothing)
cv2.createTrackbar('Lower Saturation slider', 'Trackbars', 50, 255, nothing)
cv2.createTrackbar('Upper Value slider', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Lower Value slider', 'Trackbars', 180, 255, nothing)

# Save output to a file
fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
out = cv2.VideoWriter('teste_vid_out.avi',fourcc, 20.0, (640,480))

while(True):
	
	# Capture frame-by-frame
	_, frame = cap.read()


	# Our operations on the frame come here
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Capture the image masking
	upper = np.array([cv2.getTrackbarPos('Upper Hue slider', 'Trackbars'), \
		cv2.getTrackbarPos('Upper Saturation slider', 'Trackbars'), \
		cv2.getTrackbarPos('Upper Value slider', 'Trackbars')],np.uint8)
	lower = np.array([cv2.getTrackbarPos('Lower Hue slider', 'Trackbars'), \
		cv2.getTrackbarPos('Lower Saturation slider', 'Trackbars'), \
		cv2.getTrackbarPos('Lower Value slider', 'Trackbars')],np.uint8)

	# Threshold the HSV image
	mask = cv2.inRange(hsv, lower, upper)

	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(frame, frame, mask=mask )


	# Find contours within thresholded image
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	height, width, depth = frame.shape


	try:
		# Finds largest contour by area and captures point list of contour
		areas = [cv2.contourArea(b) for b in contours]
		max_index = np.argmax(areas)
		cnt = contours[max_index]

		# Calculates centroid of largest contour (cnt). NOTE: Breaks if threshold trackbar is set to 255. Not sure why?
		M = cv2.moments(cnt)

		# These are some debugging print statements
		print('m00:', int(M['m00']))
		print('m10:', int(M['m10']))
		print('m01:', int(M['m01']))

		if M['m00'] != 0:
			centroid_x = int(M['m10']/M['m00'])
			centroid_y = int(M['m01']/M['m00'])
		else:
			centroid_x = centroid_x
			centroid_y = centroid_y

		print('Working')

		# Writes the text to show the x-y coords. on the image
		xtext = "Centroid x-coord.: %02.3f" %(100*(float(centroid_x)/width - .5))
		ytext = "Centroid y-coord.: %02.3f" %(100*(float(centroid_y)/height - .5))

		# Draw contours
		cv2.drawContours(frame, cnt, -1, (0,255,0), 3)


		# Draws the smallest possible rectangle that encloses the largest contour in the image
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

		# Add cross-hairs for reference
		vertpts = np.array([[width/2,height/2 + height/10],[width/2,height/2 - height/10]], np.int32)
		horpts = np.array([[width/2 + height/10,height/2],[width/2 - height/10,height/2]], np.int32)

		vertpts = vertpts.reshape((-1,1,2))
		horpts = horpts.reshape((-1,1,2))

		cv2.polylines(frame,[vertpts, horpts],True,(0,255,255))

		# Prints information on the screen
		font = cv2.FONT_HERSHEY_SIMPLEX

		cv2.putText(frame, xtext, (10, (height - height/8) + 25), font, .5, (255, 255, 255), 2)
		cv2.putText(frame, ytext, (10, (height - height/8) + 40), font, .5, (255, 255, 255), 2)

	except ValueError:
		nothing
		print('No contour detected')


	# Display the resulting frame(s)
	cv2.imshow('frame', frame)
	cv2.imshow('mask', mask)
	if cv2.waitKey(1) & 0xFF == 27:
		break
	


# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
