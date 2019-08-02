from morph.centroidtracker import CentroidTracker
from morph.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')


print("[INFO] starting video stream...")
vs = VideoStream(src='udp://192.168.1.1:5600').start()
time.sleep(2.0)

writer = None

W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
total = 0


fps = FPS().start()

while True:
	
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	
	if args["input"] is not None and frame is None:
		break

	
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter('output.mp4', fourcc, 30,(W, H), True)

	status = "Waiting"
	rects = []

	if totalFrames % 30 == 0:
		status = "Detecting"
		trackers = []

		
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		objects_on_frame = 0

		for i in np.arange(0, detections.shape[2]):
			
			confidence = detections[0, 0, i, 2]

			
			if confidence > 0.4:
			
				idx = int(detections[0, 0, i, 1])

				if CLASSES[idx] != "person":
					continue

				objects_on_frame += 1

				
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")
				

				
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				trackers.append(tracker)

	else:

		for tracker in trackers:
			status = "Tracking"


			tracker.update(rgb)
			pos = tracker.get_position()


			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())


			rects.append((startX, startY, endX, endY))


	objects = ct.update(rects)


	for (objectID, centroid) in objects.items():
		to = trackableObjects.get(objectID, None)


		if to is None:
			to = TrackableObject(objectID, centroid)

		else:
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)


			if not to.counted:
				if direction < 0 and centroid[1] < H // 2:
					total += 1
					to.counted = True

				elif direction > 0 and centroid[1] > H // 2:
					total += 1
					to.counted = True


		trackableObjects[objectID] = to

		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
		cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)

	info = [
		("Total", total),
		("Objs on frame", objects_on_frame)
	]


	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


	if writer is not None:
		writer.write(frame)


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF


	if key == ord("q"):
		break

	totalFrames += 1
	fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
	writer.release()


if not args.get("input", False):
	vs.stop()


else:
	vs.release()

cv2.destroyAllWindows()
