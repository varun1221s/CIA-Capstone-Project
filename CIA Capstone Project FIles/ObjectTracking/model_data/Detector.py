import cv2
import numpy as np
import time

# Set a random seed for numpy
np.random.seed(20)

class Detector:
	def __init__(self, videoPath, configPath, modelPath, classesPath, modelType, confThreshold, sThreshold, nmsThreshold, batchSize, inputImageSize, bValue):

		# Initialize instance variables
		self.videoPath = videoPath
		self.configPath = configPath
		self.modelPath = modelPath
		self.classesPath = classesPath
		self.modelType = modelType
		self.confThreshold = confThreshold
		self.sThreshold = sThreshold
		self.nmsThreshold = nmsThreshold
		self.batchSize = batchSize
		self.inputImageSize = inputImageSize
		self.bValue = bValue

		# Load the DNN model from the specified paths
		self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)

		# Set the input size for the model and configure the input
		inputSize = self.inputImageSize.get()

		self.net.setInputSize(int(inputSize),int(inputSize))		#  lowest fps 320, 320 // 224, 224 // 128, 128 highest fps
		if self.modelType == 'SSD':
			self.net.setInputScale(1.0/127.5)	# 1.0/127.5 or 98/25000
			self.net.setInputMean((127.5,127.5,127.5))	# 0,0,0 or 127.5,127.5,127.5
		
		elif self.modelType == 'YOLOv3' or 'YOLOv3-tiny':
			self.net.setInputScale(98/25000)	# 1.0/127.5 or 98/25000
			self.net.setInputMean((0,0,0))	# 0,0,0 or 127.5,127.5,127.5
		
		elif self.modelType == 'Cones':
			self.net.setInputScale(1.0/127.5)	# 1.0/127.5 or 98/25000
			self.net.setInputMean((127.5,127.5,127.5))	# 0,0,0 or 127.5,127.5,127.5

		elif self.modelType == 'Backpack':
			self.net.setInputScale(98/25000)	# 1.0/127.5 or 98/25000
			self.net.setInputMean((0,0,0))	# 0,0,0 or 127.5,127.5,127.5

		self.net.setInputSwapRB(True)

		# Call the readClasses function to load the class labels and colors
		self.readClasses()

	def readClasses(self):

		# Read the class labels from the specified path
		with open(self.classesPath, 'r') as f:
			self.classesList = f.read().splitlines()
		
		# Modify the classesList based on the model type
		if self.modelType == 'SSD':
			self.classesList.insert(0, '__Background__')

		elif self.modelType == 'YOLOv3':
			if '__Background__' in self.classesList:
				self.classesList.remove('__Background__')
		
		elif self.modelType == 'YOLOv3-tiny':
			if '__Background__' in self.classesList:
				self.classesList.remove('__Background__')

		elif self.modelType == 'Backpack':
			if '__Background__' in self.classesList:
				self.classesList.remove('__Background__')

		elif self.modelType == 'Cones':
			if '__Background__' in self.classesList:
				self.classesList.remove('__Background__')

		# Generate random colors for each class label
		self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

		#print(self.classesList)
	
	def onVideoBatch(self):
		cap = cv2.VideoCapture(self.videoPath)

		# Open the video file
		if (cap.isOpened()==False):
			print("Error")
			return
		
		# Read the first frame from the video
		#(success, image) = cap.read()

		batch_size = self.batchSize.get() # define the batch size
		frame_buffer = []
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		processed_frames = 0
		while True:
    		# read a batch of frames from the input source
			batch = []
			for i in range(batch_size):
				success, image = cap.read()
				if not success:
					break
				batch.append(image)
			
			if not batch:
				break

			# Add the progress update code here
			processed_frames += len(batch)
			progress = int((processed_frames / total_frames) * 100)
			print(f"Progress: {progress}%")

			results = []
			for image in batch:
				classLabelIDs, confidences, bboxs = self.net.detect(image, self.confThreshold.get())
				# convert the bounding boxes and confidence values to lists
				bboxs = list(bboxs)
				confidences = list(np.array(confidences).reshape(1, -1)[0])
				confidences = list(map(float, confidences))
				bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, self.sThreshold.get(), self.nmsThreshold.get())
				results.append((classLabelIDs, confidences, bboxs, bboxIdx))

			# Draws bounding boxes around the detected objects in the image, along with the class labels and confidence scores.
			for i in range(len(batch)):
				classLabelIDs, confidences, bboxs, bboxIdx = results[i]
				if bboxIdx is not None and len(bboxIdx) != 0:
					for j in bboxIdx:
						bbox = bboxs[j]
						classConfidence = confidences[j]
						classLabelID = classLabelIDs[j]
						classLabel = self.classesList[classLabelID]
						classColor = [int(c) for c in self.colorList[classLabelID]]

						displayText = "{}:{:.2f}".format(classLabel, classConfidence)

						x,y,w,h = bbox
						if self.bValue.get() == "Enabled":
							cv2.rectangle(batch[i], (x,y), (x+w, y+h), color = classColor, thickness = 1)
							cv2.putText(batch[i], displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
						
							lineWidth = min(int(w * .3), int(h * .3))

							# TOP LEFT CORNER OF BOUNDING BOX
							cv2.line(batch[i], (x,y), (x+lineWidth,y), classColor, thickness=4)
							cv2.line(batch[i], (x,y), (x,y+lineWidth), classColor, thickness=4)

							# TOP RIGHT CORNER OF BOUNDING BOX
							cv2.line(batch[i], (x+w,y), (x+w-lineWidth,y), classColor, thickness=4)
							cv2.line(batch[i], (x+w,y), (x+w,y+lineWidth), classColor, thickness=4)

							# BOTTOM LEFT CORNER OF BOUNDING BOX
							cv2.line(batch[i], (x,y+h), (x+lineWidth,y+h), classColor, thickness=4)
							cv2.line(batch[i], (x,y+h), (x,y+h-lineWidth), classColor, thickness=4)

							# BOTTOM RIGHT CORNER OF BOUNDING BOX
							cv2.line(batch[i], (x+w,y+h), (x+w-lineWidth,y+h), classColor, thickness=4)
							cv2.line(batch[i], (x+w,y+h), (x+w,y+h-lineWidth), classColor, thickness=4)
						else:
							cv2.rectangle(batch[i], (x,y), (x+w, y+h), color = classColor, thickness = 3)
							cv2.putText(batch[i], displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
							lineWidth = min(int(w * .3), int(h * .3))

				#cv2.putText(batch[i], "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

				frame_buffer.append(batch[i])
				
		
		startTime = 0
		paused = False
		frame_idx = 0
		while True:
			frame = frame_buffer[frame_idx]
			currentTime = time.time()
			fps = 1 / (currentTime - startTime)
			startTime = currentTime
			cv2.putText(frame, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
			cv2.imshow("Result", frame)

			if not paused:
				if frame_idx < len(frame_buffer) - 1:
					frame_idx += 1

			key = cv2.waitKey(20) & 0xFF
			if key == ord("q"):
				break
			elif key == ord("p"):
				paused = not paused
			elif key == ord("r"):
				frame_idx = max(0, frame_idx - 30)
			elif key == ord("f"):
				frame_idx = min(len(frame_buffer) - 1, frame_idx + 30)
			elif key == ord(" "):
				if frame_idx < len(frame_buffer) - 1:
					frame_idx += 1
				paused = True
	
			# CHECKS TO SEE IF WEBCAM IS ENABLED
			if self.videoPath == 0:
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
				
			# CHECKS TO SEE IF INPUT FILE IS VIDEO
			elif self.videoPath.endswith('.mp4'):
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
			
			# INPUT FILE IS A PICTURE
			else:
				key = cv2.waitKey(0) & 0xFF
				if key == ord("q"):
					break
				
		cv2.destroyAllWindows()
		cap.release()


	# Function to run object detection on a video
	def onVideo(self):
		cap = cv2.VideoCapture(self.videoPath)

		# Open the video file
		if (cap.isOpened()==False):
			print("Error")
			return
		
		# Read the first frame from the video
		(success, image) = cap.read()

		# Initialize variables for calculating FPS
		startTime = 0
		paused = False
		frame_idx = 0
		# Loop through all frames in the video
		while success:
			if not paused:
				# Calculate FPS for the current frame
				currentTime = time.time()
				fps = 1/(currentTime - startTime)
				startTime = currentTime
				# Run object detection on the current frame
				classLabelIDs, confidences, bboxs = self.net.detect(image, self.confThreshold.get())

				# Convert the bounding boxes and confidence values to lists
				bboxs = list(bboxs)
				confidences = list(np.array(confidences).reshape(1, -1)[0])
				confidences = list(map(float, confidences))
				bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, self.sThreshold.get(), self.nmsThreshold.get())

				# Draws bounding boxes around the detected objects in the image, along with the class labels and confidence scores.
				if len(bboxIdx) != 0:
					for i in range(0, len(bboxIdx)):

						bbox = bboxs[np.squeeze(bboxIdx[i])]
						classConfidence = confidences[np.squeeze(bboxIdx[i])]
						classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
						classLabel = self.classesList[classLabelID]
						classColor = [int(c) for c in self.colorList[classLabelID]]

						displayText = "{}:{:.2f}".format(classLabel, classConfidence)

						x,y,w,h = bbox

						cv2.rectangle(image, (x,y), (x+w, y+h), color = classColor, thickness = 2)
						cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
						
						lineWidth = min(int(w * .3), int(h * .3))

						if self.bValue.get() == "Enabled":
							# TOP LEFT CORNER OF BOUNDING BOX
							cv2.line(image, (x,y), (x+lineWidth,y), classColor, thickness=4)
							cv2.line(image, (x,y), (x,y+lineWidth), classColor, thickness=4)

							# TOP RIGHT CORNER OF BOUNDING BOX
							cv2.line(image, (x+w,y), (x+w-lineWidth,y), classColor, thickness=4)
							cv2.line(image, (x+w,y), (x+w,y+lineWidth), classColor, thickness=4)

							# BOTTOM LEFT CORNER OF BOUNDING BOX
							cv2.line(image, (x,y+h), (x+lineWidth,y+h), classColor, thickness=4)
							cv2.line(image, (x,y+h), (x,y+h-lineWidth), classColor, thickness=4)

							# BOTTOM RIGHT CORNER OF BOUNDING BOX
							cv2.line(image, (x+w,y+h), (x+w-lineWidth,y+h), classColor, thickness=4)
							cv2.line(image, (x+w,y+h), (x+w,y+h-lineWidth), classColor, thickness=4)
						else:
							cv2.rectangle(image, (x,y), (x+w, y+h), color = classColor, thickness = 3)
							cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
							lineWidth = min(int(w * .3), int(h * .3))

				cv2.putText(image, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
				cv2.imshow("Result", image)

			key = cv2.waitKey(20) & 0xFF
			if key == ord("q"):
				break
			elif key == ord("p"):
				paused = not paused
			elif key == ord("r"):
				frame_idx = -30
			elif key == ord("f"):
				frame_idx = 30
			elif key == ord(" "):
				paused = not paused
			
			if not paused:
				if frame_idx != 0:
					cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) +frame_idx)
					frame_idx = 0
				success, image = cap.read()

				if not success:  # If the video ends, set the video position to the last frame
					cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
					success, image = cap.read()

			# CHECKS TO SEE IF WEBCAM IS ENABLED
			if self.videoPath == 0:
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
			
			# CHECKS TO SEE IF INPUT FILE IS VIDEO
			elif self.videoPath.endswith('.mp4'):
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
			
			# INPUT FILE IS A PICTURE
			else:
				key = cv2.waitKey(0) & 0xFF
				if key == ord("q"):
					break

			#(success, image) = cap.read()
		cv2.destroyAllWindows()
		cap.release()
