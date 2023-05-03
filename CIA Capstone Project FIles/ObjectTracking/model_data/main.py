from Detector import *
import os
from tkinter import *
from tkinter import filedialog
import customtkinter
 
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")
root = customtkinter.CTk()

root_folder = os.path.dirname(os.path.abspath(__file__))
print(root_folder)

def modelData(videoPath, model):
    vPath = videoPath
    configPath = None
    modelPath = None
    classesPath = None
    
    if getModel(model) == "SSD MobileNet":
        configPath = os.path.join(root_folder, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
        modelPath = os.path.join(root_folder, "frozen_inference_graph.pb")
        classesPath = os.path.join(root_folder, "coco.names")
        modelType = 'SSD'
        confThreshold = conf_slider_var
        sThreshold = sThreshold_slider_var
        nmsThreshold = nmsThreshold_slider_var
        batchSize = batchSize_slider_var
        inputImageSize = inputImageSize_var
        bValue = bValue_var
        detector = Detector(vPath, configPath, modelPath, classesPath, modelType, confThreshold, sThreshold, nmsThreshold, batchSize, inputImageSize, bValue)
        if batchValue_var.get() == "Disabled":
            detector.onVideo()
        elif batchValue_var.get() == "Enabled" and vPath == 0:
            print("Cant Use Webcam when Batch Processing is Enabled")
        else:
            detector.onVideoBatch()
	
    elif getModel(model) == "YOLOv3":
        configPath = os.path.join(root_folder, "yolov3.cfg")
        modelPath = os.path.join(root_folder, "yolov3.weights")
        classesPath = os.path.join(root_folder, "cocoYOLO.names")
        modelType = 'YOLOv3'
        confThreshold = conf_slider_var
        sThreshold = sThreshold_slider_var
        nmsThreshold = nmsThreshold_slider_var
        batchSize = batchSize_slider_var
        inputImageSize = inputImageSize_var
        bValue = bValue_var
        detector2 = Detector(vPath, configPath, modelPath, classesPath, modelType, confThreshold, sThreshold, nmsThreshold, batchSize, inputImageSize, bValue)
        if batchValue_var.get() == "Disabled":
            detector2.onVideo()
        elif batchValue_var.get() == "Enabled" and vPath == 0:
            print("Cant Use Webcam when Batch Processing is Enabled")
        else:
            detector2.onVideoBatch()

    elif getModel(model) == "Cones":
        configPath = os.path.join(root_folder, "yolov3cones.cfg")
        modelPath = os.path.join(root_folder, "yolov3cones.weights")
        classesPath = os.path.join(root_folder, "yolov3cones.names")
        modelType = 'Cones'
        confThreshold = conf_slider_var
        sThreshold = sThreshold_slider_var
        nmsThreshold = nmsThreshold_slider_var
        batchSize = batchSize_slider_var
        inputImageSize = inputImageSize_var
        bValue = bValue_var
        detector3 = Detector(vPath, configPath, modelPath, classesPath, modelType, confThreshold, sThreshold, nmsThreshold, batchSize, inputImageSize, bValue)
        if batchValue_var.get() == "Disabled":
            detector3.onVideo()
        elif batchValue_var.get() == "Enabled" and vPath == 0:
            print("Cant Use Webcam when Batch Processing is Enabled")
        else:
              detector3.onVideoBatch()
	
    elif getModel(model) == "Backpack":
        configPath = os.path.join(root_folder, "yolov3-ourmodel.cfg")
        modelPath = os.path.join(root_folder, "yolov3_training_last.weights")
        classesPath = os.path.join(root_folder, "ourmodel.names")
        modelType = 'Backpack'
        confThreshold = conf_slider_var
        sThreshold = sThreshold_slider_var
        nmsThreshold = nmsThreshold_slider_var
        batchSize = batchSize_slider_var
        inputImageSize = inputImageSize_var
        bValue = bValue_var
        detector4 = Detector(vPath, configPath, modelPath, classesPath, modelType, confThreshold, sThreshold, nmsThreshold, batchSize, inputImageSize, bValue)
        if batchValue_var.get() == "Disabled":
            detector4.onVideo()
        elif batchValue_var.get() == "Enabled" and vPath == 0:
            print("Cant Use Webcam when Batch Processing is Enabled")
        else:
              detector4.onVideoBatch()
	
    elif getModel(model) == "YOLOv3-tiny":
        configPath = os.path.join(root_folder, "yolov3-tiny.cfg")
        modelPath = os.path.join(root_folder, "yolov3-tiny.weights")
        classesPath = os.path.join(root_folder, "cocoYOLO.names")
        modelType = 'YOLOv3-tiny'
        confThreshold = conf_slider_var
        sThreshold = sThreshold_slider_var
        nmsThreshold = nmsThreshold_slider_var
        batchSize = batchSize_slider_var
        inputImageSize = inputImageSize_var
        bValue = bValue_var
        detector5 = Detector(vPath, configPath, modelPath, classesPath, modelType, confThreshold, sThreshold, nmsThreshold, batchSize, inputImageSize, bValue)
        if batchValue_var.get() == "Disabled":
            detector5.onVideo()
        elif batchValue_var.get() == "Enabled" and vPath == 0:
            print("Cant Use Webcam when Batch Processing is Enabled")
        else:
              detector5.onVideoBatch()
    
def clickButton():
    model = model_menu.get()
    videoPath = filedialog.askopenfilename()
    
    if videoPath:
        # Process the video file with OpenCV
        modelData(videoPath, model)

def clickButtonWebcam():
	global buttonClicked
	buttonClicked = not buttonClicked

	videoPath = 0
	model = model_menu.get()

	modelData(videoPath, model)

def change_appearance_mode_event(new_appearance_mode: str):
	customtkinter.set_appearance_mode(new_appearance_mode)
	print(new_appearance_mode)

def getModel(selected_value):
	print(f"Selected model: {selected_value}")
	return selected_value

def getInputSize(selected_value):
	print(f"Input Size: {selected_value}")
	return selected_value

## WINDOW SETTINGS
root.geometry(f"{1190}x{720}")
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure((2, 3), weight=1)
root.grid_rowconfigure((0,1,2), weight=1)

# SIDE BAR W/ OPTION MENU
root.sidebar_frame = customtkinter.CTkFrame(root, width=140, corner_radius=0)
root.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
root.sidebar_frame.grid_rowconfigure(4, weight=1)

sidebar_label = customtkinter.CTkLabel(root.sidebar_frame, text="Settings", font=("Helvetica", 14), pady=10)
sidebar_label.pack(fill="x")

root.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(root.sidebar_frame, values=["Dark", "Light", "System"], command=change_appearance_mode_event)
root.appearance_mode_optionemenu.pack(fill="x", padx=10, pady=(0, 10))

## CHOOSE BEAUTIFY IN SIDE BAR
root.sidebar_frame.model_label = customtkinter.CTkLabel(root.sidebar_frame, text="Beautify", font=("Helvetica", 14), pady=10)
root.sidebar_frame.model_label.pack(fill="x")

bValue_var = customtkinter.StringVar()
beautify_menu = customtkinter.CTkOptionMenu(root.sidebar_frame, variable=bValue_var, values=["Enabled", "Disabled"], command=lambda value: bValue_var.set(value))
beautify_menu.pack(fill="x", padx=10, pady=(0, 10))
beautify_menu.set("Disabled")

## CHOOSE BATCH PROCESSING IN SIDE BAR
root.sidebar_frame.model_label = customtkinter.CTkLabel(root.sidebar_frame, text="Batch Processing", font=("Helvetica", 14), pady=10)
root.sidebar_frame.model_label.pack(fill="x")

batchValue_var = customtkinter.StringVar()
batch_menu = customtkinter.CTkOptionMenu(root.sidebar_frame, variable=batchValue_var, values=["Enabled", "Disabled"], command=lambda value: batchValue_var.set(value))
batch_menu.pack(fill="x", padx=10, pady=(0, 10))
batch_menu.set("Disabled")

## CHOOSE BATCH PROCESSING SIZE
batchSize_label_var = customtkinter.StringVar()
batchSize_label = customtkinter.CTkLabel(root.sidebar_frame, textvariable=batchSize_label_var, font=("Helvetica", 14), pady=10)
batchSize_label.pack(padx=10, pady=(0, 10))

batchSize_slider_var = customtkinter.IntVar(value=1)
batchSize_slider = customtkinter.CTkSlider(root.sidebar_frame, from_=1, to=64, variable=batchSize_slider_var, command=lambda value: batchSize_slider_var.set(int(value)))
batchSize_slider.pack(padx=10, pady=(0,10))

batchSize_label_var.set(f"Batch Size: {batchSize_slider_var.get():.0f}")
batchSize_slider_var.trace_add("write", lambda name, index, mode, var=batchSize_slider_var, lbl=batchSize_label_var: lbl.set(f"Batch Size: {var.get():.0f}"))

## CHOOSE THE MODEL IN SIDE BAR
root.sidebar_frame.model_label = customtkinter.CTkLabel(root.sidebar_frame, text="Model", font=("Helvetica", 14), pady=10)
root.sidebar_frame.model_label.pack(fill="x")

model_menu = customtkinter.CTkOptionMenu(root.sidebar_frame, values=["SSD MobileNet", "YOLOv3", "YOLOv3-tiny", "Cones", "Backpack"], command=lambda value: getModel(value))
model_menu.pack(fill="x", padx=10, pady=(0, 10))

## CHOOSE THE CONFIDENCE IN SIDE BAR
conf_label_var = customtkinter.StringVar()
conf_label = customtkinter.CTkLabel(root.sidebar_frame, textvariable=conf_label_var, font=("Helvetica", 14), pady=10)
conf_label.pack(padx=10, pady=(0, 10))

conf_slider_var = customtkinter.DoubleVar(value=0.5)
conf_slider = customtkinter.CTkSlider(root.sidebar_frame, from_=0.1, to=0.99, variable=conf_slider_var, command=lambda value: conf_slider_var.set(float(value)))
conf_slider.pack(padx=10, pady=(0,10))

## CHOOSE THE SCORE THRESHOLD IN SIDE BAR
sThreshold_label_var = customtkinter.StringVar()
sThreshold_label = customtkinter.CTkLabel(root.sidebar_frame, textvariable=sThreshold_label_var, font=("Helvetica", 14), pady=10)
sThreshold_label.pack(padx=10, pady=(0, 10))

sThreshold_slider_var = customtkinter.DoubleVar(value=0.5)
sThreshold_slider = customtkinter.CTkSlider(root.sidebar_frame, from_=0.1, to=0.99, variable=sThreshold_slider_var, command=lambda value: sThreshold_slider_var.set(float(value)))
sThreshold_slider.pack(padx=10, pady=(0,10))

## CHOOSE THE NMS THRESHOLD IN SIDE BAR
nmsThreshold_label_var = customtkinter.StringVar()
nmsThreshold_label = customtkinter.CTkLabel(root.sidebar_frame, textvariable=nmsThreshold_label_var, font=("Helvetica", 14), pady=10)
nmsThreshold_label.pack(padx=10, pady=(0, 10))

nmsThreshold_slider_var = customtkinter.DoubleVar(value=0.5)
nmsThreshold_slider = customtkinter.CTkSlider(root.sidebar_frame, from_=0.1, to=0.99, variable=nmsThreshold_slider_var, command=lambda value: nmsThreshold_slider_var.set(float(value)))
nmsThreshold_slider.pack(padx=10, pady=(0,10))

# set the label text to the initial value of the slider
conf_label_var.set(f"Confidence Value: {conf_slider_var.get():.2f}")

# update the label text whenever the slider value changes
conf_slider_var.trace_add("write", lambda name, index, mode, var=conf_slider_var, lbl=conf_label_var: lbl.set(f"Confidence Value: {var.get():.2f}"))

# set the label text to the initial value of the slider
sThreshold_label_var.set(f"Score Threshold: {sThreshold_slider_var.get():.2f}")

# update the label text whenever the slider value changes
sThreshold_slider_var.trace_add("write", lambda name, index, mode, var=sThreshold_slider_var, lbl=sThreshold_label_var: lbl.set(f"Score Threshold: {var.get():.2f}"))

# set the label text to the initial value of the slider
nmsThreshold_label_var.set(f"NMS Threshold: {nmsThreshold_slider_var.get():.2f}")

# update the label text whenever the slider value changes
nmsThreshold_slider_var.trace_add("write", lambda name, index, mode, var=nmsThreshold_slider_var, lbl=nmsThreshold_label_var: lbl.set(f"NMS Threshold: {var.get():.2f}"))


## CHOOSE INPUT IMAGE SIZE IN SIDE BAR
root.sidebar_frame.model_label = customtkinter.CTkLabel(root.sidebar_frame, text="Input Image Size", font=("Helvetica", 14), pady=10)
root.sidebar_frame.model_label.pack(fill="x")

inputImageSize_var = customtkinter.StringVar()
inputImageSize_menu = customtkinter.CTkOptionMenu(root.sidebar_frame, variable=inputImageSize_var, values=["128", "224", "256", "320", "416", "512", "608", "832"], command=lambda value: getInputSize(value))
inputImageSize_menu.pack(fill="x", padx=10, pady=(0, 10))
inputImageSize_menu.set("128")


## TAB SETTINGS
tabview = customtkinter.CTkTabview(root, width=250)
tabview.grid(row=0, column=2, padx=20, pady=(20, 0), sticky="")
tabview.add("Object Tracking Through File")
tabview.add("Object Tracking Through Webcam")
tabview.tab("Object Tracking Through File").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
tabview.tab("Object Tracking Through Webcam").grid_columnconfigure(0, weight=1)

## BUTTON FOR OBJECT TRACKING THROUGH INPUT FILE
browse_button = customtkinter.CTkButton(tabview.tab("Object Tracking Through File"), text="Browse", command=clickButton)
browse_button.grid(row=1, column=0, padx=20, pady=(20, 20), sticky="")

label_tab_2 = customtkinter.CTkLabel(tabview.tab("Object Tracking Through File"), text="Press Q to Stop The Video\nPress P to Pause The Video\nPress R to Rewind The Video\nPress F to Fast Forward The Video\nHold Spacebar to go Frame by Frame")
label_tab_2.grid(row=0, column=0, padx=20, pady=20, sticky="")

## ENTRY FOR OBJECT TRACKING THROUGH INPUT FILE
#entryBox = customtkinter.CTkEntry(tabview.tab("Object Tracking Through Input File"), placeholder_text="Path To File", width=300)
#entryBox.grid(row=1, column=0, padx=20, pady=(20, 20), sticky="")

## BUTTON FOR OBJECT TRACKING THROUGH WEBCAM
buttonClicked = False
myButton = customtkinter.CTkButton(tabview.tab("Object Tracking Through Webcam"), text="Press Button to Enable Webcam", command=clickButtonWebcam)
myButton.grid(row=2, column=0, padx=20, pady=(20, 10), sticky="")
label_tab_3 = customtkinter.CTkLabel(tabview.tab("Object Tracking Through Webcam"), text="Press Q to Stop Webcam\nMust Have Batch Processing Disabled")
label_tab_3.grid(row=0, column=0, padx=20, pady=20, sticky="")

## CREATE TABVIEW2 FOR RECOMMENDED SETTINGS
tabview2 = customtkinter.CTkTabview(root, width=900, height = 300)
tabview2.grid(row=1, column=2, padx=20, pady=(20, 0), sticky="")

## Reflective recommended settings
tabview2.add("Our Project")
reflective_settings_label = customtkinter.CTkLabel(tabview2.tab("Our Project"), text="The CIA Labs' Object Tracking and Locating Project is an innovative research initiative focused on developing a comprehensive solution for identifying and tracking objects in a scene. The primary goal is to create a custom model that can effectively recognize objects, assign names to them, and track their movement using bounding boxes. This project aims to leverage existing state-of-the-art models and fuse them with the newly developed model to achieve superior object tracking and locating capabilities, ultimately enhancing situational awareness and intelligence analysis.", font=("Helvetica", 14), wraplength=850, pady=10, padx=10)
reflective_settings_label.pack(fill="x")

## SSD Mobilenet recommended settings
tabview2.add("SSD MobileNet")
ssd_settings_label = customtkinter.CTkLabel(tabview2.tab("SSD MobileNet"), text="SSD mobilenet is a powerful object detection algorithm, but it requires some tuning to achieve good results. Here are some recommended settings:\n\n- Confidence threshold: 0.5\n- Non-maximum suppression threshold: 0.5\n- Input image size: 300x300\n- Number of channels: 3\n- Mean values: (127.5, 127.5, 127.5)\n- Scale factor: 0.007843\n\nNote that these settings may vary depending on your specific use case and dataset.", font=("Helvetica", 14), pady=10, padx=10)
ssd_settings_label.pack(fill="x")

## YOLOv3 recommended settings
tabview2.add("YOLOv3")
yolov3_settings_label = customtkinter.CTkLabel(tabview2.tab("YOLOv3"), text="YOLOv3 is a powerful object detection algorithm, but it requires some tuning to achieve good results. Here are some recommended settings:\n\n- Confidence threshold: 0.5\n- Non-maximum suppression threshold: 0.5\n- Input image size: 416x416\n- Number of channels: 3\n- Mean values: (0,0,0)\n- Scale factor: 0.00392\n\nNote that these settings may vary depending on your specific use case and dataset.", font=("Helvetica", 14), pady=10, padx=10)
yolov3_settings_label.pack(fill="x")

## YOLOv3-tiny recommended settings
tabview2.add("YOLOv3-tiny")
yolov3tiny_settings_label = customtkinter.CTkLabel(tabview2.tab("YOLOv3-tiny"), text="YOLOv3-tiny is a powerful object detection algorithm, but it requires some tuning to achieve good results. Here are some recommended settings:\n\n- Confidence threshold: 0.5\n- Non-maximum suppression threshold: 0.5\n- Input image size: 416x416\n- Number of channels: 3\n- Mean values: (0,0,0)\n- Scale factor: 0.00392\n\nNote that these settings may vary depending on your specific use case and dataset.", font=("Helvetica", 14), pady=10, padx=10)
yolov3tiny_settings_label.pack(fill="x")

## Reflective recommended settings
tabview2.add("Cones")
reflective_settings_label = customtkinter.CTkLabel(tabview2.tab("Cones"), text="Our Model\nYOLOv3 is a powerful object detection algorithm, but it requires some tuning to achieve good results. Here are some recommended settings:\n\n- Confidence threshold: 0.5\n- Non-maximum suppression threshold: 0.5\n- Input image size: 416x416\n- Number of channels: 3\n- Mean values: (0,0,0)\n- Scale factor: 0.00392\n\nNote that these settings may vary depending on your specific use case and dataset.", font=("Helvetica", 14), pady=10, padx=10)
reflective_settings_label.pack(fill="x")

## Reflective recommended settings
tabview2.add("Backpack")
reflective_settings_label = customtkinter.CTkLabel(tabview2.tab("Backpack"), text="Our Model", font=("Helvetica", 14), pady=10, padx=10)
reflective_settings_label.pack(fill="x")

## ----------------------- QUESTIONS TAB SETTINGS-----------------------
tabview2.add("Questions")
# Create the questions frame
questions_canvas = Canvas(tabview2.tab("Questions"))
questions_frame = customtkinter.CTkFrame(questions_canvas)
questions_scrollbar = Scrollbar(tabview2.tab("Questions"), orient="vertical", command=questions_canvas.yview)
questions_canvas.configure(yscrollcommand=questions_scrollbar.set)

# Pack the scrollbar and canvas
questions_scrollbar.pack(side="right", fill="y")
questions_canvas.pack(side="left", fill="both", expand=True)
questions_canvas.create_window((0, 0), window=questions_frame, anchor="nw")

questions_canvas.bind('<Configure>', lambda e: questions_canvas.configure(scrollregion= questions_canvas.bbox("all"))) # ALLOWS USER TO SCROLL UP AND DOWN

questions_text = ["Confidence Value: This is the minimum probability score that an object must have in order to be considered a valid detection.A higher value will result in fewer false positives, but may also miss some true positives. A lower value will result in more detections, but may also increase the number of false positives.",
		  "Score Threshold: Increasing the score_threshold can lead to more accurate detections, but may also result in missed detections",
		  "NMS Threshold: The threshold used to suppress overlapping detections of the same object. Higher values will suppress more detections, resulting in fewer false positives, but may also miss some true positives. A lower value will result in more detections, but may also increase the number of false positives.",
		  "Input image size: This is the size of the input image that the algorithm expects. YOLOv3 is designed to work with a fixed input size, which should be a multiple of 32. A larger input size will generally result in better accuracy, but may also be slower to process.",
		  "Number of channels: This is the number of color channels in the input image. Most images are in RGB format, which has three channels.",
		  "Mean values: This is the mean value of each color channel in the input image. Subtracting the mean value from each pixel helps to normalize the input data, making it easier for the algorithm to learn.",
		  "Scale factor: This is the scale factor used to normalize the input image pixel values. Dividing the pixel values by this factor scales the values to the range [0,1], which is more suitable for the neural network to process. This particular scale factor is used because it corresponds to the inverse of 255, which is the maximum value of an 8-bit color channel.",
		  "Batch Processing & Batch Size: When using batch processing in video processing, the frames are processed in batches, rather than individually. In the case of a batch size of 16, the processing is done on groups of 16 frames at a time. This can lead to the impression that frames are being skipped, as the output is being produced at a lower frequency than the input frames. For example, if the input video is 30 frames per second and a batch size of 16 is used, it would take approximately half a second (16/30) to process a batch of frames. During that half-second period, 14 frames would be processed, and 16 frames would be skipped until the next batch is processed. This skipping of frames is typically not noticeable to the human eye, especially when the processing is done quickly and the resulting output video is played back at a normal speed."]
for i in range(len(questions_text)):
    label = customtkinter.CTkLabel(questions_frame, text=f"{questions_text[i]}\n", wraplength=850, anchor="w")
    label.pack(pady=10, padx=10, fill="both")

	
if __name__ == '__main__':
	root.title('Object Tracker GUI')
	root.mainloop()

