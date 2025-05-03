from ultralytics import YOLO

# load the model - this will load the weight of the model
model = YOLO("yolov8x")


# pass the data to the model 
results = model.predict('input_videos/video_1.mp4', save=True)

print(results)
print("=====================")
for box in results[0].boxes:
    print(box)
