A supplement Document for this GitHub Repo.

1. Create a Folder for Images
     -  Create a folder and name it ‘images’.
2. Run FFmpeg
     - Run the following command in your terminal: ``ffmpeg -i "video path/name" -vf "fps=1/2" -frames:v 20 output%02d.png``
     - This command will extract frames from the video file at a rate of one frame every 2 seconds, up to a maximum of 20 frames. The frames will be saved as PNG files with names like output01.png, output02.png, etc. Please change this to match the number of frames you would like to extract.
3. Install LabelImg & Labelme
      -Run the following command in your terminal to install labelImg and labelme:
`pip install labelImg` & `pip install labelme`
4. Annotate Images
Use labelImg or labelme to annotate the images. Save the annotated images to a new folder called ‘labels’.
5. Create a ‘Test’ Folder:
Create a new folder called ‘test’. Move the ‘labels’ and ‘images’ folders into the ‘test’ folder.
6. Create a ‘Dataset’ Folder:
Create another folder called ‘dataset’. Move the ‘test’ folder into the ‘dataset’ folder.
7. Install labelme2yolo: ``pip install labelme2yolo``
Run the following command in your terminal to convert labels to Yolo formate:
``labelme2yolo --json_dir 'Path to test folder'``
This will create a new folder called ‘YOLODataset’ inside of your test folder.
8. Install Ultralytics
Run the following command in your terminal to install ultralytics:
'pip install ultralytics'
9. Train the Model
Run the following command in your terminal to train the model:
``yolo task=segment mode=train epochs=200 data=dataset.yaml model=yolov8m-seg.pt imgsz=640 batch=-1``
        - This command will train a YOLO model on your dataset for 200 epochs. Adjust epochs as needed. The model configuration is specified by yolov8m-seg.pt, the input image size is 640, and the batch size is determined automatically based on your GPU memory.
 - If you want to define ``patience: Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent overfitting by stopping training when performance plateaus.`` add `patience= ###`
 - For example, ``yolo task=segment mode=train epochs=200 patience=200 data=dataset.yaml model=yolov8m-seg.pt imgsz=640 batch=-1``
For more information about Yolo model training, please visit: https://docs.ultralytics.com/modes/train/#train-settings
- Please ensure that you replace the placeholders with your actual file paths and names where necessary. Also, make sure to run these commands in the appropriate directory.
