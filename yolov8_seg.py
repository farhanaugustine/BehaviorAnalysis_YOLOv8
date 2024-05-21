from ultralytics import YOLO
import cv2
import numpy as np
import csv
# 1. Load Model
model = YOLO(r'C:\Users\Lin\Desktop\Behavior Analysis Models\YOLO Model_segmentation\OFT_V1.pt')
# Load video
video_path=r'C:\Users\Lin\Desktop\Behavior Analysis Models\YOLO Model_segmentation\20230904_Mouse_196_OFT_720p_30fps_60sec.mp4'
cap= cv2.VideoCapture(video_path)
ret = True

# Initialize frame counter
frame_counter = 0

# Create a VideoWriter object to save the mask video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_mask = cv2.VideoWriter('mask_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
out_frame = cv2.VideoWriter('frame_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Define multiple ROIs (replace with your actual coordinates)
rois = [
    {'top_left': (240, 150), 'bottom_right': (440, 300), 'entry_count': 0, 'exit_count': 0, 'prev_position_inside': False, 'entry_frame': 0, 'exit_frame': 0, 'total_frames': 0, 'total_distance': 0, 'prev_position': None},
    {'top_left': (740, 150), 'bottom_right': (940, 300), 'entry_count': 0, 'exit_count': 0, 'prev_position_inside': False, 'entry_frame': 0, 'exit_frame': 0, 'total_frames': 0, 'total_distance': 0, 'prev_position': None},
    {'top_left': (240, 500), 'bottom_right': (440, 650), 'entry_count': 0, 'exit_count': 0, 'prev_position_inside': False, 'entry_frame': 0, 'exit_frame': 0, 'total_frames': 0, 'total_distance': 0, 'prev_position': None},
    {'top_left': (740, 500), 'bottom_right': (940, 650), 'entry_count': 0, 'exit_count': 0, 'prev_position_inside': False, 'entry_frame': 0, 'exit_frame': 0, 'total_frames': 0, 'total_distance': 0, 'prev_position': None},
    # Add more ROIs as needed
]

# Create a CSV file and write the header
csv_data = [["Frame", "Center_X", "Center_Y"]]

# Read frames
while ret:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, save=False) #iou=0.2,

    # Get the size of the frame
    height, width, _ = frame.shape

    # Draw the ROIs on the frame
    for roi in rois:
        cv2.rectangle(frame, roi['top_left'], roi['bottom_right'], (0, 255, 0), 2)

    # Check if there are any detections in the current frame
    if results:
        for result in results:
            if result.masks is not None:
                for mask in result.masks:
                    # Convert the mask data to a binary image
                    binary_mask = (mask.data.cpu().numpy()[0] > 0).astype('uint8')

                    # Resize the binary mask to the original frame size
                    binary_mask_resized = cv2.resize(binary_mask, (width, height))

                    # Convert the binary mask to a 3-channel image
                    binary_mask_3ch = cv2.cvtColor(binary_mask_resized, cv2.COLOR_GRAY2BGR)

                    # Label each connected component in the binary image
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_resized)

                    # Get the largest connected component (excluding the background)
                    largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

                    # Calculate the center of the largest connected component
                    center_x, center_y = centroids[largest_component_index].astype(int)

                    # Scale the center coordinates to the original frame size
                    center_x = int(center_x / binary_mask_resized.shape[1] * width)
                    center_y = int(center_y / binary_mask_resized.shape[0] * height)

                    # Invert the y-coordinate
                    center_y1 = height - center_y
                    center_x1 = center_x - width

                    # Check if the object's centroid is inside any ROI
                    for roi in rois:
                        position_inside = roi['top_left'][0] <= center_x <= roi['bottom_right'][0] and roi['top_left'][1] <= center_y <= roi['bottom_right'][1]

                        # If the object has entered the ROI, record the entry frame and position
                        if position_inside and not roi['prev_position_inside']:
                            roi['entry_count'] += 1
                            roi['entry_frame'] = frame_counter
                            roi['prev_position'] = (center_x, center_y)

                        # If the object is inside the ROI, calculate the distance traveled since the last frame and add it to the total distance
                        if position_inside and roi['prev_position'] is not None:
                            distance = ((center_x - roi['prev_position'][0]) ** 2 + (center_y - roi['prev_position'][1]) ** 2) ** 0.5
                            roi['total_distance'] += distance

                        # If the object has exited the ROI, record the exit frame and calculate the total frames
                        if not position_inside and roi['prev_position_inside']:
                            roi['exit_count'] += 1
                            roi['exit_frame'] = frame_counter
                            roi['total_frames'] += roi['exit_frame'] - roi['entry_frame']

                        # Update the object's position
                        roi['prev_position_inside'] = position_inside
                        roi['prev_position'] = (center_x, center_y)

                    # Draw a circle at the center of the mask on the frame
                    cv2.circle(frame, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)

                    # Save the frame number and the center coordinates to the CSV data
                    csv_data.append([frame_counter, center_x1, center_y1])

                    # Save the mask as a video
                    colored_mask = cv2.applyColorMap(binary_mask_3ch *255, cv2.COLORMAP_JET)
                    out_mask.write(colored_mask)

    # Display the counts on the frame
    for i, roi in enumerate(rois, 1):
        cv2.putText(frame, f'ROI {i} Entries: {roi["entry_count"]}', (10, 20 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'ROI {i} Exits: {roi["exit_count"]}', (10, 100 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the frame with ROIs and counts as a video
    out_frame.write(frame)

    # Increment frame counter
    frame_counter += 1

# Write the CSV data to the file
with open('centers.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

cap.release()
out_mask.release()
out_frame.release()
cv2.destroyAllWindows()

# Print the counts for each ROI
for i, roi in enumerate(rois, 1):
    print(f'ROI {i}:')
    print('Entries:', roi['entry_count'])
    print('Exits:', roi['exit_count'])
    print('Total frames spent:', roi['total_frames'])
    if roi['total_frames'] > 0:  # Avoid division by zero
        average_velocity = roi['total_distance'] / roi['total_frames']
        print('Average velocity:', average_velocity, 'pixels per frame')

# Write the counts, total frames spent, and average velocity for each ROI to the CSV file
with open('centers.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    for i, roi in enumerate(rois, 1):
        if roi['total_frames'] > 0:  # Avoid division by zero
            average_velocity = roi['total_distance'] / roi['total_frames']
        else:
            average_velocity = 0
        writer.writerow([f'ROI {i}', roi['entry_count'], roi['exit_count'], roi['total_frames'], average_velocity])
