import cv2
import glob
import os

def video_to_frames(video_path, output_dir, frame_rate=1):
    """
    Converts a video to frames and saves the frames in the specified directory.
   
    Parameters:
    - video_path: str, path to the input video file
    - output_dir: str, directory where the frames will be saved
    - frame_rate: int, save every 'frame_rate' frames
    """
   
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    # Open the video file
    cap = cv2.VideoCapture(video_path)
   
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
   
    frame_count = 0
   
    while True:
        ret, frame = cap.read()
        if not ret:
            break
       
        # Save the frame if it's the correct frame according to the frame_rate
        if frame_count % frame_rate == 0:
            frame_file = os.path.join(output_dir, f"{frame_count:04d}.jpg")
            cv2.imwrite(frame_file, frame)
            #print(f"Saved: {frame_file}")
       
        frame_count += 1
   
    # Release the video file
    cap.release()
    print("Video processing completed.")

# Example usage:

# folders = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '17']
actions = ['WalkDog']


vid_path_name = ''
vid_name = os.path.basename(vid_path_name)[:-4]
path_output = '/images/human36m/' + vid_name
os.mkdir(path_output)
video_to_frames(vid_path_name, path_output)
print(vid_path_name)
