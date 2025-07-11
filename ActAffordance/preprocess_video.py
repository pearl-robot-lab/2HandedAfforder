import cv2
import os
import json
from PIL import Image
from argparse import ArgumentParser

def process_video(video_path, text_prompt, output_path):
    # Open the video file using OpenCV
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Get total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = 0

    while True:
        # Read the next frame from the video
        ret, frame = video_capture.read()
        
        if not ret:
            break  # End of video

        # Convert frame (BGR to RGB) and save as PNG
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Pillow
        img = Image.fromarray(frame_rgb)  # Convert the frame to an image

        # Create folder for the current frame number
        frame_folder = os.path.join(output_path, str(frame_number))
        os.makedirs(frame_folder, exist_ok=True)

        # Save the frame as PNG
        img.save(os.path.join(frame_folder, 'inpainting.png'))

        # Create a JSON file for the narration text
        annotation = {
            "narration": text_prompt
        }
        with open(os.path.join(frame_folder, 'annotation.json'), 'w') as json_file:
            json.dump(annotation, json_file, indent=4)

        print(f"Processed frame {frame_number}/{total_frames}")
        frame_number += 1

    # Release the video capture object
    video_capture.release()
    print("Video processing complete!")

if __name__ == "__main__":
    # Example usage:
    parser = ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument('text_prompt')
    parser.add_argument('output_path')
    args = parser.parse_args()
    video_path = args.video_path
    text_prompt = args.text_prompt
    output_path = args.output_path
    # Process the video
    process_video(video_path, text_prompt, output_path)
