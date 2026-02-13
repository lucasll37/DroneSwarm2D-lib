"""
create_video.py

This script reads PNG frames from a specified directory, sorts them by frame number,
and creates a video (MP4 format) using imageio.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import shutil
from typing import Optional
import imageio.v2 as imageio

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_frame_number(filename: str) -> int:
    """
    Extracts the frame number from a filename.
    
    Assumes filenames follow the pattern: <prefix>_<number>.png
    
    Args:
        filename (str): The filename to extract the number from.
        
    Returns:
        int: The extracted frame number.
    """
    return int(filename.split('_')[1].split('.')[0])

# -----------------------------------------------------------------------------
# Main Video Creation Function
# -----------------------------------------------------------------------------
def create_video_from_frames(
    frames_dir: str,
    output_dir: str = "./video",
    fps: int = 30,
    codec: str = "libx264",
    remove_frames: bool = False
) -> Optional[str]:
    """
    Creates a video from PNG frames in a specified directory.
    
    Args:
        frames_dir (str): Path to the directory containing PNG frame images.
        output_dir (str): Directory where the output video will be saved. 
                         Defaults to "./video".
        fps (int): Frames per second for the output video. Defaults to 30.
        codec (str): Video codec to use. Defaults to "libx264".
        remove_frames (bool): If True, removes the frames directory after 
                             creating the video. Defaults to False.
    
    Returns:
        Optional[str]: Path to the created video file, or None if an error occurred.
    """
    # Validate that the frames directory exists
    if not os.path.exists(frames_dir):
        print(f"Error: Directory '{frames_dir}' does not exist.")
        return None
    
    # Extract the type of scenery from the folder name
    folder_name = os.path.basename(os.path.normpath(frames_dir))
    type_of_cenary = folder_name.split("-")[-1]
    
    # List all PNG filenames in the frames directory
    filenames = [fn for fn in os.listdir(frames_dir) if fn.endswith(".png")]
    
    if not filenames:
        print(f"Error: No PNG files found in '{frames_dir}'.")
        return None
    
    # Sort filenames by frame number
    filenames = sorted(filenames, key=get_frame_number)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create video writer
    video_filename = os.path.join(output_dir, f"{type_of_cenary}.mp4")
    writer = imageio.get_writer(video_filename, fps=fps, codec=codec)
    
    # Process each frame and add to the video
    for fn in filenames:
        path = os.path.join(frames_dir, fn)
        try:
            img = imageio.imread(path)
            writer.append_data(img)
        except Exception as e:
            print(f"Error processing {fn}: {e}")
            continue
    
    writer.close()
    print(f"Video saved as: {video_filename}")
    
    # Optionally remove the frames directory
    if remove_frames:
        try:
            shutil.rmtree(frames_dir)
            print(f"Frames directory '{frames_dir}' removed.")
        except Exception as e:
            print(f"Error removing frames directory: {e}")
    
    return video_filename

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Main function for command-line usage.
    Reads frames from './tmp' directory structure.
    """
    tmp_dir = "./tmp"
    
    if not os.path.exists(tmp_dir):
        print(f"Error: Directory '{tmp_dir}' does not exist.")
        return
    
    # Get the first folder in tmp directory
    folders = os.listdir(tmp_dir)
    if not folders:
        print(f"Error: No folders found in '{tmp_dir}'.")
        return
    
    folder = folders[0]
    frames_dir = os.path.join(tmp_dir, folder)
    
    # Create video from frames
    create_video_from_frames(
        frames_dir=frames_dir,
        output_dir="./video",
        fps=30,
        codec="libx264",
        remove_frames=False  # Set to True to remove frames after video creation
    )

if __name__ == "__main__":
    main()