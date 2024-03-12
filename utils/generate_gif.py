import glob
import os

from PIL import Image

def create_gif(image_sequence, output_gif, duration=50):
    frames = []
    for image_file in image_sequence:
        print(image_file)
        if os.path.exists(image_file):
            frames.append(Image.open(image_file).resize((192*2, 108*2)))
        else:
            print(f"File {image_file} not found.")
    
    if frames:
        frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        print(f"GIF generated: {output_gif}")
    else:
        print("No frames found. GIF not generated.")

if __name__ == "__main__":
    src_image_dir = r"D:\qidi\bytetrack\out\build\x64-Debug\bytetrack_demo\visualization"
    image_sequence = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    output_gif = "hand.gif"  
    create_gif(image_sequence, output_gif)
