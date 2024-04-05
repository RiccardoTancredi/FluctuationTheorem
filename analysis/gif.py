import glob
from PIL import Image
def make_gif(frame_folder, name="prova.gif"):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frames = frames + [frames[-1]]*100
    frame_one = frames[0]
    frame_one.save(name, format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

if __name__ == "__main__":
    make_gif("imgs/prova/0nM/10/")