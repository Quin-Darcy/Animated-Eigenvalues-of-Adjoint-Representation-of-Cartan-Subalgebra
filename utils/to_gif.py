#!/usr/bin/env python3
import glob
from PIL import Image
import re


class Gif:
    def __init__(self, plots_folder, save_folder, dim):
        self.plots_folder = plots_folder+"/*.png"
        self.gif_name = save_folder+"/sl"+str(dim)+"(C).gif"

        self.make_gif()

    def make_gif(self):
        frames = []
        im = [img for img in glob.glob(self.plots_folder)]
        im.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+',                  var)])
        for i in im:
            temp = Image.open(i)
            new_frame = temp.copy()
            frames.append(new_frame)
            temp.close()
        for i in range(len(frames)-2, -1, -1):
            frames.append(frames[i])

        frames[0].save(self.gif_name, format='GIF', append_images=frames[1:],
            save_all=True, duration=40, loop=0)
        print('\nDONE. GIF saved to: '+str(self.gif_name))
