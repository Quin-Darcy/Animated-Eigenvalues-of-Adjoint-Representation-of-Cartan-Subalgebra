from PIL import Image
import math


class Plot:
    def __init__(self, data, frame) -> None:
        self.xy_range = None

        self.frame = frame
        self.points = [d[0] for d in data]
        self.colors = [d[1] for d in data]
        self.size = int(math.sqrt(len(self.points) / 1.0))
        self.plot = Image.new('RGB', (self.size, self.size), color='black')

        self.set_xy_range()
        self.make_image()

    def set_xy_range(self) -> None:
        horiz = [d[0] for d in self.points]
        vertz = [d[1] for d in self.points]
        width = abs(max(horiz) - min(horiz))
        height = abs(max(vertz) - min(vertz))

        self.xy_range = (width + 0.000001, height + 0.000001)

    def make_image(self) -> None:
        px = self.plot.load()
        color_index = 0
        for point in self.points:
            x = math.floor(self.size * (point[0]-self.xy_range[0]/2) / self.xy_range[0])
            y = math.floor(self.size * (point[1]-self.xy_range[1]/2) / self.xy_range[1])
            px[x, y] = self.colors[color_index]
            color_index += 1

        self.plot.save('Plots/plot'+str(self.frame)+'.png')
