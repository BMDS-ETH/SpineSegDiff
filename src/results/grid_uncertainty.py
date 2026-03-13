
from pathlib import Path
from PIL import Image, ImageOps
def make_grid(images, rows, cols):

    imgs = [im.rotate(90, expand=True)  for im in images]
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*(w+10), rows*(h+10)))
    for i, image in enumerate(imgs):
        grid.paste( ImageOps.expand(image, border=5, fill='white'), box=(i%cols*w, i//cols*h))
        #grid.paste( ImageOps.expand(image, border=5, fill='white'), box=(i%cols*h, i//cols*w))

        #grid.paste(image, box=(i%cols*w, i//cols*h))
        #grid.paste( ImageOps.expand(image, border=5, fill='white'), box=(i//cols*w, i%cols*w))
    return grid

vis_path = Path('results/visualization-heatmaps')
print([i for i in sorted(vis_path.rglob('*SPIDER_T1wT2w/*.png'))])
images=[Image.open(i.as_posix()) for i in sorted(vis_path.rglob('*SPIDER_T1wT2w/*.png'))]

make_grid(images, rows=3, cols = 2).save(vis_path /'heatmaps_overview.png')