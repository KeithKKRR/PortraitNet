from PIL import Image

from datasets.data_augmentation import DataAugmentation

img = Image.open("datasets/EG1800/testing/00001.png")
mask = Image.open("datasets/EG1800/testing/00001_matte.png").convert("1")
img_res, mask_res, edge = DataAugmentation.process_image(img, mask)
img.show()
mask.show()