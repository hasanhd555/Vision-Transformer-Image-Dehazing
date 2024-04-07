import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img
from skimage import exposure


def apply_ahe(img):
    # Convert image to float in range [0, 1]
    img_float = img.astype(np.float32) / 255.0

    # Apply AHE to each channel separately
    img_ahe = np.zeros_like(img_float)
    for i in range(img_float.shape[2]):
        img_ahe[:, :, i] = exposure.equalize_adapthist(img_float[:, :, i], clip_limit=0.03)

    # Convert image back to the original data type
    img_uint8 = (img_ahe).astype(img.dtype)

    return img_uint8

def white_balance(image):
  """
  This function performs white balance adjustment on an image using OpenCV.

  Args:
      image: The input image as a NumPy array.

  Returns:
      A NumPy array representing the white balanced image.
  """
  # Convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Find the average intensity of the grayscale image (assuming white areas have high intensity)
  avg_intensity = np.mean(gray)

  # Calculate a scaling factor to adjust the intensity of all color channels
  scale = 255.0 / avg_intensity

  # Create a new image for the white balanced output
  balanced_image = image.astype(np.float32)  # Convert to float32 for scaling

  # Multiply each color channel by the scaling factor
  balanced_image[:,:,0] *= scale  # Blue channel
  balanced_image[:,:,1] *= scale  # Green channel
  balanced_image[:,:,2] *= scale  # Red channel

  # Clip the pixel values to be within the range [0, 255] and convert back to uint8
  balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)

  return balanced_image

def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	# horizontal flip
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs


class PairLoader(Dataset):
	def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = os.path.join(data_dir, sub_dir)
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
		target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1

		# Apply AHE to source and target images
		source_img = white_balance(source_img)
		
		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)

		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}

class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		# Apply AHE to the image
		img = white_balance(img)

		return {'img': hwc_to_chw(img), 'filename': img_name}