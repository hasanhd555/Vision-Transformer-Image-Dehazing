import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img
import cv2
import numpy as np

def apply_clahe(img):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the input image.

    Args:
        img: Input image as a NumPy array.

    Returns:
        Image with CLAHE applied.
    """
    # Scale pixel values to [0, 255] and convert to uint8
    img = img * 255.0
    img = img.astype(np.uint8)
    
    # Split image into channels
    r, g, b = cv2.split(img)
    
    # Create a CLAHE object (Clip Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    # Apply CLAHE to each channel separately
    r_clahe = clahe.apply(r)
    g_clahe = clahe.apply(g)
    b_clahe = clahe.apply(b)
    
    # Merge the CLAHE enhanced channels
    img_clahe = cv2.merge((r_clahe, g_clahe, b_clahe))

    # Convert back to float32 and normalize to range [0, 1]
    img_clahe = img_clahe.astype(np.float32) / 255.0
    
    return img_clahe



def simulate_polarized_image(gray_image, polarization_angle):
    """
    Simulates a polarized image based on the input grayscale image and polarization angle.

    Args:
        gray_image: The input grayscale image as a NumPy array.
        polarization_angle: The polarization angle in degrees.

    Returns:
        A simulated polarized image as a NumPy array.
    """
    # Convert polarization angle to radians
    polarization_angle_rad = np.radians(polarization_angle)

    # Simulate polarized image using polarization angle
    polarized_img = np.cos(gray_image + polarization_angle_rad)

    return polarized_img

def dehaze_using_polarization(polarized_images):
    """
    Performs dehazing using polarized images and the Haze-Line Model.

    Args:
        polarized_images: A list of polarized images as NumPy arrays.

    Returns:
        A dehazed image as a NumPy array.
    """
    # Combine polarized images using simple averaging
    combined_image = np.mean(polarized_images, axis=0)

    # Estimate atmospheric light using the brightest pixel
    atmospheric_light = np.max(combined_image)

    # Estimate transmission map using Haze-Line Model
    transmission_map = 1 - (combined_image / atmospheric_light)

    # Remove haze using transmission map
    dehazed_image = np.zeros_like(combined_image)
    for i in range(3):  # Process each channel separately
        dehazed_image[:, :, i] = (combined_image[:, :, i] - atmospheric_light) / transmission_map[:, :, i] + atmospheric_light

    # Clip values to ensure they are within valid range
    #dehazed_image = np.clip(dehazed_image, 0, 255).astype(np.uint8)

    return dehazed_image




def dehazing_preproc(image):
    """
    This function performs pre-processing on an image, including polarization-based dehazing.

    Args:
        image: The input image as a NumPy array.

    Returns:
       A NumPy array representing the dehazed image in its original data type.
    """

    # Apply any necessary preprocessing steps (e.g., CLAHE, grayscale conversion)
    #image = apply_clahe(image)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Simulate polarized images (assuming a single input image):
    num_polarizations = 3  # Adjust as needed
    polarized_images = []
    for i in range(num_polarizations):
        polarized_img = simulate_polarized_image(image, polarization_angle=(i + 1) * 60)
        polarized_images.append(polarized_img)

    # Apply polarization-based dehazing:
    dehazed_image = dehaze_using_polarization(polarized_images)

    # Convert back to color if needed:
    #if len(image.shape) == 3:
        #dehazed_image = cv2.cvtColor(dehazed_image, cv2.COLOR_GRAY2BGR)

    return dehazed_image.astype(image.dtype)


# Example usage:
# preprocessed_image = dehazing_preproc(input_image)
import cv2
import numpy as np
from sklearn.cluster import KMeans

def non_local_image_dehazing(image):
    # Convert the image to the RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Cluster the pixels of the haze-free image into distinct colors
    kmeans = KMeans(n_clusters=256, random_state=0).fit(image_rgb.reshape(-1, 3))
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Identify the haze-lines in RGB space
    haze_lines = []
    for i in range(256):
        pixels = image_rgb[labels == i]
        if pixels.shape[0] > 10:
            haze_line = np.polyfit(pixels[:, 2], pixels[:, :2], 1)
            haze_lines.append(haze_line)

    # Estimate the per-pixel transmission
    transmission = np.zeros(image_rgb.shape[:2])
    for i in range(256):
        pixels = image_rgb[labels == i]
        if pixels.shape[0] > 10:
            for j in range(pixels.shape[0]):
                x = pixels[j, 2]
                y = np.polyval(haze_lines[i], x)
                transmission[pixels[j, :2]] = np.linalg.norm(pixels[j, :2] - y)

    # Recover the distance map and the haze-free image
    distance_map = transmission / np.max(transmission)
    haze_free_image = (image_rgb.astype(np.float32) - transmission.reshape(image_rgb.shape[:2] + (1,)) * A) / (1 - transmission)

    return haze_free_image, distance_map



def white_balance(image):
  """
  This function performs white balance adjustment on an image using OpenCV.

  Args:
      image: The input image as a NumPy array.

  Returns:
      A NumPy array representing the white balanced image.
  """
  # Convert the image to grayscale
  img_dtype = image.dtype
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
  balanced_image = np.clip(balanced_image, 0, 255).astype(img_dtype)

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
		source_img = non_local_image_dehazing(source_img)
		
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
		img = non_local_image_dehazing(img)

		return {'img': hwc_to_chw(img), 'filename': img_name}