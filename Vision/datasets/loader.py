import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img
import cv2
import numpy as np




import numpy as np
import cv2

def dark_channel_prior(image, window_size=15):
    """
    Compute the dark channel prior of an image.

    Args:
    - image: Input hazy image.
    - window_size: Size of the window for computing the dark channel.

    Returns:
    - dark_channel: Dark channel prior of the input image.
    """
    # Compute the dark channel
    dark_channel = np.min(image, axis=2)
    dark_channel = cv2.erode(dark_channel, np.ones((window_size, window_size), dtype=np.uint8))
    return dark_channel

def estimate_airlight_dcp(image, dark_channel, percentile=0.1):
    """
    Estimate airlight using the Dark Channel Prior method.

    Args:
    - image: Input hazy image.
    - dark_channel: Dark channel prior of the input image.
    - percentile: Percentage of brightest pixels to consider for airlight estimation.

    Returns:
    - airlight: Estimated airlight value.
    """
    # Flatten the dark channel to find the brightest pixels
    dark_channel_flat = dark_channel.flatten()
    
    # Sort the dark channel values in descending order
    sorted_indices = np.argsort(-dark_channel_flat)
    
    # Calculate the number of pixels to consider based on the given percentile
    num_pixels = int(percentile * len(sorted_indices))
    
    # Select the brightest pixels and compute the average color
    brightest_pixels = image.reshape(-1, 3)[sorted_indices[:num_pixels]]
    airlight = np.mean(brightest_pixels, axis=0)
    
    return airlight

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter

def load_image(image_path):
    # Load image and convert to RGB color space
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def convert_to_spherical(image):
    # Convert image to spherical coordinates
    r = np.linalg.norm(image, axis=2)
    phi = np.arctan2(image[:,:,1], image[:,:,0])
    theta = np.arccos(image[:,:,2] / (r + 1e-10))  # Add a small epsilon to avoid division by zero
    return r, phi, theta

def kmeans_clustering(spherical_coords, n_clusters=3):
    # Flatten the spherical coordinates
    pixels = np.column_stack((spherical_coords[1].flatten(), spherical_coords[2].flatten()))
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixels)
    # Get cluster labels
    cluster_labels = kmeans.labels_
    return cluster_labels.reshape(spherical_coords[0].shape)

def estimate_max_radius(cluster_labels, r):
    # Estimate maximum radius for each cluster
    max_radius = np.zeros(np.max(cluster_labels) + 1)
    for i in range(len(max_radius)):
        max_radius[i] = np.max(r[cluster_labels == i])
    return max_radius

def estimate_transmission(r, max_radius, cluster_labels):
    # Estimate transmission for each pixel
    transmission_estimated = np.zeros_like(r)
    for i in range(len(max_radius)):
        transmission_estimated[cluster_labels == i] = r[cluster_labels == i] / max_radius[i]
    return transmission_estimated

def regularize_transmission(transmission, image, sigma=0.1, lam=0.1):
    # Regularize transmission
    regularized_transmission = np.zeros_like(transmission)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            neighborhood = image[max(0, x-1):min(x+2, image.shape[0]), max(0, y-1):min(y+2, image.shape[1])]
            t_diff = transmission[x, y] - transmission[neighborhood[:,:,0], neighborhood[:,:,1]]
            i_diff = np.linalg.norm(image[x, y] - image[neighborhood[:,:,0], neighborhood[:,:,1]], axis=2)
            regularization_term = np.sum(t_diff**2 / (2 * sigma**2) + lam * t_diff * i_diff)
            regularized_transmission[x, y] = transmission[x, y] - regularization_term
    return regularized_transmission

def dehaze_image(image, transmission, airlight):
    # Dehaze the image
    dehazed_image = np.zeros_like(image)
    for i in range(3):
        dehazed_image[:,:,i] = ((image[:,:,i] - airlight[i]) / np.maximum(transmission, 0.1)) + airlight[i]
    return np.clip(dehazed_image, 0, 255).astype(np.uint8)


def post_proc(image):
    """
    Perform color correction on the input image using histogram equalization.

    Args:
    - image: Input image (hazy or dehazed).

    Returns:
    - corrected_image: Color-corrected image.
    """
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply histogram equalization to the L channel
    lab_image[:,:,0] = cv2.equalizeHist(lab_image[:,:,0])
    
    # Convert the LAB image back to RGB color space
    corrected_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    
    return corrected_image







def dehazing_preproc(image):
    """
    This function performs pre-processing on an image, including polarization-based dehazing.

    Args:
        image: The input image as a NumPy array.

    Returns:
       A NumPy array representing the dehazed image in its original data type.
    """
    hazy_image = image
    airlight = np.array([255, 255, 255])  # Sample airlight value, should be estimated
    IA = hazy_image - airlight

# Convert IA to spherical coordinates
    r, phi, theta = convert_to_spherical(IA)

# Cluster pixels according to [phi(x), theta(x)]
    cluster_labels = kmeans_clustering((r, phi, theta))

# Estimate maximum radius for each cluster
    max_radius = estimate_max_radius(cluster_labels, r)

# Estimate transmission
    transmission_estimated = estimate_transmission(r, max_radius, cluster_labels)

# Regularize transmission
    regularized_transmission = gaussian_filter(transmission_estimated, sigma=0.1)

# Dehaze the image
    dark_channel = dark_channel_prior(hazy_image)

# Estimate airlight using DCP
    airlight_dcp = estimate_airlight_dcp(hazy_image, dark_channel)
    dehazed_image = dehaze_image(hazy_image, regularized_transmission, airlight_dcp)
    dehazed_image = post_proc(dehazed_image)
    

    return dehazed_image


# Example usage:
# preprocessed_image = dehazing_preproc(input_image)



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


def apply_clahe(img):

    # img = img * 255.0
    # img = img.astype(np.uint8)
    image_8bit = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    img_LAB = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2Lab)

    l, a, b = cv2.split(img_LAB)
    
    # Create a CLAHE object (Clip Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE to each channel separately
    l_clahe = clahe.apply(l)
    
    # Merge the CLAHE enhanced channels
    img_clahe = cv2.merge((l_clahe, a, b))

    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_Lab2BGR)

    img_clahe = img_clahe.astype(np.float32) / 255.0
    
    return img_clahe


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
		source_img =apply_clahe(source_img)
		
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
		img = apply_clahe(img)

		return {'img': hwc_to_chw(img), 'filename': img_name}
	


    
	

