import os

from skimage import exposure
import matplotlib
import pydicom
import numpy as np
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
import math
import scipy


def get_thalamus_mask(img_atlas: np.ndarray) -> np.ndarray:
    result = np.zeros_like(img_atlas)
    result[(121 <= img_atlas) & (img_atlas <= 150)] = 1
    return result

def find_centroid(mask: np.ndarray) -> np.ndarray:
    idcs = np.where(mask == 1)
    centroid = np.stack([
        np.mean(idcs[0]),
        np.mean(idcs[1]),
        np.mean(idcs[2]),
    ])
    return centroid


def visualize_axial_slice(
        img: np.ndarray,
        mask: np.ndarray,
        mask_centroid: np.ndarray,
        ):
    """ Visualize the axial slice (firs dim.) of a single region with alpha fusion. """
    img_slice = img[mask_centroid[0].astype('int'), :, :]
    mask_slice = mask[mask_centroid[0].astype('int'), :, :]

    cmap = matplotlib.colormaps["bone"]
    norm = matplotlib.colors.Normalize(vmin=np.amin(img_slice), vmax=np.amax(img_slice))
    fused_slice = \
        0.5*cmap(norm(img_slice))[..., :3] + \
        0.5*np.stack([mask_slice, np.zeros_like(mask_slice), np.zeros_like(mask_slice)], axis=-1)

    return fused_slice


def multiply_quaternions(
        q1: tuple[float, float, float, float],
        q2: tuple[float, float, float, float]
        ) -> tuple[float, float, float, float]:
    """ Multiply two quaternions, expressed as (1, i, j, k). """
    # Your code here:
    #   ...
    return (
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    )


def conjugate_quaternion(
        q: tuple[float, float, float, float]
        ) -> tuple[float, float, float, float]:
    """ Multiply two quaternions, expressed as (1, i, j, k). """
    # Your code here:
    #   ...
    return (
        q[0], -q[1], -q[2], -q[3]
    )

def translation(
        point: tuple[float, float, float],
        translation_vector: tuple[float, float, float]
        ) -> tuple[float, float, float]:
    """ Perform translation of `point` by `translation_vector`. """
    x, y, z = point
    v1, v2, v3 = translation_vector
    return (x+v1, y+v2, z+v3)

def axial_rotation(
        point: tuple[float, float, float],
        angle_in_rads: float,
        axis_of_rotation: tuple[float, float, float]) -> tuple[float, float, float]:
    """ Perform axial rotation of `point` around `axis_of_rotation` by `angle_in_rads`. """
    x, y, z = point
    v1, v2, v3 = axis_of_rotation
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm
    p = (0, x, y, z)
    #   Quaternion associated to axial rotation.
    cos, sin = math.cos(angle_in_rads / 2), math.sin(angle_in_rads / 2)
    q = (cos, sin * v1, sin * v2, sin * v3)
    #   Quaternion associated to image point
    q_star = conjugate_quaternion(q)
    p_prime = multiply_quaternions(q, multiply_quaternions(p, q_star))
    #   Interpret as 3D point (i.e. drop first coordinate)
    return p_prime[1], p_prime[2], p_prime[3]

def translation_then_axialrotation(point: tuple[float, float, float], parameters: tuple[float, ...]):
    """ Apply to `point` a translation followed by an axial rotation, both defined by `parameters`. """
    x, y, z = point
    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters
    t_x, t_y, t_z = translation([x, y, z], [t1, t2, t3])
    r_x, r_y, r_z = axial_rotation([t_x, t_y, t_z], angle_in_rads,[v1, v2, v3])
    return [r_x, r_y, r_z]

def vector_of_residuals(ref_points: np.ndarray, inp_points: np.ndarray) -> np.ndarray:
    """ Given arrays of 3D points with shape (point_idx, 3), compute vector of residuals as their respective distance """
    distances = np.linalg.norm(inp_points - ref_points, axis=1)
    return distances

def coregister_landmarks(ref_landmarks: np.ndarray, inp_landmarks: np.ndarray):
    """ Coregister two sets of landmarks using a rigid transformation. """
    initial_parameters = [
        0, 0, 0,    # Translation vector
        0,          # Angle in rads
        1, 0, 0,    # Axis of rotation
    ]
    # Find better initial parameters
    centroid_ref = np.mean(ref_landmarks, axis=0)
    centroid_inp = np.mean(inp_landmarks, axis=0)
    initial_parameters[0] = centroid_ref[0] - centroid_inp[0]
    initial_parameters[1] = centroid_ref[1] - centroid_inp[1]
    initial_parameters[2] = centroid_ref[2] - centroid_inp[2]

    def function_to_minimize(parameters):
        """ Transform input landmarks, then compare with reference landmarks."""
        new_inp_landmarks = []
        for point in inp_landmarks:
            new_point = translation_then_axialrotation(point, parameters)
            new_inp_landmarks.append(new_point)
        new_inp_landmarks = np.array(new_inp_landmarks)
        current_value = vector_of_residuals(ref_landmarks, new_inp_landmarks)
        return current_value
    

    # Apply least squares optimization
    result = least_squares(
        function_to_minimize,
        x0=initial_parameters,
        verbose=0)
    return result

def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """Rotate the image on the axial plane."""
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)

def rotate_on_axial_plane_rgb(img: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """Rotate the image on the axial plane."""
    rotated_img = np.zeros_like(img)
    rotated_img[:,:] = scipy.ndimage.rotate(img[:,:], angle_in_degrees, reshape=False)
    return rotated_img


def median_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the median sagittal plane of the CT image provided."""
    return img_dcm[:, img_dcm.shape[2] // 2, :]

def mean_absolute_error(img_input: np.ndarray, img_reference) -> np.ndarray:
    """ Compute the MAE between two images. """
    return np.mean(np.abs(img_input - img_reference))

def mean_squared_error(img_input: np.ndarray, img_reference) -> np.ndarray:
    """ Compute the MSE between two images. """
    return np.mean((img_input - img_reference)**2)

def mutual_information(img_input: np.ndarray, img_reference) -> np.ndarray:
    """ Compute the Shannon Mutual Information between two images. """
    nbins = [10, 10]
    # Compute entropy of each image
    hist = np.histogram(img_input.ravel(), bins=nbins[0])[0]
    prob_distr = hist / np.sum(hist)
    entropy_input = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))  # Why +1e-7?
    hist = np.histogram(img_reference.ravel(), bins=nbins[0])[0]
    prob_distr = hist / np.sum(hist)
    entropy_reference = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))  # Why +1e-7?
    # Compute joint entropy
    joint_hist = np.histogram2d(img_input.ravel(), img_reference.ravel(), bins=nbins)[0]
    prob_distr = joint_hist / np.sum(joint_hist)
    joint_entropy = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))
    # Compute mutual information
    return entropy_input + entropy_reference - joint_entropy

def median_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the median sagittal plane of the CT image provided."""
    return img_dcm[:, :, img_dcm.shape[1] // 2]

def preprocess_landmarks(landmarks):
    max_value = np.max(landmarks)
    normalized_landmarks = landmarks / max_value
    preprocessed_landmarks = np.round(normalized_landmarks, 2) * 100
    return preprocessed_landmarks.astype(int)

def normalize_intensity(input_image, reference_image):
    # Flatten the images to 1D arrays
    input_flat = input_image.flatten()
    reference_flat = reference_image.flatten()
    # Perform histogram matching
    matched_flat = exposure.match_histograms(input_flat, reference_flat)
    # Reshape the matched image back to its original shape
    matched_image = np.reshape(matched_flat, input_image.shape)
    return matched_image

if __name__ == "__main__":
    # Loading the DICOM files
    pixel_data = []
    reference_data = []
    
    reference_path = "./data/image_reference/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm"
    reference_dataset = pydicom.dcmread(reference_path)
    reference_array = reference_dataset.pixel_array

    aal_path = "./data/image_aal/AAL3_1mm.dcm"
    aal_dataset = pydicom.dcmread(aal_path)
    aal_array = aal_dataset.pixel_array

    directory = "./data/RM_Brain_3D-SPGR"
    directories = sorted(os.listdir(directory))
    instances = {}

    for filename in directories:
        if filename.endswith(".dcm"):
            path = os.path.join(directory, filename)
            dataset = pydicom.dcmread(path)
            instance_number = dataset.InstanceNumber
            instances[instance_number] = path
    sorted_instances = sorted(instances.items(), key=lambda x: x[0])
    sorted_paths = [path for _, path in sorted_instances]
    pixel_data = []
    for path in sorted_paths:
        dataset = pydicom.dcmread(path)
        pixel_data.append(np.flip(dataset.pixel_array, axis=0))
    
    # Visualization variables

    pixel_len_mm = [2, 0.51, 0.51]

    volume = np.stack(pixel_data, axis=0)
    volume_ref = np.stack(reference_array, axis=0)
    volume_aal = np.stack(aal_array, axis=0)
    plane_index = volume.shape[0] // 2 
    plane_index_ref = volume_ref.shape[0] // 2
    plane_index_aal = volume_aal.shape[0] // 2
    
    horizontal_plane = volume[plane_index + 15, :, :]
    horizontal_plane_ref = volume_ref[plane_index_ref, :, :]
    horizontal_plane_aal = volume_aal[plane_index_aal, :, :]

    # Initial visualizations
    fig, ax = plt.subplots(1, 3)
    images = [horizontal_plane, horizontal_plane_ref, horizontal_plane_aal]
    titles = ["Input image", "Reference image", "AAL image"]

    for i in range(3):
        ax[i].imshow(images[i], cmap=matplotlib.colormaps["bone"])
        ax[i].set_title(titles[i])

    fig.suptitle("Input, Reference, and AAL images in the horizontal plane")
    plt.show()
    
    # Preprocess the input data, reference data to match the AAL file for the mask

    # Reshape the phantom and AAL data
    img_phantom = reference_dataset.pixel_array[6:-6, 6:-7, 6:-6]     # Crop phantom to atlas size
    aal_array = aal_array[:, :-1, :]     # Crop atlas size so that the sum of the shapes is divisible by 3

    thalamus_mask = get_thalamus_mask(aal_array)
    mask_centroid = find_centroid(thalamus_mask)
    mask_centroid_idx = mask_centroid[0].astype('int')

    # Reshape the input data (crop, zoom, and initial rotation)
    pixel_data = np.array(pixel_data)
    z_start = (pixel_data.shape[0] - 181) + 3 # adjust with the mask
    z_end = z_start + 181
    cropped_data = pixel_data[z_start:z_end, 48:456, 83:438]
    resize_factors = (181 / cropped_data.shape[0], 216 / cropped_data.shape[1], 181 / cropped_data.shape[2])
    zoom_data = zoom(cropped_data, resize_factors, order=1)
    rotate_val = 3
    rotated_data = rotate_on_axial_plane(zoom_data, rotate_val)
    processed_input = normalize_intensity(rotated_data, img_phantom)
    processed_input = preprocess_landmarks(processed_input)


    # Preprocessing visualization
    fig, ax = plt.subplots(1, 3)
    images = [processed_input[mask_centroid_idx], img_phantom[mask_centroid_idx],aal_array[mask_centroid_idx]]
    titles = ["Input image", "Reference image", "AAL image"]

    for i in range(3):
        ax[i].imshow(images[i], cmap=matplotlib.colormaps["bone"])
        ax[i].set_title(titles[i])

    fig.suptitle("Input, Reference, and AAL images\n in the horizontal plane after preprocessing with thalamus region showing")
    plt.show()

    # Before coregistration
    fig, ax = plt.subplots(1, 2)
    images = [processed_input[mask_centroid_idx], img_phantom[mask_centroid_idx]]
    titles = ["Input image", "Reference image"]
    for i in range(2):
        ax[i].imshow(images[i], cmap=matplotlib.colormaps["bone"])
        ax[i].set_title(titles[i])
    fig.suptitle("Input image and Reference image in horizontal plane\n before coregistration")
    plt.show()

    # Sagittal and coronal comparison
    fig, ax = plt.subplots(2, 2, figsize=(6,7))
    images = [
        median_sagittal_plane(np.flip(processed_input)),
        median_coronal_plane(np.flip(processed_input)),
        median_sagittal_plane(np.flip(img_phantom)),
        median_coronal_plane(np.flip(img_phantom))
    ]
    titles = [
        "Sagittal input",
        "Coronal input",
        "Sagittal reference",
        "Coronal reference"
    ]
    for i, (image, title) in enumerate(zip(images, titles)):
        ax[i//2, i%2].imshow(image, cmap=matplotlib.colormaps["bone"])
        ax[i//2, i%2].set_title(title)

    fig.suptitle("Sagittal and Coronal median comparison before coregistration")
    plt.show()


    down_sampled_ref = img_phantom[::4, ::4, ::4].reshape(-1,3)
    down_sampled_inp_shape = processed_input[::4, ::4, ::4].shape
    down_sampled_inp = processed_input[::4, ::4, ::4].reshape(-1,3)
    
    ref_landmarks = img_phantom.reshape(-1, 3)
    inp_landmarks = processed_input.reshape(-1, 3)
    
    print('Residual vector of distances between each pair of landmark points (ref vs. input):')
    vec = vector_of_residuals(ref_landmarks, inp_landmarks)
    print("  >> Residual vector shape:", vec.shape)
    print(f'  >> Mean: {np.mean(vec.flatten())}.')
    print(f'  >> Max: {np.max(vec.flatten())}.')
    print(f'  >> Min: {np.min(vec.flatten())}.')
    print(f'  >> Result: {vec}.')

    limit = inp_landmarks.shape[0]
    # Vector of residuals: visualization
    ref_show = ref_landmarks[:limit]
    inp_show = inp_landmarks[:limit]
    fig = plt.figure()
    axs = np.asarray([fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')])
    axs[0].scatter(ref_show[..., 0], ref_show[..., 1], ref_show[..., 2], marker='o')
    axs[0].set_title('Reference landmarks')
    axs[1].scatter(inp_show[..., 0], inp_show[..., 1], inp_show[..., 2], marker='^')
    axs[1].set_title('Input landmarks')
    # Uniform axis scaling
    all_points = np.concatenate([ref_show, inp_show], axis=0)
    range_x = np.asarray([np.min(all_points[..., 0]), np.max(all_points[..., 0])])
    range_y = np.asarray([np.min(all_points[..., 1]), np.max(all_points[..., 1])])
    range_z = np.asarray([np.min(all_points[..., 2]), np.max(all_points[..., 2])])
    max_midrange = max(range_x[1]-range_x[0], range_y[1]-range_y[0], range_z[1]-range_z[0]) / 2
    for ax in axs.flatten():
        ax.set_xlim3d(range_x[0]/2 + range_x[1]/2 - max_midrange, range_x[0]/2 + range_x[1]/2 + max_midrange)
        ax.set_ylim3d(range_y[0]/2 + range_y[1]/2 - max_midrange, range_y[0]/2 + range_y[1]/2 + max_midrange)
        ax.set_zlim3d(range_z[0]/2 + range_z[1]/2 - max_midrange, range_z[0]/2 + range_z[1]/2 + max_midrange)
    fig.suptitle("Landmark points comparison before coregistration")
    plt.show()

    # Coregister landmarks
    print("Coregistering landmarks started")
    result = coregister_landmarks(down_sampled_ref[:limit], down_sampled_inp[:limit]) # 30 mins with downsampling
    solution_found = result.x
    
    # Best parameters from downsampling with factor of 4:
    #     >> Translation: (-1.78, -2.00, -2.38).
    #     >> Rotation: 0.15 rads around axis (0.62, 0.51, 0.62)

    t1, t2, t3, angle_in_rads, v1, v2, v3 = result.x
    best_params = result.x
    # best_params = [-1.78, -2.00, -2.38, 0.15, 0.62, 0.51, 0.62]
    t1, t2, t3, angle_in_rads, v1, v2, v3 = best_params
    print('Best parameters:')
    print(f'  >> Translation: ({t1:0.02f}, {t2:0.02f}, {t3:0.02f}).')
    print(f'  >> Rotation: {angle_in_rads:0.02f} rads around axis ({v1:0.02f}, {v2:0.02f}, {v3:0.02f}).')

    inp_landmarks[:] = np.asarray([translation_then_axialrotation(point, best_params) for point in inp_landmarks[:]])

    fig = plt.figure()
    axs = np.asarray([fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')])
    axs[0].scatter(ref_show[..., 0], ref_show[..., 1], ref_show[..., 2], marker='o')
    axs[0].set_title('Reference landmarks')
    axs[1].scatter(inp_show[..., 0], inp_show[..., 1], inp_show[..., 2], marker='^')
    axs[1].set_title('Input landmarks')
    # Uniform axis scaling
    all_points = np.concatenate([ref_show, inp_show], axis=0)
    range_x = np.asarray([np.min(all_points[..., 0]), np.max(all_points[..., 0])])
    range_y = np.asarray([np.min(all_points[..., 1]), np.max(all_points[..., 1])])
    range_z = np.asarray([np.min(all_points[..., 2]), np.max(all_points[..., 2])])
    max_midrange = max(range_x[1]-range_x[0], range_y[1]-range_y[0], range_z[1]-range_z[0]) / 2
    for ax in axs.flatten():
        ax.set_xlim3d(range_x[0]/2 + range_x[1]/2 - max_midrange, range_x[0]/2 + range_x[1]/2 + max_midrange)
        ax.set_ylim3d(range_y[0]/2 + range_y[1]/2 - max_midrange, range_y[0]/2 + range_y[1]/2 + max_midrange)
        ax.set_zlim3d(range_z[0]/2 + range_z[1]/2 - max_midrange, range_z[0]/2 + range_z[1]/2 + max_midrange)
    plt.show()
    fig.suptitle("Landmark points comparison after coregistration")

    inp_landmarks_orig = inp_landmarks.reshape(181, 216, 181)

    # Sagittal and coronal comparison
    fig, ax = plt.subplots(2, 2, figsize=(6,7))
    images = [
        median_sagittal_plane(np.flip(inp_landmarks_orig)),
        median_coronal_plane(np.flip(inp_landmarks_orig)),
        median_sagittal_plane(np.flip(img_phantom)),
        median_coronal_plane(np.flip(img_phantom))
    ]
    titles = [
        "Sagittal input",
        "Coronal input",
        "Sagittal reference",
        "Coronal reference"
    ]
    for i, (image, title) in enumerate(zip(images, titles)):
        ax[i//2, i%2].imshow(image, cmap=matplotlib.colormaps["bone"])
        ax[i//2, i%2].set_title(title)

    fig.suptitle("Sagittal and Coronal median comparison after coregistration")
    plt.show()


    fig, ax = plt.subplots(1, 2)
    images = [inp_landmarks_orig[mask_centroid_idx], img_phantom[mask_centroid_idx]]
    titles = ["Input image", "Reference image"]
    for i in range(2):
        ax[i].imshow(images[i], cmap=matplotlib.colormaps["bone"])
        ax[i].set_title(titles[i])
    fig.suptitle("Input image and Reference image in horizontal plane\n after coregistration")
    plt.show()

    volume = np.stack(inp_landmarks_orig, axis=0)
    volume_ref = np.stack(img_phantom, axis=0)
    volume_aal = np.stack(aal_array, axis=0)
    plane_index = volume.shape[0] // 2 
    plane_index_ref = volume_ref.shape[0] // 2
    plane_index_aal = volume_aal.shape[0] // 2
    
    # Visualization of difference between reference and original
    img_orig = inp_landmarks_orig[mask_centroid_idx, :, :]
    img_ref = img_phantom[mask_centroid_idx, :, :]
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img_orig, cmap='bone')
    axs[0].set_title('Image input')
    axs[1].imshow(img_orig - img_ref, cmap='bone')
    axs[1].set_title('Difference')
    axs[2].imshow(img_ref, cmap='bone')
    axs[2].set_title('Image reference')
    fig.suptitle("Image differences: Image input - Image reference")
    plt.show()

    # Compute metrics
    mae = mean_absolute_error(img_ref, img_orig)
    print('MAE:')
    print(f'  >> Result: {mae:.02f} HU')

    mse = mean_squared_error(img_ref, img_orig)
    print('MSE:')
    print(f'  >> Result: {mse:.02f} HU^2')

    mutual_inf = mutual_information(img_ref, img_orig)
    print('Mutual Information:')
    print(f'  >> Result: {mutual_inf:02f} bits')
    
    # Second part: Thalamus region mapping
    fused_phantom = visualize_axial_slice(img_phantom, thalamus_mask, mask_centroid)
    fused_orig = visualize_axial_slice(inp_landmarks_orig, thalamus_mask, mask_centroid)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(fused_orig, cmap='bone')
    axs[0].set_title("Input image")
    axs[1].imshow(fused_phantom, cmap='bone')
    axs[1].set_title("Reference image")
    fig.suptitle("Input and Reference image fused with thalamus region")
    plt.show()

    # Invert input image from reference space into input space
    # All values on the y axis -0.35, all values on x axis -0.34
    red_channel = fused_orig[..., 0]
    fused_flat = red_channel.flatten()
    original_flat = img_orig.flatten()
    # Match intensity with high intensity orig image
    denormalized_flat = exposure.match_histograms(fused_flat, original_flat)
    denormalized_image = np.reshape(denormalized_flat, red_channel.shape)
    # Translate back from coregistration best values for translation
    decoregistration_params = [1.78, 2.00, 2.38, -0.15, -0.62, -0.51, -0.62]
    pre_translate = denormalized_image.reshape(-1, 3)
    translated = np.array([translation_then_axialrotation(point, decoregistration_params) for point in pre_translate])
    translated = translated.reshape(red_channel.shape)
    # Normalize intensity with original image 
    denormalized_intensity = normalize_intensity(translated,  rotated_data[plane_index])
    # Rotate back from the input image preprocessing
    degrees = -3
    rotated = rotate_on_axial_plane_rgb(denormalized_intensity, degrees)
    # Calculate the resize factors in reverse
    resize_factors = (512 / rotated.shape[0]/1.2, 512 / rotated.shape[1]/1.4)
    # Resize the image back to its original size
    reverted_data = zoom(rotated, resize_factors, order=1)
    # Determine the desired padding sizes
    pad_height = (512 - reverted_data.shape[0]) // 2
    pad_width = (512 - reverted_data.shape[1]) // 2
    # Pad the image with zeros
    padded_image = np.pad(reverted_data, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')


    fig, axs = plt.subplots(1, 3, figsize=(10,4))
    axs[0].imshow(fused_orig, cmap='bone')
    axs[0].set_title("Input image fused with\n thalamus region in reference space")
    axs[1].imshow(padded_image, cmap='bone')
    axs[1].set_title("Input image fused with\n thalamus region in input space")
    axs[2].imshow(pixel_data[mask_centroid_idx + 33], cmap='bone')
    axs[2].set_title("Original Input image")
    fig.suptitle("Final results")
    plt.show()