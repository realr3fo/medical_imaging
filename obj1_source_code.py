import os

import matplotlib
import pydicom
import numpy as np
import scipy
from matplotlib import pyplot as plt, animation


def median_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the median sagittal plane of the CT image provided."""
    return img_dcm[:, :, img_dcm.shape[1] // 2]  # Why //2?


def median_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the median sagittal plane of the CT image provided."""
    return img_dcm[:, img_dcm.shape[2] // 2, :]


def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the maximum intensity projection on the sagittal orientation."""
    return np.max(img_dcm, axis=2)


def AIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the average intensity projection on the sagittal orientation."""
    return np.mean(img_dcm, axis=2)


def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the maximum intensity projection on the coronal orientation."""
    return np.max(img_dcm, axis=1)


def AIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the average intensity projection on the coronal orientation."""
    return np.mean(img_dcm, axis=1)


def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """Rotate the image on the axial plane."""
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)

def find_centroid(mask: np.ndarray) -> np.ndarray:
    # Your code here:
    #   Consider using `np.where` to find the indices of the voxels in the mask
    #   ...
    idcs = np.where(mask == 1)
    centroid = np.stack([
        np.mean(idcs[0]),
        np.mean(idcs[1]),
        np.mean(idcs[2]),
    ])
    return centroid

def normalize_array(arr):
    max_value = np.max(arr)
    normalized_arr = arr / max_value
    normalized_arr = np.round(normalized_arr, 2) * 100
    return normalized_arr.astype(int)


def visualize_axial_slice(
        img: np.ndarray,
        mask: np.ndarray,
        mask_centroid: np.ndarray,
        ):
    """ Visualize the axial slice (firs dim.) of a single region with alpha fusion. """
    img_slice = img[:, :, :]
    mask_slice = mask[:, :, :] 
    fused_slices = []

    for i in range(img.shape[0]):
        cmap = matplotlib.colormaps['bone']
        norm = matplotlib.colors.Normalize(vmin=np.amin(img_slice[i]), vmax=np.amax(img_slice[i]))
        fused_slice = \
            0.8*cmap(norm(img_slice[i]))[..., :3] + \
            0.2*np.stack([mask_slice[i], np.zeros_like(mask_slice[i]), np.zeros_like(mask_slice[i])], axis=-1)
        fused_slices.append(fused_slice[...,0])
    fused_slices = np.array(fused_slices)
    return fused_slices

if __name__ == "__main__":
    pixel_data = []
    segmentation_data = []
    
    segmentation_path = "./data/99-segmentation.dcm"
    segmentation_dataset = pydicom.dcmread(segmentation_path)
    segmentation_array = segmentation_dataset.pixel_array

    directory = "./data/99-3.000000-C-A-P-42120-aq2/"
    directories = sorted(os.listdir(directory))
    for filename in directories:
        if filename.endswith(".dcm"):
            path = os.path.join(directory, filename)
            dataset = pydicom.dcmread(path)
            pixel_data.append(dataset.pixel_array)
    
    pixel_len_mm = [5, 0.78, 0.78]
    img_dcm = np.array(pixel_data)
    img_dcm = normalize_array(img_dcm)

    segmentation_array = np.flip(segmentation_array, axis=1)
    mask_centroid = find_centroid(segmentation_array[37:73]) # Tumor sequence
    segmented_img_dcm = visualize_axial_slice(img_dcm, segmentation_array[37:73], mask_centroid)
    
    # Show median planes
    fig, ax = plt.subplots(1, 2)
    # Sagittal planes
    ax[0].imshow(median_sagittal_plane(segmented_img_dcm), cmap="bone", alpha=1, aspect=pixel_len_mm[0] / pixel_len_mm[1])
    ax[0].set_title("Sagittal")
    # Coronal planes
    ax[1].imshow(median_coronal_plane(segmented_img_dcm), cmap="bone", alpha=1, aspect=pixel_len_mm[0] / pixel_len_mm[2])
    ax[1].set_title("Coronal")
    fig.suptitle("Median planes")
    plt.show()
    
    # Show sagittal planes
    # Based on  MIP/AIP/Median planes
    fig, ax = plt.subplots(1, 3)
    # Median planes
    ax[0].imshow(median_sagittal_plane(segmented_img_dcm), cmap="bone", aspect=pixel_len_mm[0] / pixel_len_mm[1])
    ax[0].set_title("Median")
    # MIP planes
    ax[1].imshow(MIP_sagittal_plane(segmented_img_dcm), cmap="bone", aspect=pixel_len_mm[0] / pixel_len_mm[1])
    ax[1].set_title("MIP")
    # AIP planes
    ax[2].imshow(AIP_sagittal_plane(segmented_img_dcm), cmap="bone", aspect=pixel_len_mm[0] / pixel_len_mm[1])
    ax[2].set_title("AIP")
    fig.suptitle("Sagittal planes")
    plt.show()

    # Show sagittal planes
    fig, ax = plt.subplots(1, 3)

    # Median planes
    ax[0].imshow(median_coronal_plane(segmented_img_dcm), cmap="bone", aspect=pixel_len_mm[0] / pixel_len_mm[1])
    ax[0].set_title("Median")

    # MIP planes
    ax[1].imshow(MIP_coronal_plane(segmented_img_dcm), cmap="bone", aspect=pixel_len_mm[0] / pixel_len_mm[1])
    ax[1].set_title("MIP")

    # AIP planes
    ax[2].imshow(AIP_coronal_plane(segmented_img_dcm), cmap="bone", aspect=pixel_len_mm[0] / pixel_len_mm[1])
    ax[2].set_title("AIP")

    fig.suptitle("Coronal")
    plt.show()


    # Create projections varying the angle of rotation on sagittal plane
    img_min_seg = np.amin(segmented_img_dcm)
    img_max_seg = np.amax(segmented_img_dcm)
    cm_seg = matplotlib.colormaps["bone"]

    fig, ax = plt.subplots()
    os.makedirs("results/MIP/Sagittal/", exist_ok=True)

    n = 48
    projections = []
    projections_seg = []

    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):

        rotated_img_seg = rotate_on_axial_plane(segmented_img_dcm, alpha)
        projection_seg = MIP_sagittal_plane(rotated_img_seg)

        ax.imshow(
            projection_seg,
            cmap=cm_seg,
            vmin=img_min_seg,
            vmax=img_max_seg,
            aspect=pixel_len_mm[0] / pixel_len_mm[1],
        )

        plt.savefig(f"results/MIP/Sagittal/Projection_{idx}.png")
        projections_seg.append(projection_seg)

    animation_data = [
        [
            ax.imshow(
                img_seg,
                animated=True,
                cmap=cm_seg,
                vmin=img_min_seg,
                vmax=img_max_seg,
                aspect=pixel_len_mm[0] / pixel_len_mm[1],
            )
        ]
        for img_seg in projections_seg
    ]
    anim = animation.ArtistAnimation(fig, animation_data, interval=30, blit=True)
    anim.save("results/MIP/Sagittal/Animation.gif")
    plt.show()