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
        mask1: np.ndarray,
        mask2: np.ndarray,
        mask3: np.ndarray,
        mask4: np.ndarray,
        ):
    """ Visualize the axial slice (firs dim.) of a single region with alpha fusion. """
    img_slice = img[:, :, :]
    mask_slice_1 = mask1[:, :, :] 
    mask_slice_2 = mask2[:, :, :] 
    mask_slice_3 = mask3[:, :, :] 
    mask_slice_4 = mask4[:, :, :] 
    fused_slices = []

    for i in range(img.shape[0]):
        cmap = matplotlib.colormaps['bone']
        norm = matplotlib.colors.Normalize(vmin=np.amin(img_slice[i]), vmax=np.amax(img_slice[i]))
        fused_slice = 0.8 * cmap(norm(img_slice[i]))[...,:3] \
        + 0.2 * np.stack([np.zeros_like(mask_slice_1[i]), mask_slice_1[i], np.zeros_like(mask_slice_1[i])], axis=-1) \
        + 0.2 * np.stack([mask_slice_2[i], np.zeros_like(mask_slice_2[i]), np.zeros_like(mask_slice_2[i])], axis=-1) \
        + 0.2 * np.stack([np.zeros_like(mask_slice_3[i]), np.zeros_like(mask_slice_3[i]), mask_slice_3[i]], axis=-1) \
        + 0.2 * np.stack([mask_slice_4[i], mask_slice_4[i], np.zeros_like(mask_slice_4[i])], axis=-1)
        fused_slices.append(fused_slice)
    fused_slices = np.array(fused_slices)
    return fused_slices

def find_sequence_slices(seg_dataset):
    total_seg_len = seg_dataset['NumberOfFrames'].value
    seg_count = len(seg_dataset['SegmentSequence'].value)
    len_per_seg = total_seg_len//seg_count
    segmentation_array = seg_dataset.pixel_array
    segmentation_array = np.flip(segmentation_array, axis=1)
    dict_slices = dict()
    for i in range(seg_count):
        seg_name = segmentation_dataset['SegmentSequence'][i]['SegmentDescription'].value
        dict_slices[seg_name] = segmentation_array[i*len_per_seg:i*len_per_seg+len_per_seg]
    return dict_slices

if __name__ == "__main__":
    pixel_data = []
    segmentation_data = []
    
    segmentation_path = "./data/99-segmentation.dcm"
    segmentation_dataset = pydicom.dcmread(segmentation_path)
    
    # Find sequence
    sequence_slice = find_sequence_slices(segmentation_dataset)

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

    segmented_img_dcm = visualize_axial_slice(img_dcm, sequence_slice["Liver"], sequence_slice["Tumor"], sequence_slice["vessels"], sequence_slice["aorta"])
    
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