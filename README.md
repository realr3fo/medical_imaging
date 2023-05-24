# DICOM Image Coregistration

This repository contains a project that focuses on loading, visualizing, and performing coregistration of DICOM images. The project aims to achieve the following objectives:

## Objective 1: DICOM Loading and Visualization
- Download the HCC-TACE-Seg dataset and work with the assigned patient, HCC _01.
- Visualize the DICOM images using a third-party DICOM visualizer such as 3D-Slicer.
- Utilize PyDicom to load the segmentation image and corresponding CT image, arranging the pixel array based on relevant headers.
- Create an animation (e.g., GIF file) showcasing a rotating Maximum Intensity Projection on the coronal-sagittal planes.

## Objective 2: 3D Rigid Coregistration
- Perform coregistration of the given images using either defined landmarks or a custom function similarity measure.
- Implement the image coregistration process without relying on libraries like PyElastix.
- Utilize the "icbm avg 152 t1 tal nlin symmetric VI" image as the reference and the "RM Brain 3D-SPGR" image of an anonymized patient as the input.
- Visualize the Thalamus region in the input image space.

Feel free to explore the repository and its contents to gain insights into DICOM image handling, visualization, and 3D rigid coregistration.