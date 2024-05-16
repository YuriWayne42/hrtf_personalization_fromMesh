# Predicting Global Head-Related Transfer Functions From Scanned Head Geometry Using Deep Learning and Compact Representations

Authors: Yuxiang Wang, You Zhang, Zhiyao Duan, Mark Bocko

## Required toolbox
[SOFA toolbox](https://github.com/sofacoustics/SOFAtoolbox)\
[SHtools](https://www.mathworks.com/matlabcentral/fileexchange/15279-shtools-spherical-harmonics-toolbox)\
[PyColormap4Matlab](https://www.mathworks.com/matlabcentral/fileexchange/68239-pycolormap4matlab)\
[Spherical Conformal Map](https://www.mathworks.com/matlabcentral/fileexchange/65551-spherical-conformal-map)\
[spherical-cap-harmonics](https://github.com/eesd-epfl/spherical-cap-harmonics)

## Datasets used
[HUTUBS HRTF database](https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960)\
Included both the HRIRs and scanned head mesh of the subjects.

## HRTF preprocessing
Process the HRTF into SHT representations in both frequency and temporal domains.

## Head mesh processing
Isolate the ear region of each scanned head mesh, and process into SCH representations.

## Deep learning model
Build the neural network to link the connections between the ear/torso geometry information to the HRTFs.
