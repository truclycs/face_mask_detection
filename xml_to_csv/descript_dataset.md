# Real-World Masked Face Dataset，RMFD

    Link: https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset

    License: Public by  National Engineering Research Center for Multimedia Software (NERCMS), School of Computer Science, Wuhan University.

    Descripton:

    - RFMD_part_1 can be downloaded directly. RFMD_part_2 (4 compressed files) and RFMD_part_3 (3 compressed files) need to download all compressed files before decompressing them.

    - Different from the facial mask recognition (or detection) dataset, the masked face recognition dataset must include multiple masked and unmasked face images of the same subject.

        1. Real mask face recognition data set: After sorting, cleaning and labeling a sample from the Internet, it contains 5,000 mask faces and 90,000 normal faces of 525 people.

        2. Simulated mask face recognition data set: Put masks on the faces in the public data set to obtain a simulated mask face data set of 10,000 and 500,000 faces.


#  WIDER Face and MAFA

    Link: https://drive.google.com/file/d/1QspxOJMDf_rAWVV7AU_Nc0rjo1_EPEDW/view

    License: Free.
    
    Description:
      1. This dataset is composed by [MAFA dataset](http://www.escience.cn/people/geshiming/mafa.html)[WIDER FACE dataset](http://shuoyang1213.me/WIDERFACE/)
      2. Trainset has 6120 images, 3006 of them comes from MAFA dataset with face mask, 3114 from WIDER Face which do not have face mask.
      3. Testset has 1839 images, 1059 of them come from MAFA dataset, 780 of them come from WIDER Face.
      4. Some wrong annotations.


# Mask Detection dataset

    Link: https://github.com/VirajDeshwal/mask-detection-dataset

    License: https://github.com/VirajDeshwal/mask-detection-dataset/blob/master/LICENSE
    
    Description:
       - The dataset includes around 2500 images with the shape of iPhone (480*848).
       - Not yet labeled.


# Mask-Detection-Dataset

    Link: https://github.com/archie9211/Mask-Detection-Dataset

    License: MIT 
    
    Description:

      - It has 4000+ images of masked people on and 4000 images of non-masks.
        
	| class number | class name |
	| ------------ | ---------- |
	| 0            | mask       |
	| 1            | nomask     |



	format of annotation file
	\<object-class> \<x> \<y> \<width> \<height>

	    0 0.513672 0.610879 0.246094 0.384937
	    0 0.567647 0.253906 0.205882 0.156250
	    1 0.373529 0.246094 0.158824 0.140625

# Face Mask Dataset Street View

    Link: https://drive.google.com/drive/u/0/folders/1T6N4qd4ep2YgxXdpCrkngOe_o1NszHH_

    License: Free
    
    Description: 
        - 600 images with labeled.
        - A lot of images have background on the street, have many people in one image with mask or no-mask.


# Face Mask Detection Dataset Kaggle

    Link: https://www.kaggle.com/andrewmvd/face-mask-detection

    License: CC0: Public Domain

    Description:
        - 853 images and their annotation files with mask and nonmask.
        - A lot of images have background on the street, have many people in one image with mask or no-mask.
