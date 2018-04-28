# SIFCM
Image Segmentation using Spatial Intuitionistic Fuzzy C Means Clustering

Folked from https://github.com/ab93/SIFCM

## Changes made:

    * Vectorize computations (> 150x speed up).
    * Indentation changed to 4 spaces.
    * Create class instance using image array instead of image file name.
    * Works on images of non-square shape.
    * Expose parameters these parameters to class instance creation:
        * m: fuzziness parameter.
        * kernel_size, kernel_shape: define neighborhood relations.
        * lam: intuitionistic parameter?
    * Add comments
    * Correct a bug in the original code:
        at line 194 of original code:

            ```
	    for j in range(self.n_clusters):
            ```

        `j` should be `i`, according to their paper.


## Original doc:


A fuzzy algorithm is presented for image segmentation of 2D gray scale images whose quality have been degraded by various kinds of noise. Traditional Fuzzy C Means (FCM) algorithm is very sensitive to noise and does not give good results. To overcome this problem, a new fuzzy c means algorithm was introduced that incorporated spatial information. The spatial function is the sum of all the membership functions within the neighborhood of the pixel
under consideration. The results showed that this approach was not as sensitive to noise as compared to the traditi-onal FCM algorithm and yielded better results. The algorithm we have proposed adds an intuitionistic approach in the membership function of the existing spatial FCM (sFCM). Intuitionistic refers to the degree of hesitation that arises as a consequence of lack of information and knowledge. Proposed method is comparatively less hampered by noise and performs better than existing algorithms.

Note:
Please refer to the link: http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=7238446&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D7238446

This project is a part of the above paper. Please cite it if you wish to distribute or work further.

DOI : http://dx.doi.org/10.5281/zenodo.31303

