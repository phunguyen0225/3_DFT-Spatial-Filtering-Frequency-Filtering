# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv

import numpy as np
import cv2


class Filtering:

    def __init__(self, image):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        """
        self.image = image
        self.mask = self.get_mask

    def get_mask(self, shape):
        """Computes a user-defined mask
        takes as input:
        shape: the shape of the mask to be generated
        rtype: a 2d numpy array with size of shape
        """
        mask = np.ones((shape[0], shape[1]))
        cv2.circle(mask, (286, 232), 7, (0, 0, 0), - 1, cv2.LINE_AA)
        cv2.circle(mask, (230, 280), 7, (0, 0, 0), - 1, cv2.LINE_AA)
        return mask

        return mask

    def post_process_image(self, image):
        """Post processing to display DFTs and IDFTs
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        You can perform post processing as needed. For example,
        1. You can perfrom log compression
        2. You can perfrom a full contrast stretch (fsimage)
        3. You can take negative (255 - fsimage)
        4. etc.
        """
        ans = np.round(image).astype(np.uint8)
        ans = (ans - np.min(ans)) * (255/(np.max(ans) - np.min(ans))) + 0.5

        ans = np.round(ans)
        return ans

    def filter(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do post processing on the magnitude and depending on the algorithm (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8
        """
        fft_shift = np.fft.fftshift(np.fft.fft2(self.image))
        dft_shift = np.log(abs(fft_shift)).astype(np.uint8)
        dft_shift = self.post_process_image(dft_shift)

        fft_filter = self.get_mask(np.shape(fft_shift)) * fft_shift
        dft_filter = np.log(abs(fft_filter)).astype(np.uint8)
        dft_filter = self.post_process_image(dft_filter)

        temp = np.fft.ifftshift(fft_filter)

        final = self.post_process_image(np.fft.ifft2(temp))

        return [final, dft_shift, dft_filter]
