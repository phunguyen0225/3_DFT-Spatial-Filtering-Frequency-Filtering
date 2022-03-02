import numpy as np
import math


class Filtering:

    def __init__(self, image):
        self.image = image

    def get_gaussian_filter(self):
        """Initialzes and returns a 5X5 Gaussian filter
            Use the formula for a 2D gaussian to get the values for a 5X5 gaussian filter
        """
        size = 5
        gaussian_filter = np.zeros((size, size))
        mid = size / 2

        for i in range(-mid, mid + 1):
            for j in range(-mid, mid+1):
                gaussian_filter[i + mid][j +
                                         mid] = (1 / (2 * math.pi * (1**2))) * np.exp(-1 * (i**2 + j**2) / (2 * (1**2)))
        return gaussian_filter

    def get_laplacian_filter(self):
        """Initialzes and returns a 3X3 Laplacian filter"""

        size = 3
        laplacian_filter = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i in [0, 2] and j in [0, 2]:
                    laplacian_filter[i][j] = 0
                if (i == 1 and j in [0, 2]) or (j == 1 and i in [0, 2]):
                    laplacian_filter[i][j] = 1
                if i == 1 and j == 1:
                    laplacian_filter[i][j] = -4
        return laplacian_filter

    def filter(self, filter_name):
        """Perform filtering on the image using the specified filter, and returns a filtered image
            takes as input:
            filter_name: a string, specifying the type of filter to use ["gaussian", laplacian"]
            return type: a 2d numpy array
                """
        new_image = np.shape(self.image)
        # get row and col
        row = new_image[0]
        col = new_image[1]

        if filter_name == "gaussian":
            mask = self.get_gaussian_filter()
            padding = np.zeros((row + 8, col + 8))
            ans = np.zeros((row + 4, col + 4))
            padding_row = np.shape(padding)[0]
            padding_col = np.shape(padding)[1]

            temp = 0
            for x in range(np.shape(mask)[0]):
                for y in range(np.shape(mask)[1]):
                    temp += mask[x][y]

            for x in range(4, padding_row-4):
                for y in range(4, padding_col-4):
                    padding[x][y] = self.image[x-4][y-4]

            for x in range(2, np.shape(padding)[0]-4):
                for y in range(2, np.shape(padding)[1]-4):
                    sum = 0
                    for a in range(0, 5):
                        if (a == 0):
                            sum = (mask[a][0]*padding[x-2][y-2]) + (mask[a][1]*padding[x-2][y-1]) + (mask[a][2]*padding[x-2][y]) + \
                                (mask[a][3]*padding[x-2][y+1]) + \
                                (mask[a][4]*padding[x-2][y+2])
                        if (a == 1):
                            sum += (mask[a][0]*padding[x-1][y-2]) + (mask[a][1]*padding[x-1][y-1]) + (mask[a][2]*padding[x-1][y]) + \
                                (mask[a][3]*padding[x-1][y+1]) + \
                                (mask[a][4]*padding[x-1][y+2])
                        if (a == 2):
                            sum += (mask[a][0]*padding[x][y-2]) + (mask[a][1]*padding[x][y-1]) + (mask[a][2]*padding[x][y]) + \
                                (mask[a][3]*padding[x][y+1]) + \
                                (mask[a][4]*padding[x][y+2])
                        if (a == 3):
                            sum += (mask[a][0]*padding[x+1][y-2]) + (mask[a][1]*padding[x+1][y-1]) + (mask[a][2]*padding[x+1][y]) + \
                                (mask[a][3]*padding[x+1][y+1]) + \
                                (mask[a][4]*padding[x+1][y+2])
                        if (a == 4):
                            sum += (mask[a][0]*padding[x+2][y-2]) + (mask[a][1]*padding[x+2][y-1]) + (mask[a][2]*padding[x+2][y]) + \
                                (mask[a][3]*padding[x+2][y+1]) + \
                                (mask[a][4]*padding[x+2][y+2])

                    ans[x-2][y-2] = sum/(1/temp)

        if filter_name == "laplacian":
            mask = self.get_laplacian_filter()
            padding = np.zeros((row + 2, col + 2))
            ans = np.zeros((row, col))
            padding_row = np.shape(padding)[0]
            padding_col = np.shape(padding)[1]

            for x in range(1, padding_row-1):
                for y in range(1, padding_col-1):
                    padding[x][y] = self.image[x-1][y-1]

            for x in range(1, padding_row-1):
                for y in range(1, padding_col-1):
                    sum = 0
                    for a in range(0, 3):
                        if (a == 0):
                            sum += (mask[a][0] * padding[x-1][y-1]) + (mask[a][1]
                                                                       * padding[x-1][y]) + (mask[a][2] * padding[x-1][y+1])
                        if (a == 1):
                            sum += (mask[a][0] * padding[x][y-1]) + (mask[a][1]
                                                                     * padding[x][y]) + (mask[a][2] * padding[x][y+1])
                        if (a == 2):
                            sum += (mask[a][0] * padding[x+1][y-1]) + (mask[a][1]
                                                                       * padding[x+1][y]) + (mask[a][2] * padding[x+1][y+1])
                    ans[x-1][y-1] = sum

        return ans
