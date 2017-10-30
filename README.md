[![Vehicle detection and tracking](http://img.youtube.com/vi/W8TCPlMTOyY/0.jpg)](https://www.youtube.com/watch?v=W8TCPlMTOyY)

This is my solution to project 5 of Udacity self-driving car nanodegree. The goal is to detect and track cars on video stream.

Code:
- All commented code can be found at vehicle-detection-and-tracking-v2.ipynb jupyter notebook.
- test_images/ contains images of road to test and finetune pipeline
- project\_video.mp4, challenge\_video.mp4, harder\_challenge_video.mp4 - videos for pipeline testing
- dataset/ - folder for datasets unpacking

# Overview

Project code consist of following steps:

1. Load datasets
2. Extract features from datasets images
3. Train classifier to detect cars. (I use simple default SVM with rbf kernel)
4. Scan video frame with sliding windows and detect hot boxes
5. Use hot boxes to estimate cars positions and sizes
6. Use hot boxes from previous steps to remove false positives hot boxes and make detection more robust

# Datasets

In this project I use two datasets. First is project dataset. It is splitted into [cars images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-car images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). Here is examples of dataset images:

![default dataset images](https://github.com/parilo/carnd-vehicle-detection-and-tracking/blob/master/default-dataset.png)

After playing with original dataset I found that it has bias towards black cars. Autti dataset solved this problem for me along with increasing performance of the classifier. I augment original dataset with 10000 car images and 40000 non-car images from Autti. By changing proportion of original and Autti dataset images in training samples you may fine tune classifier performance.
You may get Autti dataset from [here](https://github.com/udacity/self-driving-car/tree/master/annotations). This dataset contains of road images with labeled cars, pedestrians and other road participants. So it is needed to extract car and non-car images from original images. Here is examples of Autti dataset cars and non-cars.

![autti dataset images] (https://github.com/parilo/carnd-vehicle-detection-and-tracking/blob/master/autti-dataset.png)

# Histogram of Oriented Gradients (HOG)

After playing with picture pixels, histogram and HOG features I decided to use only little amount of HOG features. My feature vector consist of 128 components which I extract from grayscaled images, since grayscaled images contains all sctructure information. I beleive that it is better to detect cars by only structure information and avoid color information because cars may have big variety of coloring. Small amount of featues help to make the classifier faster while loosing a little amount of accuracy. My parameters of feature extraction are

```python
color_space = 'GRAY' # Can be GRAY, RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 1 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
```

Here is examples of my HOG features

![used hog featues] (https://github.com/parilo/carnd-vehicle-detection-and-tracking/blob/master/hog.png)

# SVM classifier

I decided to use SVM classifier with rbf kernel and default sklearn parameters. With my small amount of features it shows time performance about 4 frames per second for whole pipeline. After combining datasets I use 13790 cars and 48163 non-cars images for training. Resulting accuracy is about 98%. Also I used [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) for feature normalization along resulting dataset. I found that it is really importaint step. It adds about 4% accuracy for my classifier.

# Sliding windows

For searching cars in an input image I use sliding window technics. It means that I iterate over image area that could contain cars with approximately car sized box and try to classify whether box contain car or not. As cars may be of different sizes due to distance from a camera we need a several amount of box sizes for near and far cars. I use 3 square sliding window sizes of 128, 96 and 80 pixels side size. While iterating I use 50% window overlapping in horizontal and vertical directions. Here is an examples of sliding windows lattices which I use. One of sliding window drawn in blue on each image while rest of the lattice are drawn in black. For computational economy and additional robustness areas of sliding windows don't conver whole image but places where cars appearance is more probable.

![sliding window lattice] (https://github.com/parilo/carnd-vehicle-detection-and-tracking/blob/master/sliding-windows.png)

# Estimation of car positions and sizes

After sliding window application we have hot boxes - sliding window positions that were classified as containing car. Number of hot boxes crowded at real cars positions along with less number of hot boxes crowded in false positives of classifier. We need to estimate real cars positions and sizes based on this information. I propose simple algorithm for that.

We have many less or more overlapped boxes and we need to
join it around peaks to convert many overlapped boxes into
smaller amount of not or slightly overlapped ones.
Idea is take fist box (called average box) form input boxes
and join it with all boxes
that is close enough (here for two boxes:
they need to overlap by 30% of area of any one of two)
After joining two boxes we need to update average box (here
just increasing size to cover both joining boxes).
Loop while we are able to join futhermore.
For left boxes repeat all procedure.
As a result we get list of joined boxes, average boxes strengths, the number of
boxes it was joined to. Based on this we may estimate average boxes size and positions as following:

```python
    def get_box (self):
        """Uses joined boxes information to compute
        this average box representation as hot box.
        This box has average center of all boxes and have
        size of 2 standard deviation by x and y coordinates of its points

	self.boxes is joined hot boxes
        """
        if len(self.boxes) > 1:
            center = np.average (np.average (self.boxes, axis=1), axis=0).astype(np.int32).tolist()

            # getting all x and y coordinates of
            # all corners of joined boxes separately
            xs = np.array(self.boxes) [:,:,0]
            ys = np.array(self.boxes) [:,:,1]

            half_width = int(np.std (xs))
            half_height = int(np.std (ys))
            return (
                (
                    center[0] - half_width,
                    center[1] - half_height
                ), (
                    center[0] + half_width,
                    center[1] + half_height
                ))
        else:
            return self.boxes [0]
```

I use thresholding by average boxes strength to filter out false positives.
Here you can see examples of algorithm results. Overlapping by hotboxes shown as heat map where each pixel holds number of overlapped hot boxes.

![average boxes examples] (https://github.com/parilo/carnd-vehicle-detection-and-tracking/blob/master/one-image-detection-examples.png)

# Video processing

Same average boxes algorithm may be used to estimate cars base on last several frames of the video. We just need to accumulate hot boxes over number of last frames and then apply same algorithm here with higher threshold. See example of processed video at begining of the description. With described pipeline I get about 4 frames per second performance.

# Conclusion

Detecting cars with SVM in sliding windows is interesting method but it has a number of disadvantages. While trying to make my classifier more quick I faced with problem that it triggers not only on cars but on other parts of an image that is far from car look like. So it doesn't generalizes well and produces lot of false positives in some situations. To struggle this I used bigger amount of non-car images for SVM training. Also sliding windows slowes computation as it requires many classifier tries per image. Again for computational reduction not whole area of input image is scanned. So when road has another placement in the image like in strong curved turns or camera movements sliding windows may fail to detect cars.

I think this is interesting approach for starting in this field. But it is not ready for production use. I think convolutional neural network approach may show more robustness and speed. As it could be easily accelerated via GPU. Also it may let to locate cars in just one try. For example we may ask CNN to calculate number of cars in the image. And by activated neurons locate positions of the cars. In that case SVM approach may help to generate additional samples for CNN training.

