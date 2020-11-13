import csv
import numpy as np
import random
import random
from PIL import Image
from dataLoader import loadData, createImageDictionaries
from sklearn import metrics

''' k means algorithm steps:
1. pick random representatives
2. find closest representative for each data point, give it a class
3. find average position of each data point in each class, make that the new representative
'''

# Helper function that displays an image to the screen given a 1d list of pixels:
def displayImage(pixels):
    pixArray = np.array(pixels)
    pixArray2D = np.reshape(pixArray, (-1, 48))
    img = Image.fromarray(pixArray2D)
    img.show()

# Helper function that computes the mean square distance of an image and a representative:
def meanSquareDistance(rep, dataPoint):
    sum = 0
    # Standard deviation formula:
    for i in range(len(rep)):
        squareDistance = (rep[i] - dataPoint[i])**2
        sum += squareDistance
    MSD = sum/ len(rep)
    return MSD


# Helper function that averages all the pixels in all the images of a given class
# and then returns that average image:
def findClassAvg(rep, images):
    # displayImage(rep['pix'])

    # Create empty list of pixels:
    avgPix = [0] * len(images[0]['pix'])
    classImageCount = 0
    for image in images:

        # if the image belongs to the class we are looking for:
        if image['class'] == rep['class']:

            # Skip images that are missing pixels:
            if image['pix'] == None:
                continue

            # tally up the number of images that we are averaging:
            classImageCount += 1

            # iterate over all pixels in the image and add it to the respective pixel of avgPix:
            for i in range(len(image['pix'])):
                avgPix[i] += image['pix'][i]

    # Divide each sum of pixels by the number of images to create an average:
    for i in range(len(avgPix)):
        avgPix[i] //= classImageCount
    return avgPix

# Experimental function to average all the faces for a given ethnicity (takes forever probably doesn't work)
def findAvgFace(images, ethnicities, selection):
    # Filter all images that have the selected ethnicity:
    filteredImages = []
    for j in range(len(images)):
        if int(ethnicities[j]) != selection:
            continue
        images[j] = images[j].split()
        for i in range(len(images[j])):
            images[j][i] = int(images[j][i])
            filteredImages.append(images[j])

    # Average the images:
    avgPix = [0] * len(filteredImages[0])
    for image in filteredImages:
        for i in range(len(image)):
            avgPix[i] += image[i]
    for i in range(len(avgPix)):
        avgPix[i] //= len(filteredImages)

    displayImage(avgPix)


# Attempted implementation of the kmeans algorithm the professor showed in class:
def kmeans(images):
    clusterNumber = 20 # number of expected clusters, will probably be in the order of hundreds
    iterationCount = 15 # number of times to create representatives, find distance to reps, etc.

    # pick k random images from images, using all of them takes forever:
    images = random.choices(images, k = 100)

    # PART 1: CREATE [clusterNumber] RANDOM REPRESENTATIVES:
    reps = []
    for i in range(clusterNumber):
        # Grab a random image from images, copy its pixels, and create a new dictionary from it:
        rep = {'pix': images[random.randint(0, len(images)-1)]['pix'].copy(), 'class': i}
        reps.append(rep)

    # Do the next two parts [iterationCount] times:
    for i in range(iterationCount):
        # PART 2: ASSIGN CLASS TO EACH DATA POINT (IMAGE) BASED ON CLOSEST REPRESENTATIVE:
        # Find minimum distance from an image to a representative:
        for i in range(len(images)):
            # Initialize minimum distance to the distance to the first representative:
            minDist = meanSquareDistance(reps[0]['pix'], images[i]['pix'])
            images[i]['class'] = reps[0]['class']

            # Loop over every representative but the first one (already checked it):
            for rep in reps[1:]:
                dist = meanSquareDistance(rep['pix'], images[i]['pix'])
                # if we found a closer representative: update the image to belong to its class:
                if dist < minDist:
                    minDist = dist
                    images[i]['class'] = rep['class']

        # PART 3: FIND AVERAGE POSITION OF EACH IMAGE IN EACH CLASS, MAKE THAT THE NEW REPRESENTATIVE:
        for i in range(len(reps)):
            # calculate average:
            reps[i]['pix'] = findClassAvg(reps[i], images)

    # Return the reps so we can see how they were changed:
    return reps


rows = []
pixels = []
labels = []
ethnicity = []
rows, pixels, labels, ethnicity = loadData()

images = createImageDictionaries()
reps = kmeans(images)
for rep in reps:
    displayImage(rep['pix'])
#rows = np.array(rows, dtype=object)
#pixels = np.array(pixels, dtype=object)
dataSize = (len(rows))
trainingSetSize = 19754
testDataSize = dataSize - 19754
x_train = []
x_test = []
y_train = []
y_test = []
print(len(pixels))

luckyHat = random.sample(range(23705), 19754)
luckyHat.sort(reverse=True)
for i in range(0, 19754):
    x_train.append(pixels[luckyHat[i]])
    y_train.append(ethnicity[luckyHat[i]])
    del pixels[luckyHat[i]]
    del ethnicity[luckyHat[i]]

for i in range(0, testDataSize):
    x_test.append(pixels[i])
    y_test.append(ethnicity[i])
x_train = np.array(x_train, dtype=object)
x_test = np.array(x_test, dtype=object)
y_train = np.array(y_train, dtype=object)
y_test = np.array(y_test, dtype=object)
print(x_train.shape)

##so x train is 19754 ID array but in medium tutorial it is (19754, 2304)
#2304 cuz we got 48x48 images...will fix later









#to do
#make training and test data partitions
#normalize data to be between 0-1