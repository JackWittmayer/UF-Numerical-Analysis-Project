import csv
import numpy as np
import random
import random
from PIL import Image
from dataLoader import loadData
from sklearn import metrics
# from mnist import MNIST
#
# mndata = MNIST('samples')
#
# images, labels = mndata.load_training()
# # or
# images, labels = mndata.load_testing()

''' k means algorithm steps:
1. pick random representatives
2. find closest representative for each data point, give it a color
3. find average position of each data point in each class, make that the new representative
'''
def meanSquareDistance(rep, dataPoint):
    sum = 0
    for i in range(len(rep)):
        squareDistance = (rep[i] - dataPoint[i])**2
        sum += squareDistance
    sum /= len(rep)
    return sum

def findClassAvg(rep, images):
    # pixArray = np.array(rep['pix'])
    #
    # pixArray2D = np.reshape(pixArray, (-1, 48))
    # img = Image.fromarray(pixArray2D)
    # img.show()
    avgPix = [0] * len(images[0]['pix'])
    j = 0
    for image in images:
        if image['class'] == rep['class']:
            if image['pix'] == None:
                continue
            j += 1
            for i in range(len(image['pix'])):
                avgPix[i] += image['pix'][i]
    for i in range(len(avgPix)):
        avgPix[i] //= j
    # pixArray = np.array(avgPix)
    #
    # pixArray2D = np.reshape(pixArray, (-1, 48))
    # img = Image.fromarray(pixArray2D)
    # img.show()
    return avgPix

def findAvgFace(images, ethnicities, selection):
    filteredImages = []
    for j in range(len(images)):
        if int(ethnicities[j]) != selection:
            continue
        images[j] = images[j].split()
        for i in range(len(images[j])):
            images[j][i] = int(images[j][i])
            filteredImages.append(images[j])
    avgPix = [0] * len(filteredImages[0])
    for image in filteredImages:
        for i in range(len(image)):
            avgPix[i] += image[i]
    for i in range(len(avgPix)):
        avgPix[i] //= len(filteredImages)
    pixArray = np.array(avgPix)
    pixArray2D = np.reshape(pixArray, (-1, 48))
    img = Image.fromarray(pixArray2D)
    img.show()


def kmeans(images):
    clusterNumber = 4
    images = random.choices(images, k = 100)
    # turn list of space separated pixel values into list of lists of pixel ints with class label:
    for j in range(len(images)):
        images[j] = images[j].split()
        for i in range(len(images[j])):
            images[j][i] = int(images[j][i])
        images[j] = {'pix': images[j], 'class': 0}
        # 1. Create two random representatives:
    reps = []
    for i in range(clusterNumber):
        rep = {'pix': images[random.randint(0, len(images)-1)]['pix'].copy(), 'class': i}
        reps.append(rep)

    for i in range(15):
        # 2. Assign class to each data point based on closest representative:
        for i in range(len(images)):
            minDist = meanSquareDistance(reps[0]['pix'], images[i]['pix'])
            images[i]['class'] = reps[0]['class']
            for rep in reps[1:]:
                dist = meanSquareDistance(rep['pix'], images[i]['pix'])
                if dist < minDist:
                    minDist = dist
                    images[i]['class'] = rep['class']

        # 3. find average position of each data point in each class, make that the new representative:
        for i in range(len(reps)):
            # calculate average:
            reps[i]['pix'] = findClassAvg(reps[i], images)

    return reps


rows = []
pixels = []
labels = []
ethnicity = []
rows, pixels, labels, ethnicity = loadData()

findAvgFace(pixels, ethnicity, 0)
# reps = kmeans(pixels)
# for rep in reps:
#     pixArray = np.array(rep['pix'])
#
#     pixArray2D = np.reshape(pixArray, (-1, 48))
#     img = Image.fromarray(pixArray2D)
#     img.show()
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