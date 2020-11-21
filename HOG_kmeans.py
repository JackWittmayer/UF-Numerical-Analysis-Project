import csv
import random
import random
from PIL import Image
from dataLoader import loadData, createImageDictionaries, getBabiesOldiesHOG
#from face_recognition import *
from helper import *
import pickle

''' k means algorithm steps:
1. pick random representatives
2. find closest representative for each data point, give it a class
3. find average position of each data point in each class, make that the new representative
'''


# function to track convergence of algorithm
def Jclust(reps, images):
    sum = 0
    for rep in reps:
        for image in images:
            if rep['class'] == image['class']:
                for j in range(len(image['HOG'])):
                    sum = sum + abs(image['HOG'][j] - rep['HOG'][j]) ** 2
    return (sum / len(images))


# Attempted implementation of the kmeans algorithm the professor showed in class:
def kmeans(imageInput, trainingSetSize, inputReps):
    print('running k means!')
    clusterNumber = 2  # number of expected clusters, will probably be in the order of hundreds
    # iterations count should be dependent on convergence of Jclust function
    images = []
    trainingSetIndicies = []
    # pick k random images from images, using all of them takes forever:
    # images = random.choices(images, k = 1000)

    ##Data selection, tracking, then normalization

    set = list(range(0, len(imageInput)))
    # set = list(range(0, len(imageInput)-1))
    random.shuffle(set)
    for i in range(trainingSetSize):
        images.append(imageInput[set[i]])
        trainingSetIndicies.append(set[i])

    # PART 1: CREATE [clusterNumber] RANDOM REPRESENTATIVES:
    reps = []
    JClusterResults = []
    if len(inputReps) == 0:
        for i in range(clusterNumber):
            randomInt = random.randint(0, len(images) - 1)
            # Grab a random image from images, copy its HOGels, and create a new dictionary from it:
            rep = {'pix': images[randomInt]['pix'].copy(), 'HOG': images[randomInt]['HOG'].copy(), 'class': i}
            reps.append(rep)
            print("pix length", len(rep['pix']))
            #displayImage(rep['pix'])
    else:
        reps = inputReps
        #for rep in reps:
            #displayImage(rep['pix'])

    # Do the next two parts [iterationCount] times:
    loop = 0
    converged = False
    while (converged == False):
        loop += 1
        # PART 2: ASSIGN CLASS TO EACH DATA POINT (IMAGE) BASED ON CLOSEST REPRESENTATIVE:
        # Find minimum distance from an image to a representative:
        for i in range(len(images)):
            # Initialize minimum distance to the distance to the first representative:
            #displayImage(images[i]['pix'])
            minDist = meanSquareDistance(reps[0]['HOG'], images[i]['HOG'])
            images[i]['class'] = reps[0]['class']

            # Loop over every representative but the first one (already checked it):
            for rep in reps[1:]:
                dist = meanSquareDistance(rep['HOG'], images[i]['HOG'])
                # if we found a closer representative: update the image to belong to its class:
                if dist < minDist:
                    minDist = dist
                    images[i]['class'] = rep['class']

        # PART 3: FIND AVERAGE POSITION OF EACH IMAGE IN EACH CLASS, MAKE THAT THE NEW REPRESENTATIVE:

        for i in range(len(reps)):
            # calculate average:
            reps[i]['HOG'] = findClassAvgHOG(reps[i], images)
        JClusterResults.append(Jclust(reps, images))
        # if a couple of iterations have gone by and the Jclust function isn't decreasing by more than .5% we have converged
        if len(JClusterResults) > 4:
            if (JClusterResults[len(JClusterResults) - 1] / JClusterResults[len(JClusterResults) - 2] > .999):
                converged = True

                # Return the reps so we can see how they were changed:
    return reps, JClusterResults, images, trainingSetIndicies


def accuracy_test(images, trainingSet):
    count = 0
    # Checking for age matches
    for i in range(len(trainingSet)):
        if int(images[i]['class']) == int(images[i]['age']):
            count += 1

    return count / len(trainingSet)


def predetermineReps(imageData, inputReps):
    for i in range(2):
        set_ = [0, len(imageData) - 1]
        rep = {'pix': imageData[set_[i]]['pix'].copy(), 'HOG': imageData[set_[i]]['HOG'].copy(), 'class': i}
        inputReps.append(rep)
    return inputReps


# makes it so code isn't run when file is imported:
if __name__ == "__main__":
    rows = []
    HOGels = []
    imageData = []
    labels = []
    ethnicity = []

    # rows, imageData, ethnicity, HOGels = loadData()
    babiesOldiesData = getBabiesOldiesHOG()

    inputReps = []
    inputRepClasses = []

    inputReps = predetermineReps(babiesOldiesData, inputReps)
    reps, JClustResults, imageResults, trainingIndices = kmeans(babiesOldiesData, 100, inputReps)
    print(accuracy_test(imageResults, trainingIndices))

    print(JClustResults)
    print(len(JClustResults))

