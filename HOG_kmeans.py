import csv
import random
import random
from PIL import Image
from dataLoader import loadData, createImageDictionaries, getBabiesOldiesHOG, getRandomSampleHOG
from face_recognition import *
from helper import *
from statistics import mode
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

# experimental function to map blind labels to corresponding label in original dataset
def mapLabels(images, clusterNum, testType, reps):
    print("Mapping labels to class")
    inferred_labels = {}
    for i in range(clusterNum):
        labels = []
        # 1. Make array of all the images where class = i
        def isPartOfClass(image):
            if (image['class'] == i):
                return True
        index = list(filter(isPartOfClass, images))
        # 2. Find the actual label for each image and append it to new array
        labels = [image[testType] for image in index]
        # ignore clusters that have no images in them:
        if len(labels) == 0:
            #print("length of labels is 0")
            continue

        # 3. Find most common label for that cluster using actual labels
        most_common_label = mode(labels)

        # 4. Append that label to a dictionary. Key = most common label, value: class
        if most_common_label in inferred_labels:
            inferred_labels[most_common_label].append(i)
        else:
            inferred_labels[most_common_label] = [i]

    print(inferred_labels)

    # Change class values to actual label from inferred_labels:
    for image in images:

        # iterate over all the actual labels:
        for actual_label in inferred_labels:

            # find what actual_label corresponds to that cluster:
            if image['class'] in inferred_labels[actual_label]:

                # assign that label to the class:
                image['class'] = actual_label
                break

    # do the same for the reps:
    for rep in reps:
        # iterate over all the actual labels:
        for actual_label in inferred_labels:

            # find what actual_label corresponds to that cluster:
            if rep['class'] in inferred_labels[actual_label]:

                # assign that label to the class:
                rep['class'] = actual_label
                break


# given the reps of a completed kmeans and a set of hogdicts, predict what their class would be
def predict(reps, images):
    for image in images:
        min = meanSquareDistance(reps[0]['HOG'], image['HOG'])
        image['class'] = reps[0]['class']
        for i in range(1, len(reps)):
            if meanSquareDistance(reps[i]['HOG'], image['HOG']) < min:
                min = meanSquareDistance(reps[i]['HOG'], image['HOG'])
                image['class'] = reps[i]['class']


# Kmeans now takes in clusterNum as input, so we can run it on gender, ethnicity, or age easily
# Attempted implementation of the kmeans algorithm the professor showed in class:
def kmeans(imageInput, trainingSetSize, inputReps, clusterNum):
    print('running k means!')
    clusterNumber = clusterNum  # number of expected clusters, will probably be in the order of hundreds
    # iterations count should be dependent on convergence of Jclust function
    #trainingSetIndicies = []
    # pick k random images from images, using all of them takes forever:
    images = random.choices(imageInput, k = 1000)

    ##Data selection, tracking, then normalization

    # set = list(range(0, len(imageInput)))
    # set = list(range(0, len(imageInput)-1))
    # random.shuffle(set)
    # for i in range(trainingSetSize):
    #     images.append(imageInput[set[i]])
    #     trainingSetIndicies.append(set[i])

    # PART 1: CREATE [clusterNumber] RANDOM REPRESENTATIVES:
    reps = []
    JClusterResults = []
    if len(inputReps) == 0:
        for i in range(clusterNumber):
            randomInt = random.randint(0, len(images) - 1)
            # Grab a random image from images, copy its HOGels, and create a new dictionary from it:
            rep = {'pix': images[randomInt]['pix'].copy(), 'HOG': images[randomInt]['HOG'].copy(), 'class': i}
            reps.append(rep)
            #displayImage(rep['pix'])
    else:
        reps = inputReps
        #for rep in reps:
            #displayImage(rep['pix'])

    # Do the next two parts [iterationCount] times:
    loop = 0
    converged = False
    sumClusters(images, clusterNum)
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
        # tracks how many images are in each cluster. Makes no change to data.
        sumClusters(images, clusterNum)

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
    return reps, JClusterResults, images

# runs through a number of images and tests if their assigned class is equal to their testType value
def accuracy_test(images, trainingSetSize, testType):
    # make sure the test type is valid:
    if testType not in ['age', 'gender', 'ethnicity']:
        print('test types are: age, gender, ethnicity. You specified none of them')
        return

    count = 0
    for i in range(trainingSetSize):
        if int(images[i]['class']) == int(images[i][testType]):
            count += 1
    return count/trainingSetSize

def predetermineReps(imageData, inputReps):
    for i in range(2):
        set_ = [0, len(imageData) - 1]
        rep = {'pix': imageData[set_[i]]['pix'].copy(), 'HOG': imageData[set_[i]]['HOG'].copy(), 'class': i}
        inputReps.append(rep)
    return inputReps

# Function to run kmeans many times and return the best result:
#changed kMeans to have test type as input to make it more general
def iterateKmeans(imageInput, trainingSetSize, testType, clusterNum, maxIterations = 50):
    accuracies = []
    bestAccuracy = 0
    bestReps = []
    bestDicts = []
    worstDicts = []
    imageInput = getDistributedData(imageInput, clusterNum, testType)

    for i in range(maxIterations):
        reps, JClustResults, imageResults = kmeans(imageInput, trainingSetSize, [], clusterNum)
        # createGraphOfClusterSums(testType)
        mapLabels(imageResults, clusterNum, testType, reps)
        accuracy = accuracy_test(imageResults, trainingSetSize, testType)
        print("Accuracy of kmeans test", i + 1, ":", accuracy)
        accuracies.append(accuracy)
        if accuracy > bestAccuracy:
            bestReps = reps
            bestAccuracy = accuracy
            bestDicts = imageResults
        # Grab inverted label dictionaries
        if accuracy <= 0.05:
            worstDicts = imageResults
            print("Inverted dictionary accuracy:", accuracy)
        # reset classes to 0!
        for image in imageInput:
            image['class'] = 0
    accuracies.sort(reverse = True)
    print("Top three accuracies:", accuracies[0], accuracies[1], accuracies[2])

    # Ask to save best accuracy reps:
    answer = ""
    while answer != "y" and answer != "n":
        answer = input("Would you like to save the reps with accuracy " + str(bestAccuracy) + "? Y/N").lower()
    if answer == "y":
        fileName = input("Enter the filename of the reps you would like to write. .pkl will be added to the end.")
        saveReps(bestReps, fileName)

    return bestReps, bestDicts, worstDicts
    # Uncomment to predict the class of one face using distance to best reps:

    # closestRep = testOneFace(bestReps, processOneImage("old_dude.jpg"))
    # print(closestRep["class"])
    # closestRep = testOneFace(bestReps, processOneImage("baby.jpg"))
    # print(closestRep["class"])

# makes it so code isn't run when file is imported:
if __name__ == "__main__":
    rows = []
    HOGels = []
    imageData = []
    labels = []
    ethnicity = []

    rerun = input("Would you like to run kmeans (1) or use existing reps? (2)")
    if rerun == "1":
        testType = input("What will the images be classified on? Age (1) Gender (2), or Ethnicity (3)")
        if testType == '1':
            ageSet = input("What age dataset would you like to use? BabiesOldies (1), BabiesMiddiesOldies (2)")
            print("Okay, loading dataset...")
            if ageSet == "1":
                images = getBabiesOldiesHOG()
            elif ageSet == '2':
                images = getBabiesMiddiesOldiesHOG()
        elif testType == '2' or testType == '3':
            print("Okay, loading dataset...")
            images = getRandomSampleHOG(10000)

        clusterNumber = int(input("How many clusters would you like to have?"))
        iterationNum = int(input("How many iterations would you like kmeans to go through before it stops?"))
        testTypeMapper = {'1': 'age', '2': 'gender', '3': 'ethnicity'}
        testType = testTypeMapper[testType]
        reps, bestDicts, worstDicts = iterateKmeans(images, 1000, testType, clusterNumber, iterationNum)

    elif rerun == '2':
        fileName = input("What is the name of the file you would like to load?")
        reps = loadReps(fileName)
        testType = input("What is the classification type of these reps? Age (1) Gender (2), or Ethnicity (3)")
        testTypeMapper = {'1': 'age', '2': 'gender', '3': 'ethnicity'}
        testType = testTypeMapper[testType]

    print("Creation of reps complete. We can now use it to predict the classes of new images.")
    while True:
        print("Enter q to stop predicting")
        testImage = input("Enter the file name of the image whose " + testType + " you would like to predict.")
        if testImage.lower() == 'q':
            break
        HOGvector = processOneImage(testImage)
        prediction = testOneFace(reps, HOGvector, testType)

