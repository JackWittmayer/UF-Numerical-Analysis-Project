import csv
import numpy as np
import random
import random
from PIL import Image
from dataLoader import loadData
from sklearn.cluster import MiniBatchKMeans

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
    count = 0
    for image in images:

        # if the image belongs to the class we are looking for:
        if image['class'] == rep['class']:
            # Skip images that are missing pixels:
            if image['pix'] == None:
                count += 1
                continue

            # tally up the number of images that we are averaging:
            classImageCount += 1

            # iterate over all pixels in the image and add it to the respective pixel of avgPix:
            for i in range(len(image['pix'])):
                avgPix[i] += float(image['pix'][i])
    
    # Divide each sum of pixels by the number of images to create an average:
    for i in range(len(avgPix)):
        if classImageCount == 0:
            print('something is fucked!')
        avgPix[i] /= classImageCount
    return avgPix

#function to track convergence of algorithm
def Jclust(reps, images):
    sum = 0
    for rep in reps:
        for image in images:
            if rep['class'] == image['class']:
                for j in range(len(image['pix'])):
                    sum = sum + abs(image['pix'][j] - rep['pix'][j])**2
    return (sum/len(images))

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
def kmeans(imageInput, trainingSetSize, inputReps):
    clusterNumber = 4 # number of expected clusters, will probably be in the order of hundreds
    #iterations count should be dependent on convergence of Jclust function
    images = []
    trainingSetIndicies = []
    # pick k random images from images, using all of them takes forever:
    #images = random.choices(images, k = 1000)

    ##Data selection, tracking, then normalization
    
    set = list(range(0, 100))
    #set = list(range(0, len(imageInput)-1))
    random.shuffle(set)
    for i in range(trainingSetSize):
        images.append(imageInput[set[i]])
        trainingSetIndicies.append(set[i])
    
    for image in images:
        for j in range(len(image['pix'])):
            image['pix'][j] = image['pix'][j] / 255.0
        
   
    

    # PART 1: CREATE [clusterNumber] RANDOM REPRESENTATIVES:
    reps = []
    JClusterResults = []
    if len(inputReps) == 0:
        for i in range(clusterNumber):
            # Grab a random image from images, copy its pixels, and create a new dictionary from it:
            rep = {'pix': images[random.randint(0, len(images)-1)]['pix'].copy(), 'class': i}
            reps.append(rep)
    else:
        reps = inputReps

    # Do the next two parts [iterationCount] times:
    loop = 0
    converged = False
    while (converged == False):
        loop += 1
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
        JClusterResults.append(Jclust(reps, images))
        #if a couple of iterations have gone by and the Jclust function isn't decreasing by more than .5% we have converged
        if (len(JClusterResults) > 4):
            if (JClusterResults[len(JClusterResults)-2] / JClusterResults[len(JClusterResults)-2] > .999):
                converged = True 
    
    # Return the reps so we can see how they were changed:
    return reps, JClusterResults, images, trainingSetIndicies

def accuracy_test(images, ethnicity, trainingSet):
    
    count = 0
    #re-normalizing!
    for image in images:
        for j in range((len(image['pix']))):
            image['pix'][j] = image['pix'][j] * 255.0

    for i in range(len(trainingSet)):
        if i < 5: 
            displayImage(images[i]['pix'])
            displayImage(imageData[trainingSet[i]]['pix'])
        if int(images[i]['class']) == int(ethnicity[trainingSet[i]]):
            count += 1
            
    return count/len(trainingSet)

def predetermineReps(imageData, inputReps):
    for i in range(5):
        set_ = [356, 367 ,91 , 258, 197]
        rep = {'pix': imageData[set_[i]]['pix'].copy(), 'class': i}
        inputReps.append(rep)
    for rep in inputReps:
        for j in range((len(rep['pix']))):
            rep['pix'][j] = rep['pix'][j] / 255.0 
    return inputReps

rows = []
pixels = []
imageData = []
labels = []
ethnicity = []
rows, imageData, labels, ethnicity, pixels = loadData()
inputReps = []
inputRepClasses = []

inputReps = predetermineReps(imageData, inputReps)



reps, JClustResults, imageResults, trainingIndices  = kmeans(imageData, 100, inputReps)
print(accuracy_test(imageResults, ethnicity, trainingIndices))

print(JClustResults)
print(len(JClustResults))


""" for rep in reps:
    
    for i in range(len(rep['pix'])):
        rep['pix'][i] = rep['pix'][i] * 255
    displayImage(rep['pix']) """



#rows = np.array(rows, dtype=object)
#pixels = np.array(pixels, dtype=object)
""" dataSize = (len(rows))
trainingSetSize = 19754
testDataSize = dataSize - 19754
x_train = []
x_test = []
y_train = []
y_test = []

luckyHat = random.sample(range(23705), 1000)
luckyHat.sort(reverse=True)
for i in range(0, len(luckyHat)):
    x_train.append(pixels[luckyHat[i]])
    y_train.append(ethnicity[luckyHat[i]])
    del pixels[luckyHat[i]]
    del ethnicity[luckyHat[i]]

for i in range(0, testDataSize):
    x_test.append(pixels[i])
    y_test.append(ethnicity[i])
x_train = np.array(x_train, dtype=np.uint8)
x_test = np.array(x_test, dtype=np.uint8)
x_train = int(x_train / 255.0)
x_test = int(x_test / 255.0)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(x_train.shape)
 
kMeansAlg = MiniBatchKMeans(n_clusters=4).fit(x_train)
predictions = kMeansAlg.predict(x_test)

for i in range(len(predictions)):
    count = 0
    if int(predictions[i]) == int(y_test[i]):
        count += 1
print('accuracy of library func: ' + str(count/len(predictions))) """









#to do
#make training and test data partitions
#normalize data to be between 0-1