import csv
import numpy as np
from PIL import Image
import pickle

# Turn csv file into list of dictionaries containing values for pixels, ethnicity, age, etc:

# Turn csv file into list of dictionaries containing values for pixels, ethnicity, age, etc:
def createImageDictionaries():
    # Attempt to read pickle file called "imageDicts" before reading CSV...
    # Can read pickle file MUCH faster than CSV. FASTER IS BETTER.
    try:
        infile = open("imageDicts", 'rb')
        images = pickle.load(infile)
        infile.close()
        return images
    except IOError:
        print("No imageDicts pickle file found. Creating it...")
        images = []
        with open('../age_gender.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='|')
            fields = next(data)
            for row in data:
                age = row[0]
                ethnicity = row[1]
                gender = row[2]
                imgname = row[3]
                pixels = row[4].split()

                # Turn list of strings into list of ints:
                for i in range(len(pixels)):
                    pixels[i] = int(pixels[i])

                imageDict = {'age': int(age), 'ethnicity': int(ethnicity), 'gender': int(gender), 'imgname': imgname, 'pix': pixels,


                             'class': 0}
                images.append(imageDict)

            # Write imageDict to pickled file to make initial data loading faster:
            outfile = open("imageDicts", 'wb')
            pickle.dump(images, outfile)
            outfile.close()
            return images



def getBabiesOldies():
    images = createImageDictionaries()
    output = []
    for image in images:
        if image['age'] < 2 or image['age'] > 90:
            if image['age'] < 3:
                image['age'] = 0
            if image['age'] > 90:
                image['age'] = 1
            output.append(image) 
    return output 

def loadData():
    rows = []
    images = []
    pixelData = []
    ethnicity = []
    with open('./dataset/age_gender.csv', 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        fields = next(data)
        

        for row in data:
            rows.append(row) 
            images.append(row[4])
            pixelData.append(row[4])
            ethnicity.append(row[1])
        
        for j in range(len(images)):
        # Turn space separated string of numbers into list of number strings:
            images[j] = images[j].split() # "123 234 212" -> ['123, '234', '212']
            pixelData[j] = pixelData[j].split()
        # Turn every string number into an integer:
            for i in range(len(images[j])):
                images[j][i] = int(images[j][i])
                pixelData[j][i] = int(pixelData[j][i])

        # Turn list of lists of pixels into a list of dictionaries containing entries for
        # a list of pixels ('pix) and a class value ('class'):
            images[j] = {'pix': images[j], 'class': 0}
        
        return rows, images, ethnicity, pixelData
        #48*48 images