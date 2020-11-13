import csv
import numpy as np
from PIL import Image

# Turn csv file into list of dictionaries containing values for pixels, ethnicity, age, etc:
def createImageDictionaries():
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

            imageDict = {'age': age, 'ethnicity': ethnicity, 'gender': gender, 'imgname': imgname, 'pix': pixels,
                         'class': 0}
            images.append(imageDict)
        return images


def loadData():
    rows = []
    pixels = []
    pixelData = []
    ethnicity = []
    with open('../age_gender.csv', 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        fields = next(data)
        

        for row in data:
            rows.append(row) 
            pixels.append(row[4])
            ethnicity.append(row[1])
        
        
        return rows, pixels, fields, ethnicity
        #48*48 images
        
    
    

#to do
#make training and test data partitions
#normalize data to be between 0-1