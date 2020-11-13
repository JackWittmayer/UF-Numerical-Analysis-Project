import csv
import numpy as np
from PIL import Image

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
        
        return rows, images, fields, ethnicity, pixelData
        #48*48 images
        
    
    

#to do
#make training and test data partitions
#normalize data to be between 0-1