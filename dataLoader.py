import csv
import numpy as np
from PIL import Image

def loadData():
    rows = []
    pixels = []
    pixelData = []
    ethnicity = []
    with open('dataset/age_gender.csv', 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        fields = next(data)
        

        for row in data:
            rows.append(row) 
            pixelData.append(row[4])
            ethnicity.append(row[1])
        #this will make pixels a list of lists of ints
        
        for image in pixelData:
            pixels.append(image.split())
            for i in range(len(pixels[len(pixels)-1])):
                pixels[len(pixels)-1][i] = (int(pixels[len(pixels)-1][i]))
        
        return rows, pixels, fields, ethnicity
        #48*48 images
        
    
    

#to do
#make training and test data partitions
#normalize data to be between 0-1