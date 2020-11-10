import csv
import numpy as np
from PIL import Image

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