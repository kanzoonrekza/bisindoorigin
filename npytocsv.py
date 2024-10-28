import numpy as np
import csv

# Load the .npy file
filename = 'a_5'
data = np.load('dataset/' + filename[0] + "/" + filename + '.npy')

# Open a CSV file to write
with open(filename+'.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')

    # Write each row as a single string in one cell
    for row in data:
        writer.writerow(row)
