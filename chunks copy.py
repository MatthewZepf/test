import pandas as pd
import os

# read the first 20 rows in V-Dem-CD-v15.csv
# and print the described df value for v2elpubfin
def read():
    input_file = "V-Dem-CD-v15.csv"
    reader = pd.read_csv(input_file, nrows=20)
    print(reader['v2elpubfin'].describe())

read()