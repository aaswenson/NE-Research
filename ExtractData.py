"""
This file extracts the information used to train the AI.
@author: Virat Singh, vsingh28@wisc.edu
"""

import os, tarfile

# Folder containing all the data tar files
data_folder = os.getcwd()

# Contains the time input fields
# time[i][j] contains the jth time field in the ith .dat file
times = []

# Contains the inv input fields
# quants[i][j][k] signifies the kth field in the jth inv line of the ith .dat file
# We ignore the values in 0, 1, 2 in inv as those are not quantities, but A, Z, I values
quants = []

# Gets the total quantity at some time t
# quant_total[i][j] give you the total quantity at time j in the ith .dat file
quant_total = []

# Contains the specific power input fields
# spec_pow[i] contains the ith specific power field in the ith .info file
spec_pow = []

# Contains only the quantities at time = 0
# initials[i][j] contains the initial quantity of element j of the ith .dat file
initials = []

# Filtered initial quantities
f_initials = []

def uncompress():
    """
    Uncompresses the .tar.gz files and only saves the .dat files, as this is where
    the data is saved.
    """
    os.chdir(data_folder)
    for file in os.listdir(data_folder):
        if file.endswith(".tar.gz"):
            # Read the compressed tar file
            tar = tarfile.open(data_folder + "\\" + file, "r")
            mems = tar.getmembers()
            # Extract the data files
            tar.extract(mems[0])
            # Extract the info files
            tar.extract(mems[2])

def fill_times(file):
    """
    Fills the time input fields.
    :param file: the file to read from
    """
    with open(data_folder + "/" + file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("time "):
                line = line.rstrip()
                line = line.split("time ")[1]
                times.append(line.split())

def fill_quants(file):
    """
    Fills the inv input fields.
    :param file: the file to read from
    """
    # the inv fields of one particular file
    inv = []
    with open(data_folder + "/" + file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Inv "):
                line = line.rstrip()
                line = line.split("Inv ")[1]
                line = line.split()
                # Removing the A, Z, I values and appending
                #print(line[3])
                #if line[3] != '0':
                inv.append(line[3:])
    quants.append(inv)

def fill_spec_pow(file):
    """
    Fills the specific power input fields. Specific power is calculated as such:
    specific power = constant power / heavy metal assembly mass
    :param file: the file to read from
    """
    with open(data_folder + "/" + file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            line = line.split(" ")
            if line[0] == "AssemblyHeavyMetalMass":
                metal_mass = line[1]
            if line[0] == "ConstantPower":
                pow = line[1]
    spec_pow.append(float(pow)/float(metal_mass))

def fill_quant_total():
    """
    Filling the total quantity at each time level
    """
    for i in range(0, len(quants)):
        qt = []
        for k in range(0, 80):
            total = 0
            for j in range(0, len(quants[i])):
                total += float(quants[i][j][k])
            qt.append(total)
        quant_total.append(qt)

def save():
    """
    Saves the extracted data for future use.
    The input files are in this format:
    Line 1:         Times (0 through 80)
    Line 2 - 654:   Quantities
    Line 655:       Total Quantity at the time
    Line 656:       Specific power for the file
    """
    for i in range(0, len(times)):
        with open("inputs" + str(i) + ".txt", "w") as f:
            for j in range(0, len(times[i])):
                f.write(times[i][j] + " ")
            f.write("\n")
            for j in range(0, len(quants[i])):
                for k in range(0, len(quants[i][j])):
                    f.write(quants[i][j][k] + " ")
                f.write("\n")
            for j in range(0, len(quant_total[i])):
                f.write(str(quant_total[i][j]) + " ")
            f.write("\n")
            f.write(str(spec_pow[i]))

def get_initial_quants():
    """
    Gets the initial quantities.
    """
    for i in range(len(quants)):
        inits = []
        for j in range(len(quants[i])):
            inits.append(quants[i][j][0])
        initials.append(inits)

def filter():
    """
    Removes the zero-quantities at time = 0
    """
    removed = []
    for i in range(len(initials)):
        filt = []
        rem = []
        for j in range(len(initials[i])):
            if initials[i][j] != '0':
                filt.append(initials[i][j])
            else:
                rem.append(j)
        f_initials.append(filt)
        removed.append(rem)

def run():
    """
    Simply runs the program to extract data
    """
    # Uncompress the files to extract the data
    uncompress()

    # Go through each file and fill in the lists with appropriate data
    for file in os.listdir(data_folder):
        if file.endswith(".dat"):
            fill_times(file)
            fill_quants(file)
            os.remove(file)
        if file.endswith(".info"):
            fill_spec_pow(file)
            os.remove(file)

    # Fill the total quantity at each time t
    fill_quant_total()

    # Remove the zero's from the quants at time = 0, and corresponding data
    get_initial_quants()
    filter()

    # Save the data
    #save()

if __name__ == "__main__":
    run()