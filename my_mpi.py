import sys
import time
from mpi4py import MPI
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv

# Get the command-line arguments
cores = int(sys.argv[1])  # Numbers of processors
obs_data_path = sys.argv[2]  # Path to observation data
background_data_path = sys.argv[3]  # path to background data
output_name = sys.argv[4]  # output file name

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# USEFUL FUNCTIONS
def create_cubes(A, B, n):
    """Splits the observation and background data based on n(p) and creates a single array to be scattered based on split size and displacements 

    Args:
        A (ndarray): observation data
        B (ndarray): background data
        n (int): Number of process(es)

    Returns:
        combined data
        split_input_sizes
        split_output_sizes
        split_input_displacements
        split_output_displacements
    """
    #Gets the size of input observation image/data
    array_size = A.shape[0]
    #Empty list that contains split data for input to cores
    split_input_sizes = []

    data_input_length = int(array_size*2/n)
    #Empty list that contains split data for output from cores 
    split_output_sizes = []
    data_output_length = int(data_input_length/2)
    #Empty array hat contains all the data that will be scattered
    C = np.zeros(array_size*2)
    for i in range(n):
        split = int(array_size/n)
        # For every process(s). The code splits observation and background data into corresponding chunks. 
        # Then combines obs and background chunk and saved it into our main array. 
        # ultimately, every processor will get a portion of observation in 1st half and 2nd contains background data
        C[i * (2 * split): (i+1) * (2 * split)] = np.concatenate((A[i *
                                                                    split: (i+1) * split], B[i * split: (i+1) * split]))
        #Creates the data size for input and output data for each cores
        split_input_sizes.append(data_input_length)
        split_output_sizes.append(data_output_length)
    split_input_sizes = np.array(split_input_sizes)
    split_output_sizes = np.array(split_output_sizes)
    # creating the displacements/index for input and output data for each core
    split_input_displacements = np.insert(
        np.cumsum(split_input_sizes), 0, 0)[0:-1]
    split_output_displacements = np.insert(
        np.cumsum(split_output_sizes), 0, 0)[0:-1]

    return C, split_input_sizes, split_output_sizes, split_input_displacements, split_output_displacements


def divide_array(arr, latent_size):
    """Divides the given large array into parts. Each part contains latent_size = latent_size -+1.
    This was done to avoid Runtime memory error when using Kalman Filter. Because when defining I = identity(nNodes). Using large value of nNodes can cause error.    

    Args:
        arr (ndarray): Array to be split
        latent_size (int): Latent dimension

    Returns:
        list: parts 
    """
    if len(arr) % latent_size != 0:
        latent_size += 1
    num_chunks = len(arr) // latent_size
    parts = np.array_split(arr, num_chunks)
    return parts


def update_prediction(x, K, H, y):
    res = x + np.dot(K, (y - np.dot(H, x)))
    return res


def KalmanGain(B, H, R):
    tempInv = inv(R + np.dot(H, np.dot(B, H.transpose())))
    res = np.dot(B, np.dot(H.transpose(), tempInv))
    return res


if rank == 0:
    #Load the data using core:0
    background_data = np.load(background_data_path)[:5000, :, :]
    obs_data = np.load(obs_data_path)

    img_width = background_data.shape[1]
    img_height = background_data.shape[2]
    #Flatten the images
    background_flattened = background_data.flatten()
    obs_flattened = obs_data.flatten()
    #Split the images based on (p) and define split size and displacements for each processor
    split, split_sizes_input, split_sizes_output, displacements_input, displacements_output = create_cubes(
        A=obs_flattened, B=background_flattened, n=cores)
    #Output data for final data in core:0
    outputData = np.zeros(obs_flattened.shape[0])
    array_length = split_sizes_input[0]

else:
    # Create variables on other cores
    split_sizes_input = None
    displacements_input = None
    split_sizes_output = None
    displacements_output = None
    split = None
    array_length = None
    data = None
    outputData = None

# Broadcast split array, array_length, input and output displacements to other cores
split = comm.bcast(split, root=0)  
array_length = comm.bcast(array_length, root=0)
split_sizes = comm.bcast(split_sizes_input, root=0)
displacements = comm.bcast(displacements_input, root=0)
split_sizes_output = comm.bcast(split_sizes_output, root=0)
displacements_output = comm.bcast(displacements_output, root=0)

# Create array to receive subset of data on each core
output_chunk = np.zeros(array_length)
comm.Scatterv([split, split_sizes_input, displacements_input,
              MPI.DOUBLE], output_chunk, root=0)
#create a numpy array
da_final = np.array([1])
#Split the data for each core into obs and background data
# 1st half contains observation data. And divide them into parts such that each part contains size of latent dimension
obs_data = divide_array(output_chunk[: int(array_length/2)], latent_size=100)
# 2st half contains background data. And divide them into parts such that each part contains size of latent dimension
background_data = divide_array(
    output_chunk[int(array_length/2):], latent_size=100)

start = time.time()
# Data assimilation using KALMAN BLUE
for index in range(len(obs_data)):
    #Loop over every part in obs_data
    #For each part create a apply filter/update prediction
    nNodes = len(obs_data[index])
    I = np.identity(nNodes)
    R = I*0.01
    H = I  # Observation operator
    B = np.cov(background_data[index].T)
    K = KalmanGain(B, H, R)
    #Update prediction for each part inside obs_data and background_data
    updated_da = update_prediction(
        background_data[index], K, H, obs_data[index])  # compute only the analysis
    #add the updated data to da_final for each part in obs_data
    da_final = np.concatenate([da_final, updated_da])
#Remove the first element that is 1. It was first created when da_final was initialised
da_final = da_final[1:]
end = time.time()
time_taken = end - start
#Calculate MSE between observation data(1st half of input from core:0 to each core) and da_final
mse = mean_squared_error(output_chunk[: int(array_length/2)], da_final)
print(f'MSE of rank {rank} is: ', mse)
print(f'Execution time of rank {rank} is: ', time_taken)

#Wait for each core to complete execution
comm.Barrier()
#Gather all the data from other cores
comm.Gatherv(da_final, [outputData, split_sizes_output, displacements_output,
             MPI.DOUBLE], root=0)  # Gather output data together

if rank == 0:
    #Reshape the output data into images
    outputData = outputData.reshape(-1, img_width, img_height)
    #Save the into a numpy file 
    np.save(output_name, outputData)
