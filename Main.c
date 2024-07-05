from mpi4py import MPI
import numpy as np
import time

# Size of the matrix (NxN)
N = 64

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Whether to print the matrix when completed
print_results = True

def print_matrix(matrix):
    for row in matrix:
        print("\t".join(map(str, row)))
    print()

# Define matrices
matrix1 = np.zeros((N, N), dtype=int)
matrix2 = np.zeros((N, N), dtype=int)
product_matrix = np.zeros((N, N), dtype=int)

# Counter variables
i = j = k = 0

if size == 1:
    # Single processor: perform the entire multiplication
    matrix1 = np.random.randint(1, 7, size=(N, N))
    matrix2 = np.random.randint(1, 7, size=(N, N))

    begin = time.time()

    print(f"\nMultiplying a {N}x{N} matrix using 1 processor.\n")

    # Perform matrix multiplication
    product_matrix = np.dot(matrix1, matrix2)

    end = time.time()

    if print_results:
        print("Matrix 1:")
        print_matrix(matrix1)
        print("Matrix 2:")
        print_matrix(matrix2)
        print("Product Matrix:")
        print_matrix(product_matrix)

    print(f"Runtime: {end - begin} seconds")

else:
    # Number of workers
    number_of_workers = size - 1

    if rank == 0:
        # Manager process
        begin = time.time()

        print(f"\nMultiplying a {N}x{N} matrix using {size} processor(s).\n")

        # Populate the matrices with values
        matrix1 = np.random.randint(1, 7, size=(N, N))
        matrix2 = np.random.randint(1, 7, size=(N, N))

        # Send the matrix to the worker processes
        rows = N // number_of_workers
        matrix_subset = 0

        for dest in range(1, number_of_workers + 1):
            comm.send(matrix_subset, dest=dest, tag=1)
            comm.send(rows, dest=dest, tag=1)
            comm.Send([matrix1[matrix_subset:matrix_subset + rows, :], MPI.INT], dest=dest, tag=1)
            comm.Send([matrix2, MPI.INT], dest=dest, tag=1)
            matrix_subset += rows

        # Retrieve results from all worker processors
        for i in range(1, number_of_workers + 1):
            matrix_subset = comm.recv(source=i, tag=2)
            rows = comm.recv(source=i, tag=2)
            comm.Recv([product_matrix[matrix_subset:matrix_subset + rows, :], MPI.INT], source=i, tag=2)

        end = time.time()

        if print_results:
            print("Matrix 1:")
            print_matrix(matrix1)
            print("Matrix 2:")
            print_matrix(matrix2)
            print("Product Matrix:")
            print_matrix(product_matrix)

        print(f"Runtime: {end - begin} seconds")

    else:
        # Worker process
        matrix_subset = comm.recv(source=0, tag=1)
        rows = comm.recv(source=0, tag=1)
        sub_matrix1 = np.zeros((rows, N), dtype=int)
        comm.Recv([sub_matrix1, MPI.INT], source=0, tag=1)
        comm.Recv([matrix2, MPI.INT], source=0, tag=1)

        # Perform matrix multiplication
        sub_product = np.dot(sub_matrix1, matrix2)

        comm.send(matrix_subset, dest=0, tag=2)
        comm.send(rows, dest=0, tag=2)
        comm.Send([sub_product, MPI.INT], dest=0, tag=2)

MPI.Finalize()
