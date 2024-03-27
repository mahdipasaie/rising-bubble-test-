import fenics as fe
import numpy as np
from mpi4py import MPI
from tqdm import tqdm
from CahnHiliard_solo import update_solver_on_new_mesh_pf
from ns_cahn import update_solver_on_new_mesh_ns

fe.set_log_level(fe.LogLevel.ERROR)

#################### Define Parallel Variables ####################
# Get the global communicator
comm = MPI.COMM_WORLD
# Get the rank of the process
rank = comm.Get_rank()
# Get the size of the communicator (total number of processes)
size = comm.Get_size()
#############################  END  ################################

# Parameters all should be in SI unit 
physical_parameters_dict = {
    "dy": 1/80 ,
    "max_level": 2,
    "Nx":  1,
    "Ny": 1.5,
    # "dt": dy/16,
    "dy_coarse":lambda max_level, dy: 2**max_level * dy,
    "M": 0.2*5E-5 ,
    "A": 0.25,
    "kappa": 5E-5, #Kappa = dy
    "theta": 0.5,  #crank-niclson
    "seed_center": lambda Nx, dy: [ Nx/2, 0.5 ],
    "initial_seed_radius": 0.25, 
    "Domain": lambda Nx, Ny: [(0.0, 0.0), (Nx, Ny)] ,

    "sigma": 1.96, # N/m
    # "kesi" : dy /2 , 
    "zay" : 1 , 

    ###################### Parameters for NS  ######################

    "rho_air": 100  , # kg/m^3 
    "rho_liq": 1000  , # kg/m^3 
    "mu_air": 1 , # Pa.s 
    "mu_liq": 10, # Pa.s 
    "gravity": -0.98, # m/s^2 
    "lid_vel_x": 0.0, 
    "lid_vel_y": 0.0,

    ###################### SOLVER PARAMETERS ######################

    "abs_tol_pf": 1E-6,  
    "rel_tol_pf": 1E-5,  
    "preconditioner_ns": 'ilu',  
    'maximum_iterations_ns': 50, 
    'nonlinear_solver_pf': 'snes',     # "newton" , 'snes'
    'linear_solver_pf': 'mumps',       # "mumps" , "superlu_dist", 'cg', 'gmres', 'bicgstab'
    "preconditioner_pf": 'ilu',       # 'hypre_amg', 'ilu', 'jacobi'
    'maximum_iterations_pf': 50,

    #############
     "interface_threshold_gradient": 0.001,

}



dy = physical_parameters_dict['dy']
max_level = physical_parameters_dict['max_level']
Nx = physical_parameters_dict['Nx']
Ny = physical_parameters_dict['Ny']
physical_parameters_dict['dt'] = dy/ 16
physical_parameters_dict['kesi'] = dy/ 2
dt = physical_parameters_dict['dt'] 

# Compute values from functions
dy_coarse = physical_parameters_dict['dy_coarse'](max_level, dy)
Domain = physical_parameters_dict['Domain'](Nx, Ny)
seed_center = physical_parameters_dict['seed_center'](Nx, dy)


# Defining the mesh 
nx = (int)(Nx/ dy ) 
ny = (int)(Ny / dy ) 
nx_coarse = (int)(Nx/ dy_coarse ) 
ny_coarse = (int)(Ny / dy_coarse ) 

coarse_mesh = fe.RectangleMesh( fe.Point(0.0 , 0.0 ), fe.Point(Nx, Ny), nx_coarse, ny_coarse  )
mesh = fe.RectangleMesh( fe.Point(0.0 , 0.0 ), fe.Point(Nx, Ny), nx, ny )



####### writing to file ######## 


file = fe.XDMFFile("rising_bub.xdmf")


def write_simulation_data(Sol_Func, time, file, variable_names ):

    
    # Configure file parameters
    file.parameters["rewrite_function_mesh"] = True
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True

    # Split the combined function into its components
    functions = Sol_Func.split(deepcopy=True)

    # Check if the number of variable names matches the number of functions
    if variable_names and len(variable_names) != len(functions):
        raise ValueError("The number of variable names must match the number of functions.")

    # Rename and write each function to the file
    for i, func in enumerate(functions):
        name = variable_names[i] if variable_names else f"Variable_{i}"
        func.rename(name, "solution")
        file.write(func, time)

    file.close()


##############################################################
old_solution_vector_ns = None
old_solution_vector_pf = None
old_solution_vector_0_ns = None
old_solution_vector_0_pf = None
##############################################################

########################## Initialize  problem ##############################

####### unpacking Phase-field problem 
pf_problem_dict = update_solver_on_new_mesh_pf(mesh, physical_parameters_dict, old_solution_vector_pf= None, old_solution_vector_0_pf= None, 
                                old_solution_vector_0_ns= None, variables_dict_pf= None)

# variables for solving the problem pf
solver_pf = pf_problem_dict["solver_pf"]
solution_vector_pf = pf_problem_dict["solution_vector_pf"]
solution_vector_pf_0 = pf_problem_dict["solution_vector_pf_0"]
spaces_pf = pf_problem_dict["spaces_pf"]
variables_dict_pf = pf_problem_dict["variables_dict_pf"]
vel_answer_on_pf_mesh = pf_problem_dict["vel_answer_on_pf_mesh"]


#### unpacking Navier-stockes equation ( here we use Initial condition of Phi variable )

ns_problem_dict = update_solver_on_new_mesh_ns(mesh, physical_parameters_dict, old_solution_vector_ns= None, old_solution_vector_0_ns=None, 
                                old_solution_vector_0_pf=solution_vector_pf_0, variables_dict= None)

# variable for solving the ns problem 
solver_ns = ns_problem_dict["solver_ns"]
solution_vector_ns = ns_problem_dict['solution_vector_ns']
solution_vector_ns_0 = ns_problem_dict['solution_vector_ns_0']
phi_prev_interpolated_on_ns_mesh = ns_problem_dict['phi_prev_interpolated_on_ns_mesh']
spaces_ns = ns_problem_dict["spaces_ns"]
function_space_ns = ns_problem_dict['function_space_ns']
variables_dict_ns = ns_problem_dict["variables_dict"]
Bc = ns_problem_dict["Bc"]

########################## END ##############################


def write_simulation_data_to_single_file(solution_vectors, times, file, variable_names_list, extra_funcs_dict):

    # Configure file parameters
    file.parameters["rewrite_function_mesh"] = True
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True

    for Sol_Func, time, variable_names in zip(solution_vectors, times, variable_names_list):
        # Split the combined function into its components
        functions = Sol_Func.split(deepcopy=True)

        # Check if the number of variable names matches the number of functions
        if variable_names and len(variable_names) != len(functions):
            raise ValueError("The number of variable names must match the number of functions.")

        # Rename and write each function to the file
        for i, func in enumerate(functions):
            name = variable_names[i] if variable_names else f"Variable_{i}"
            func.rename(name, "solution")
            file.write(func, time)

    # Write extra functions (like viscosity, velocity_PF) if provided
    for name, func in extra_funcs_dict.items():
        func.rename(name, "solution")
        file.write(func, times[-1])  # Assuming extra functions correspond to the last time point


    file.close()

T = 0
for it in tqdm( range(0, 10000000) ):


    # solving the problem

    solver_pf_information = solver_pf.solve()
    solver_ns_information = solver_ns.solve()

    #definning old solution vectors
    old_solution_vector_ns = solution_vector_ns
    old_solution_vector_pf = solution_vector_pf 
    old_solution_vector_0_ns = solution_vector_ns
    old_solution_vector_0_pf = solution_vector_pf

    #update the old solution vectors
    solution_vector_ns_0.assign(solution_vector_ns)
    solution_vector_pf_0.assign(solution_vector_pf)
    

    # update Time
    T += dt

    ####### updating PF problem
    pf_problem_dict = update_solver_on_new_mesh_pf(mesh, physical_parameters_dict, old_solution_vector_pf= None, old_solution_vector_0_pf=None, 
                    old_solution_vector_0_ns = old_solution_vector_0_ns, variables_dict_pf= variables_dict_pf)
    
    # variables for solving the problem pf

    solver_pf = pf_problem_dict["solver_pf"]

    solution_vector_pf = pf_problem_dict["solution_vector_pf"]
    solution_vector_pf_0 = pf_problem_dict["solution_vector_pf_0"]
    spaces_pf = pf_problem_dict["spaces_pf"]
    variables_dict_pf = pf_problem_dict["variables_dict_pf"]
    vel_answer_on_pf_mesh = pf_problem_dict["vel_answer_on_pf_mesh"]


    ####### updating NS problem
    ns_problem_dict = update_solver_on_new_mesh_ns(mesh, physical_parameters_dict, old_solution_vector_ns= None, old_solution_vector_0_ns=None, 
                            old_solution_vector_0_pf=old_solution_vector_0_pf, variables_dict= variables_dict_ns)
    

    # variable for solving the ns problem 
    solver_ns = ns_problem_dict["solver_ns"]

    solution_vector_ns = ns_problem_dict['solution_vector_ns']
    solution_vector_ns_0 = ns_problem_dict['solution_vector_ns_0']
    spaces_ns = ns_problem_dict["spaces_ns"]
    function_space_ns = ns_problem_dict['function_space_ns']
    variables_dict_ns = ns_problem_dict["variables_dict"]
    Bc = ns_problem_dict["Bc"]
    

    ####### write first solution to file ########
    if it % 10 == 0: 

        solution_vectors = [solution_vector_ns_0, solution_vector_pf_0]
        times = [T, T]  # Assuming T_ns and T_pf are defined times for NS and PF solutions
        variable_names_list = [["Vel", "Press"], ["C", "mu"]]  # Adjust variable names as needed
        extra_funcs_dict = {"pf_on_ns": phi_prev_interpolated_on_ns_mesh, "velocity_PF": vel_answer_on_pf_mesh}  # Assuming these are defined
        write_simulation_data_to_single_file(solution_vectors, times, file, variable_names_list, extra_funcs_dict)




