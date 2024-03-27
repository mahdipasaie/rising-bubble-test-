
import fenics as fe
import numpy as np
import random
from dolfin import MPI


class InitialConditions_pf(fe.UserExpression):
    def __init__(self, physical_parameters_dict, **kwargs):

        dy = physical_parameters_dict['dy']
        Nx = physical_parameters_dict['Nx']
        seed_center = physical_parameters_dict['seed_center'](Nx, dy)

        self.center_seed= seed_center
        self.rad = physical_parameters_dict['initial_seed_radius']
        self.XC = self.center_seed[0]
        self.YC = self.center_seed[1]

        super().__init__(**kwargs)


    def eval(self, values, x):
        x_p = x[0]
        y_p = x[1]

        # Calculate the squared distance from the point to the center of the circle
        distance_squared = (x_p - self.XC)**2 + (y_p - self.YC)**2

        # Check if the point is inside the circle
        if distance_squared < self.rad**2:
            values[0] = 1  # Inside the circle conctration of air
        else:
            values[0] = -1.0  # Outside the circle conctration of liquid

        values[1] = 0.0

    def value_shape(self):
        return (2,)


def define_variables(mesh):
    
    P1 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # Order parameter C1
    P2 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # mu 

    P3 = fe.VectorElement("Lagrange", mesh.ufl_cell(), 2)  # Velocity


    velocity_func_space = fe.FunctionSpace(mesh, P3)
    v_answer_on_pf_mesh = fe.Function(velocity_func_space)

    element = fe.MixedElement( [P1, P2] )

    function_space_pf = fe.FunctionSpace( mesh, element )

    Test_Functions = fe.TestFunctions(function_space_pf)
    test_1_pf, test_2_pf= Test_Functions



    solution_vector_pf = fe.Function(function_space_pf)  
    solution_vector_pf_0 = fe.Function(function_space_pf)  

    c_answer, mu_answer = fe.split(solution_vector_pf)  # Current solution
    c_answer_prev, mu_answer_prev = fe.split(solution_vector_pf_0)  # Previous solution


    num_subs = function_space_pf.num_sub_spaces()
    spaces_pf, maps_pf = [], []
    for i in range(num_subs):
        space_i, map_i = function_space_pf.sub(i).collapse(collapsed_dofs=True)
        spaces_pf.append(space_i)
        maps_pf.append(map_i)

    variables_dict = {
        "velocity_func_space": velocity_func_space,
        "v_answer_on_pf_mesh": v_answer_on_pf_mesh,
        "function_space_pf": function_space_pf,
        "Test_Functions": Test_Functions,
        "test_1_pf": test_1_pf,
        "test_2_pf": test_2_pf,
        "solution_vector_pf": solution_vector_pf,
        "solution_vector_pf_0": solution_vector_pf_0,
        "c_answer": c_answer,
        "mu_answer": mu_answer,
        "c_answer_prev": c_answer_prev,
        "mu_answer_prev": mu_answer_prev,
        "spaces_pf": spaces_pf,
        "maps_pf": maps_pf
    }

    return variables_dict


def derivative_f_chem(C, physical_parameters_dict): 

    A = physical_parameters_dict["A"]

    

    return A* ( 2 * ( C -1 )* C * (2*C -1 ) )



def Eq1(variables_dict_pf, physical_parameters_dict ): 

    c_answer = variables_dict_pf["c_answer"]
    c_answer_prev = variables_dict_pf["c_answer_prev"]
    mu_answer = variables_dict_pf["mu_answer"]
    mu_answer_prev = variables_dict_pf["mu_answer_prev"]
    test_1_pf = variables_dict_pf["test_1_pf"]
    vel_answer = variables_dict_pf["v_answer_on_pf_mesh"]
   
    theta = physical_parameters_dict["theta"]
    dt = physical_parameters_dict["dt"]
    # M = physical_parameters_dict["M"]
    sigma = physical_parameters_dict["sigma"]
    zay = physical_parameters_dict["zay"]
    kesi = physical_parameters_dict["kesi"]
     
    constant = 3* sigma *zay *kesi/ fe.sqrt(8) 

    mu_mid = (1.0-theta)*mu_answer_prev + theta*mu_answer
    
    # Weak form of the concentration evolution equation
    # F1 = fe.inner((c_answer - c_answer_prev)/dt, test_1_pf)*fe.dx \
    #     + M*fe.inner(fe.grad(mu_mid), fe.grad(test_1_pf))*fe.dx \
    #     + fe.inner(fe.dot(vel_answer, fe.grad(c_answer)), test_1_pf) * fe.dx # convection

    F1 = fe.inner((c_answer - c_answer_prev)/dt, test_1_pf)*fe.dx \
        + fe.inner(fe.dot(vel_answer, fe.grad(c_answer)), test_1_pf) * fe.dx \
        + constant * fe.inner(fe.grad(mu_mid), fe.grad(test_1_pf))*fe.dx

    
    return F1


def Eq2(variables_dict_pf, physical_parameters_dict):
    
    c_answer = variables_dict_pf["c_answer"]
    c_answer_prev = variables_dict_pf["c_answer_prev"]
    mu_answer = variables_dict_pf["mu_answer"]
    mu_answer_prev = variables_dict_pf["mu_answer_prev"]
    test_2_pf = variables_dict_pf["test_2_pf"]
    kappa = physical_parameters_dict["kappa"]
    kesi = physical_parameters_dict["kesi"]

    
    # Assuming f_chem is already defined as a function of c
    # dFdc = derivative_f_chem(c_answer, physical_parameters_dict)

    # Compute the chemical potential df/dc
    # c_answer = fe.variable(c_answer)
    # f = 100*c_answer**2*(1-c_answer)**2
    # dFdc = fe.diff(f, c_answer)
    
    F2 = (

        fe.inner( mu_answer, test_2_pf )
        + fe.inner( - kesi**2 * fe.grad(c_answer), fe.grad(test_2_pf) )
        - fe.inner( c_answer* ( c_answer**2 -1 ), test_2_pf)
        
        )*fe.dx

    return F2


def define_problem_pf(L, variables_dict_pf, physical_parameters_dict):

    solution_vector_pf = variables_dict_pf["solution_vector_pf"]
    abs_tol_pf = physical_parameters_dict["abs_tol_pf"]
    rel_tol_pf = physical_parameters_dict["rel_tol_pf"]
    linear_solver_pf = physical_parameters_dict['linear_solver_pf']
    nonlinear_solver_pf = physical_parameters_dict['nonlinear_solver_pf']
    preconditioner_pf = physical_parameters_dict['preconditioner_pf']
    maximum_iterations_pf = physical_parameters_dict['maximum_iterations_pf']

    J = fe.derivative(L, solution_vector_pf)  # Compute the Jacobian

    # Define the problem
    problem = fe.NonlinearVariationalProblem(L, solution_vector_pf, J=J)



    solver_pf = fe.NonlinearVariationalSolver(problem)

    solver_parameters = {
        'nonlinear_solver': nonlinear_solver_pf,
        'snes_solver': {
            'linear_solver': linear_solver_pf,
            'report': False,
            "preconditioner": preconditioner_pf,
            'error_on_nonconvergence': False,
            'absolute_tolerance': abs_tol_pf,
            'relative_tolerance': rel_tol_pf,
            'maximum_iterations': maximum_iterations_pf,
        }
    }


    solver_pf.parameters.update(solver_parameters)


    return solver_pf


def update_solver_on_new_mesh_pf(mesh, physical_parameters_dict, old_solution_vector_pf= None, old_solution_vector_0_pf=None, 
                                old_solution_vector_0_ns= None, variables_dict_pf= None):
    

    # define the initial condition for the first time step:
    if old_solution_vector_pf is None and old_solution_vector_0_pf is  None and variables_dict_pf is None:
    
        variables_dict_pf = define_variables(mesh)

        solution_vector_pf = variables_dict_pf["solution_vector_pf"]
        solution_vector_pf_0 = variables_dict_pf["solution_vector_pf_0"]
        spaces_pf = variables_dict_pf["spaces_pf"]
        maps_pf = variables_dict_pf["maps_pf"]
        vel_answer_on_pf_mesh = variables_dict_pf["v_answer_on_pf_mesh"]


        # interpolate initial condition  after mesh refinement:
        initial_conditions = InitialConditions_pf(physical_parameters_dict)
        solution_vector_pf_0.interpolate(initial_conditions)
        solution_vector_pf.interpolate(initial_conditions)

        #Initialize velocity function on pf mesh:
        vel_answer_on_pf_mesh.interpolate(fe.Constant((0, 0)))


        # Calculate equation 1 and 2
        eq1 = Eq1(variables_dict_pf, physical_parameters_dict )
        eq2 = Eq2(variables_dict_pf, physical_parameters_dict)

        # Define the combined weak form
        L = eq1 + eq2
        solver_pf = define_problem_pf(L, variables_dict_pf, physical_parameters_dict)

        return_dict = {
            "solver_pf": solver_pf,
            "variables_dict_pf": variables_dict_pf,
            "eq1": eq1,
            "eq2": eq2,
            "L": L,
            "solution_vector_pf": solution_vector_pf,
            "solution_vector_pf_0": solution_vector_pf_0,
            "spaces_pf": spaces_pf,
            "maps_pf": maps_pf,
            "vel_answer_on_pf_mesh":vel_answer_on_pf_mesh,
        }

        return return_dict
    

    # updte velocity on pf mesh :
    if variables_dict_pf is not None:
        

        

        
        solution_vector_pf = variables_dict_pf["solution_vector_pf"]
        solution_vector_pf_0 = variables_dict_pf["solution_vector_pf_0"]
        spaces_pf = variables_dict_pf["spaces_pf"]
        maps_pf = variables_dict_pf["maps_pf"]
        vel_answer_on_pf_mesh = variables_dict_pf["v_answer_on_pf_mesh"]



        # gettting the old solution vector for the Navier-stockes, velocity function on pf mesh:
        u_prev, p_prev= old_solution_vector_0_ns.split(deepcopy=True)
        fe.LagrangeInterpolator.interpolate(vel_answer_on_pf_mesh , u_prev)

        # Calculate equation 1 and 2
        eq1 = Eq1(variables_dict_pf, physical_parameters_dict )
        eq2 = Eq2(variables_dict_pf, physical_parameters_dict)

        # Define the combined weak form
        L = eq1 + eq2
        solver_pf = define_problem_pf(L, variables_dict_pf, physical_parameters_dict)

        return_dict = {
            "solver_pf": solver_pf,
            "variables_dict_pf": variables_dict_pf,
            "eq1": eq1,
            "eq2": eq2,
            "L": L,
            "solution_vector_pf": solution_vector_pf,
            "solution_vector_pf_0": solution_vector_pf_0,
            "spaces_pf": spaces_pf,
            "maps_pf": maps_pf,
            "vel_answer_on_pf_mesh":vel_answer_on_pf_mesh,
        }

        return return_dict
    

    


