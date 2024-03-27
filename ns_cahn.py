import fenics as fe
import numpy as np

Bc = None


class InitialConditions_ns(fe.UserExpression):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  

    def eval(self, values, x):
        values[0] = 0.0  # Initial x-component of velocity
        values[1] = 0.0  # Initial y-component of velocity
        values[2] = 0.0  # Initial pressure

    def value_shape(self):
        return (3,)
    

def define_variables_ns(mesh):

    # Define finite elements for velocity, pressure, and temperature
    P1 = fe.VectorElement("Lagrange", mesh.ufl_cell(), 2)  # Velocity
    P2 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # Pressure 

    phi_func_space = fe.FunctionSpace(mesh, P2)
    phi_prev_interpolated_on_ns_mesh = fe.Function(phi_func_space)

    # Define mixed elements
    element = fe.MixedElement([P1, P2])

    # Create a function space
    function_space_ns = fe.FunctionSpace(mesh, element)

    # Define test functions
    test_1, test_2 = fe.TestFunctions(function_space_ns)

    # Define current and previous solutions
    solution_vector_ns = fe.Function(function_space_ns)  # Current solution
    solution_vector_ns_0 = fe.Function(function_space_ns)  # Previous solution

    # Split functions to access individual components
    u_answer, p_answer = fe.split(solution_vector_ns)  # Current solution
    u_prev, p_prev = fe.split(solution_vector_ns_0)  # Previous solution


    # Collapse function spaces to individual subspaces
    num_subs = function_space_ns.num_sub_spaces()
    spaces_ns, maps = [], []
    for i in range(num_subs):
        space_i, map_i = function_space_ns.sub(i).collapse(collapsed_dofs=True)
        spaces_ns.append(space_i)
        maps.append(map_i)

    return {
        'u_answer': u_answer, 'u_prev': u_prev,
        'p_answer': p_answer, 'p_prev': p_prev,
        'solution_vector_ns': solution_vector_ns, 'solution_vector_ns_0': solution_vector_ns_0,
        'test_1': test_1, 'test_2': test_2,
        'spaces_ns': spaces_ns, 'function_space_ns': function_space_ns,
        "phi_prev_interpolated_on_ns_mesh": phi_prev_interpolated_on_ns_mesh,
        "phi_func_space": phi_func_space
    }

# Related Functions for defining equaions
def epsilon(u):  

    return 0.5 * (fe.grad(u) + fe.grad(u).T)

def sigma(u, p, mu1):

    return 2 * mu1 * epsilon(u) - p * fe.Identity(len(u))


def F1(variables_dict, physical_parameters_dict):

    dt = physical_parameters_dict["dt"]
    test_2 = variables_dict['test_2']
    u_answer = variables_dict['u_answer']

    F1 = fe.inner(fe.div(u_answer), test_2)/dt * fe.dx

    return F1


def F2(variables_dict, physical_parameters_dict):

    alpha = 6* fe.sqrt(2)

    # dt = physical_parameters_dict['dt']
    dt = physical_parameters_dict["dt"]
    rho_air = physical_parameters_dict["rho_air"]
    rho_liq = physical_parameters_dict["rho_liq"]
    mu_air = physical_parameters_dict["mu_air"]
    mu_liq = physical_parameters_dict["mu_liq"]
    gravity = physical_parameters_dict["gravity"]
    kappa = physical_parameters_dict["kappa"]

    u_answer = variables_dict['u_answer']
    u_prev = variables_dict['u_prev']
    p_answer = variables_dict['p_answer']
    test_1 = variables_dict['test_1']
    phi = variables_dict["phi_prev_interpolated_on_ns_mesh"]

    base_1 = (1 - phi)/2
    base_2 = (1 + phi)/2

    mu_mixed = mu_liq + ( mu_air - mu_liq )* base_2



    F2 = (
        fe.inner((u_answer - u_prev) / dt, test_1) * fe.dx
        + fe.inner(fe.dot(u_answer, fe.grad(u_answer)), test_1) * fe.dx
        # + (1/rho_liq) * fe.inner(sigma(u_answer, p_answer, mu_mixed), epsilon(test_1)) * fe.dx
        + (1/rho_liq) * fe.inner(sigma(u_answer, p_answer, mu_mixed), epsilon(test_1)) * fe.dx
        - (1/rho_liq) * fe.inner( gravity*( rho_air- rho_liq) * base_2 , test_1[1]) * fe.dx
    )

    return F2



def define_boundary_condition_ns(variables_dict, physical_parameters_dict) :

    global Bc

    Nx = physical_parameters_dict['Nx']
    Ny = physical_parameters_dict['Ny']
    Domain = physical_parameters_dict['Domain'](Nx, Ny)
    lid_vel_x = physical_parameters_dict['lid_vel_x']
    lid_vel_y = physical_parameters_dict['lid_vel_y']
    W = variables_dict['function_space_ns']
    # Define the Domain boundaries based on the previous setup
    (X0, Y0), (X1, Y1) = Domain

    # Define boundary conditions for velocity, pressure, and temperature
    class LeftBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[0], X0)

    class RightBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[0], X1)

    class BottomBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[1], Y0)

    class TopBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[1], Y1)

    # Instantiate boundary classes
    left_boundary = LeftBoundary()
    right_boundary = RightBoundary()
    bottom_boundary = BottomBoundary()
    top_boundary = TopBoundary()
    # Define Dirichlet boundary conditions
    bc_u_left = fe.DirichletBC(W.sub(0), fe.Constant((0, 0)), left_boundary)
    bc_u_right = fe.DirichletBC(W.sub(0), fe.Constant((0, 0)), right_boundary)
    # bc_u_bottom = fe.DirichletBC(W.sub(0), fe.Constant((0, 0)), bottom_boundary)
    bc_u_top = fe.DirichletBC(W.sub(0), fe.Constant((0, 0)), top_boundary)
    # bc_u_top_x = fe.DirichletBC(W.sub(0).sub(0), fe.Constant(lid_vel_x), top_boundary)
    # bc_u_top_y = fe.DirichletBC(W.sub(0).sub(1), fe.Constant(0), top_boundary)
    bc_u_bottom_x = fe.DirichletBC(W.sub(0).sub(0), fe.Constant(0), bottom_boundary)
    bc_u_bottom_y = fe.DirichletBC(W.sub(0).sub(1), fe.Constant(lid_vel_y), bottom_boundary)



    # bc_presssure_top = fe.DirichletBC(W.sub(1), fe.Constant(0.0), top_boundary)
    bc_presssure_down = fe.DirichletBC(W.sub(1), fe.Constant(0.0), bottom_boundary)

    # Point for setting pressure
    zero_pressure_point = fe.Point( (X0)/2,  (Y1)/2 )
    # bc_p_zero = fe.DirichletBC(W.sub(1), fe.Constant(0.0), lambda x, on_boundary: fe.near(x[0], zero_pressure_point.x()) and fe.near(x[1], zero_pressure_point.y()), method="pointwise")
    # Combine all boundary conditions
    # Bc = [bc_u_left, bc_u_right, bc_u_bottom, bc_u_top, bc_p_zero, bc_u_top_x, bc_u_top_y]

    Bc = [bc_u_left, bc_u_right, bc_u_bottom_x, bc_u_bottom_y , bc_u_top, bc_presssure_down]

    
    

    return  Bc


def define_problem_ns(L, variables_dict, physical_parameters_dict):

    global Bc  


    solution_vector_ns = variables_dict['solution_vector_ns']

    abs_tol_ns = physical_parameters_dict["abs_tol_pf"] # make it ns in fututre!
    rel_tol_ns = physical_parameters_dict["rel_tol_pf"]
    linear_solver_ns = physical_parameters_dict['linear_solver_pf']
    nonlinear_solver_ns = physical_parameters_dict['nonlinear_solver_pf']
    preconditioner_ns = physical_parameters_dict['preconditioner_pf']
    maximum_iterations_ns = physical_parameters_dict['maximum_iterations_pf']

    J = fe.derivative(L, solution_vector_ns)
    problem_ns = fe.NonlinearVariationalProblem(L, solution_vector_ns, Bc, J)
    solver_ns = fe.NonlinearVariationalSolver(problem_ns)

    solver_parameters = {
        'nonlinear_solver': nonlinear_solver_ns,
        'snes_solver': {
            'linear_solver': linear_solver_ns,
            'report': False,
            "preconditioner": preconditioner_ns,
            'error_on_nonconvergence': False,
            'absolute_tolerance': abs_tol_ns,
            'relative_tolerance': rel_tol_ns,
            'maximum_iterations': maximum_iterations_ns,
        }
    }


    solver_ns.parameters.update(solver_parameters)

    return solver_ns


def update_solver_on_new_mesh_ns(mesh, physical_parameters_dict, old_solution_vector_ns= None, old_solution_vector_0_ns=None, 
                                old_solution_vector_0_pf=None, variables_dict= None):
    
    global Bc


    # define the initial condition for the first time step with consdering PF is already defined:
    if old_solution_vector_ns is None and old_solution_vector_0_ns is  None and variables_dict is None :

        variables_dict = define_variables_ns(mesh)


        solution_vector_ns = variables_dict['solution_vector_ns']
        solution_vector_ns_0 = variables_dict['solution_vector_ns_0']
        phi_prev_interpolated_on_ns_mesh = variables_dict['phi_prev_interpolated_on_ns_mesh']
        spaces_ns = variables_dict["spaces_ns"]
        function_space_ns = variables_dict['function_space_ns']


        # interpolate initial condition  after mesh refinement:
        initial_conditions = InitialConditions_ns()
        solution_vector_ns_0.interpolate(initial_conditions)
        solution_vector_ns.interpolate(initial_conditions)

        # gettting the old solution vector for the phase field and mu field on ns mesh:
        phi_prev, u_prev = old_solution_vector_0_pf.split(deepcopy=True)
        fe.LagrangeInterpolator.interpolate(phi_prev_interpolated_on_ns_mesh , phi_prev)

        # define  boundary condition:
        Bc = define_boundary_condition_ns(variables_dict, physical_parameters_dict)


        # define the new forms after mesh refinement:
        f1_form = F1(variables_dict, physical_parameters_dict)
        f2_form = F2(variables_dict, physical_parameters_dict)

        # Define solver: 
        L= f1_form + f2_form
        solver_ns= define_problem_ns(L, variables_dict, physical_parameters_dict)


        return { 
            'solver_ns': solver_ns, 'solution_vector_ns': solution_vector_ns, 'solution_vector_ns_0': solution_vector_ns_0,
            "spaces_ns": spaces_ns, "Bc": Bc, "variables_dict": variables_dict,
            "function_space_ns": function_space_ns, "phi_prev_interpolated_on_ns_mesh":phi_prev_interpolated_on_ns_mesh,
        }


    #updating Phi on NS mesh: 
    if  variables_dict is not None:

     

        solution_vector_ns = variables_dict['solution_vector_ns']
        solution_vector_ns_0 = variables_dict['solution_vector_ns_0']
        phi_prev_interpolated_on_ns_mesh = variables_dict['phi_prev_interpolated_on_ns_mesh']
        spaces_ns = variables_dict["spaces_ns"]
        function_space_ns = variables_dict['function_space_ns']

        # define  boundary condition:
        Bc = define_boundary_condition_ns(variables_dict, physical_parameters_dict)

        # gettting the old solution vector for the phase field and mu field on ns mesh:
        phi_prev, u_prev = old_solution_vector_0_pf.split(deepcopy=True)
        fe.LagrangeInterpolator.interpolate(phi_prev_interpolated_on_ns_mesh , phi_prev)


        # define the new forms after mesh refinement:
        f1_form = F1(variables_dict, physical_parameters_dict)
        f2_form = F2(variables_dict, physical_parameters_dict )


        # Define solver: 
        L= f1_form + f2_form
        solver_ns= define_problem_ns(L, variables_dict, physical_parameters_dict)


        return { 
            'solver_ns': solver_ns, 'solution_vector_ns': solution_vector_ns, 'solution_vector_ns_0': solution_vector_ns_0,
            "spaces_ns": spaces_ns, "Bc": Bc, "variables_dict": variables_dict,
            "function_space_ns": function_space_ns, "phi_prev_interpolated_on_ns_mesh": phi_prev_interpolated_on_ns_mesh,
        }
