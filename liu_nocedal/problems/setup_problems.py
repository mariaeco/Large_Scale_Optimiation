from problems.problems import*
import numpy as np



def setup_problems():
        """
        Configura os problemas de otimização com seus parâmetros específicos.
        """

        #N var e start values -----------
        #Freudenthal Roth
        N3 = 100
        x03 = np.zeros(N3)
        
        #Rosenbrock, Extended Rosenbrock Freudenthal Roth
        N = 100
        x0 = np.zeros(N)

        #Penalty. Trigonometric, Extended Powell
        N2 = 100
        x02 = np.zeros(N2)  

        #QOR, GOR, PSP
        N_matrix = 50

        #Linear Minimum Surface
        N_lminsurf = 36  # Deve ser um quadrado perfeito
        x0_lminsurf = lminsurf_setup(N_lminsurf)[0]
        
        #Square Root 1 and 2
        N_msqrt = 36 # Deve ser um quadrado perfeito

        #Sparse Matrix Square Root
        N_msqrt_sparse = 16  #deve satistazer m = (n + 2) // 3  e n = 3m-2
        #Valores válidos de n: 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100

        problems = {
            'ROSENBROCK': {
                'objective': rosenbrock_objective,
                'setup': lambda: (N, x0), #N variaveis -- x0
                'bounds': None,
                'args': None
            },
            'PENALTY': {
                'objective': penalty_objective,
                'setup': lambda: (N2, x02), #N variaveis -- x0
                'bounds': None,
                'args': None
            },
            'TRIGONOMETRIC': {
                'objective': trigonometric_ext_objective,
                'setup': lambda: (N2, np.ones(1) / N2),  # x0 = [1/n, ..., 1/n]
                'bounds': None,
                'args': None
            },
            'EXTENDED_ROSENBROCK': {
                'objective': rosenbrock_ext_objective,
                'setup': lambda: (N, x0),  # n deve ser par
                'bounds': None,
                'args': None
            },
            'EXTENDED_POWELL': {
                'objective': powell_singular_ext_objective_wrapper,
                'setup': lambda: (N2, x02),  # n deve ser múltiplo de 4
                'bounds': None,
                'args': None
            },
            'QOR': {
                'objective': qor_objective,
                'setup': lambda: (N_matrix, np.zeros(N_matrix)),  # 50 variáveis, ponto inicial zeros
                'bounds': None,
                'args': None
            },
            'GOR': {
                'objective': gor_objective,
                'setup': lambda: (N_matrix, np.zeros(N_matrix)),  # 50 variáveis, ponto inicial zeros
                'bounds': None,
                'args': None
            },
            'PSP': {
                'objective': psp_objective,
                'setup': lambda: (N_matrix, np.zeros(N_matrix)),  # 50 variáveis, ponto inicial zeros
                'bounds': None,
                'args': None
            },
            'TRIDIAGONAL': {
                'objective': tridia_objective,
                'setup': lambda: (N, np.ones(N)),
                'bounds': None,
                'args': None
            },
            'ENGGVAL1': {
                'objective': engval1_objective,
                'setup': lambda: (N, 2 * x0),
                'bounds': None,
                'args': None
            },
            'LINEAR_MINIMUM_SURFACE': {
                'objective': lminsurf_objective,
                'setup': lambda: (N_lminsurf, x0_lminsurf),  # n=9 (3x3 grid)
                'bounds': None,
                'args': None
            },
            'SQUARE_ROOT_1': {
                'objective': msqrtals_objective,
                'setup': lambda: (N_msqrt, msqrtals_setup(N_msqrt)),
                'bounds': None,
                'args': None
            },
            'SQUARE_ROOT_2': {
                'objective': msqrtbls_objective,
                'setup': lambda: (N_msqrt, msqrtbls_setup(N_msqrt)),
                'bounds': None,
                'args': None
            },
            'FREUDENTHAL_ROTH': {
                'objective': freuroth_objective,
                'setup': lambda: (N3, x03),
                'bounds': None,
                'args': None
            },
            'SPARSE_MATRIX_SQRT': {
                'objective': spmsqrt_objective,
                'setup': lambda: (N_msqrt_sparse, spmsqrt_setup(N_msqrt_sparse)),
                'bounds': None,
                'args': None
            },
            'ULTS0': {
                'objective': ults0_objective,
                'setup': lambda: setup_ults0(),
                'bounds': [(-8, 8) for _ in range(64)],  # 8x8 grid = 64 variáveis
                'args': None
            }
        }


        return problems
    
def setup_ults0():
    """
    Configuração específica para o problema ULTS0.
    """
    N = 8  # Grade 8x8
    h = 1.0 / (N - 1)
    grid_shape = (N, N)
    
    # Ponto inicial
    np.random.seed(42)
    initial_phi_guess = 0.1 * np.random.randn(N * N)
    
    return (64, initial_phi_guess, grid_shape, h)
 