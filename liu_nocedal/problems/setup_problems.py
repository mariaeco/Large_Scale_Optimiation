from problems.problems import*
import numpy as np

"""
RESUMO DOS TIPOS DE PROBLEMAS:

QUADRÁTICOS (4 problemas):
- QOR: Quadrático com restrições, convexo
- GOR: Quadrático com restrições, convexo  
- TRIDIAGONAL: Quadrático, convexo, estrutura tridiagonal
- PENALTY: Quadrático com penalidade, não-convexo

NÃO-QUADRÁTICOS (12 problemas):
- ROSENBROCK: Não-quadrático, não-convexo, função de valle
- TRIGONOMETRIC: Não-quadrático, trigonométrico, não-convexo
- EXTENDED_ROSENBROCK: Não-quadrático, não-convexo, extensão do Rosenbrock
- EXTENDED_POWELL: Não-quadrático, não-convexo, singular
- PSP: Quadrático com penalidade não-suave, não-convexo
- ENGGVAL1: Não-quadrático, não-convexo, função de valle
- LINEAR_MINIMUM_SURFACE: Não-quadrático, não-convexo, problema de superfície mínima
- SQUARE_ROOT_1: Não-quadrático, não-convexo, problema de raiz quadrada de matriz
- SQUARE_ROOT_2: Não-quadrático, não-convexo, problema de raiz quadrada de matriz (caso 2)
- FREUDENTHAL_ROTH: Não-quadrático, não-convexo, sistema de equações
- SPARSE_MATRIX_SQRT: Não-quadrático, não-convexo, raiz quadrada de matriz esparsa
- ULTS0: Não-quadrático, não-convexo, EDP não-linear (equação de Laplace)

TOTAL: 16 problemas (4 quadráticos, 12 não-quadráticos)
"""

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
        N_msqrt_sparse = 34  #deve satistazer m = (n + 2) // 3  e n = 3m-2
        #Valores válidos de n: 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100

        problems = {
            'SQUARE_ROOT_1': {  # Não-quadrático, não-convexo, problema de raiz quadrada de matriz
                'objective': msqrtals_objective,
                'setup': lambda: (N_msqrt, msqrtals_setup(N_msqrt)),
                'bounds': None,
                'args': None
            },
            'SQUARE_ROOT_2': {  # Não-quadrático, não-convexo, problema de raiz quadrada de matriz (caso 2)
                'objective': msqrtbls_objective,
                'setup': lambda: (N_msqrt, msqrtbls_setup(N_msqrt)),
                'bounds': None,
                'args': None
            },
            'FREUDENTHAL_ROTH': {  # Não-quadrático, não-convexo, sistema de equações
                'objective': freuroth_objective,
                'setup': lambda: (N3, x03),
                'bounds': None,
                'args': None
            },
            'SPARSE_MATRIX_SQRT': {  # Não-quadrático, não-convexo, raiz quadrada de matriz esparsa
                'objective': spmsqrt_objective,
                'setup': lambda: (N_msqrt_sparse, spmsqrt_setup(N_msqrt_sparse)),
                'bounds': None,
                'args': None
            },
            'ROSENBROCK': {  # Não-quadrático, não-convexo, função de valle
                'objective': rosenbrock_objective,
                'setup': lambda: (N, x0), #N variaveis -- x0
                'bounds': None,
                'args': None
            },
            'PENALTY': {  # Quadrático com penalidade, não-convexo
                'objective': penalty_objective,
                'setup': lambda: (N2, x02), #N variaveis -- x0
                'bounds': None,
                'args': None
            },
            'TRIGONOMETRIC': {  # Não-quadrático, trigonométrico, não-convexo
                'objective': trigonometric_ext_objective,
                'setup': lambda: (N2, np.ones(1) / N2),  # x0 = [1/n, ..., 1/n]
                'bounds': None,
                'args': None
            },
            'EXTENDED_ROSENBROCK': {  # Não-quadrático, não-convexo, extensão do Rosenbrock
                'objective': rosenbrock_ext_objective,
                'setup': lambda: (N, np.zeros(N)),  # n deve ser par
                'bounds': None,
                'args': None
            },
            'EXTENDED_POWELL': {  # Não-quadrático, não-convexo, singular
                'objective': powell_singular_ext_objective_wrapper,
                'setup': lambda: (N2, x02),  # n deve ser múltiplo de 4
                'bounds': None,
                'args': None
            },
            'QOR': {  # Quadrático com restrições, convexo
                'objective': qor_objective,
                'setup': lambda: (N_matrix, np.ones(N_matrix)),  # 50 variáveis, ponto inicial zeros
                'bounds': None,
                'args': None
            },
            'GOR': {  # Quadrático com restrições, convexo
                'objective': gor_objective,
                'setup': lambda: (N_matrix, np.ones(N_matrix)),  # 50 variáveis, ponto inicial zeros
                'bounds': None,
                'args': None
            },
            'PSP': {  # Quadrático com penalidade não-suave, não-convexo
                'objective': psp_objective,
                'setup': lambda: (N_matrix, np.ones(N_matrix)),  # 50 variáveis, ponto inicial zeros
                'bounds': None,
                'args': None
            },
            'TRIDIAGONAL': {  # Quadrático, convexo, estrutura tridiagonal
                'objective': tridia_objective,
                'setup': lambda: (N, np.zeros(N)),
                'bounds': None,
                'args': None
            },
            'ENGGVAL1': {  # Não-quadrático, não-convexo, função de valle
                'objective': engval1_objective,
                'setup': lambda: (N, 2 * x0),
                'bounds': None,
                'args': None
            },
            'LINEAR_MINIMUM_SURFACE': {  # Não-quadrático, não-convexo, problema de superfície mínima
                'objective': lminsurf_objective,
                'setup': lambda: (N_lminsurf, x0_lminsurf),  # n=9 (3x3 grid)
                'bounds': None,
                'args': None
            },
            'ULTS0': {  # Não-quadrático, não-convexo, EDP não-linear (equação de Laplace)
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
 