# Funções objetivo
from typing import List, Sequence
import math
import numpy as np



# - Problema 0: ROSENBROCK -------------------------------------------------------------------------
def rosenbrock(x):
    return (x[0] - 1)**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_objective(x):
    return rosenbrock(x)

# - Problema 1: PENALTY -------------------------------------------------------------------------
def Penalti(x, a, b):
    # Somatório de (xi - 1)^2 para i = 1...n
    term1 = a * sum((xi - 1)**2 for xi in x)

    # Somatório de xi^2
    sum_x2 = sum(xi**2 for xi in x)

    # Aplicando toda a expressão
    term2 = b * (sum_x2 - 0.25)**2

    return term1 + term2

def penalty_objective(x):
    """Wrapper para a função Penalty com parâmetros padrão"""
    return Penalti(x, a=1.0, b=1e-3)

# - Problema 2: TRIGONOMETRIC -------------------------------------------------------------------------
def trigonometric(x):
    """
    Função trigonométrica vetorial.
    x: lista ou vetor de tamanho n
    retorna: lista com f(x)
    """
    n = len(x)
    soma = sum(math.cos(xj) for xj in x)
    f = []
    for i in range(n):
        fi = n - soma + (i+1)*(1 - math.cos(x[i])) - math.sin(x[i])
        f.append(fi)
    return np.array(f)

# x0 = [1/n, ..., 1/n]
# Mínimo = f=0

# Definindo a funçao objetivo (o L-BFGS-B exige que seja uma funçao escalar, e a trigonometrica retorna um vetor)
def trigonometric_ext_objective(x):
  fx = trigonometric(x)
  return np.dot(fx, fx) # ||f(x)||²



# - Problema 3: EXTENDED ROSENBROCK -------------------------------------------------------------------------

def rosenbrock_ext_residuals(x: List[float]) -> List[float]:
    n = len(x)
    assert n % 2 == 0, "n deve ser par" # Para armezenar os residuos da funçao em pares, é necessário que seja par
    m = n // 2
    r = [0.0] * n
    for k in range(m):
        i = 2*k      # índice 0-based de x_{2k-1}
        r[2*k]   = 10.0 * (x[i+1] - x[i]**2)   # r_{2k-1}
        r[2*k+1] = 1.0 - x[i]                  # r_{2k}
    return np.array(r)

def rosenbrock_ext_objective(x: List[float]) -> float:
    r = rosenbrock_ext_residuals(x)
    return np.dot(r,r) # ||r||²

# Minimo = f=0 na origem

# - Problema 4: EXTENDED POWELL -------------------------------------------------------------------------
def powell_singular_residuals(x: List[float]) -> List[float]:
    """
    Calcula os resíduos da Extended Powell Singular Function.
    x: lista de tamanho n (deve ser múltiplo de 4)
    retorna: lista de resíduos r_j
    """
    n = len(x)
    assert n % 4 == 0, "O tamanho de x deve ser múltiplo de 4"
    m = n // 4
    r = [0.0] * n

    for i in range(m):
        # índices do bloco (0-based)
        i1 = 4*i
        i2 = i1 + 1
        i3 = i1 + 2
        i4 = i1 + 3

        r[i1] = x[i1] + 10 * x[i2]                 # f_{4i-3}
        r[i2] = (5**0.5) * (x[i3] - x[i4])         # f_{4i-2}
        r[i3] = (x[i2] - 2*x[i3])**2               # f_{4i-1}
        r[i4] = (10**0.5) * (x[i1] - x[i4])**2     # f_{4i}

    return np.array(r)

def powell_singular_ext_objective(x: List[float]) -> float:
    """
    Função objetivo: soma dos quadrados dos resíduos.
    """
    r = powell_singular_residuals(x)
    return np.dot(r,r) # ||r||²

# Minimo = f=0 na origem

def powell_singular_ext_objective_wrapper(x):
    """Wrapper para Extended Powell com verificação de tamanho"""
    n = len(x)
    if n % 4 != 0:
        # Ajustar para o múltiplo de 4 mais próximo
        n = (n // 4) * 4
        if n == 0:
            n = 4
        x = x[:n]
    return powell_singular_ext_objective(x)


# instancias para os problemas QOR, GOR E PSP

a = [
    1.25, 1.40, 2.40, 1.40, 1.75, 1.20, 2.25, 1.20, 1.00, 1.10,
    1.50, 1.60, 1.25, 1.25, 1.20, 1.20, 1.40, 0.50, 0.50, 1.25,
    1.80, 0.75, 1.25, 1.40, 1.60, 2.00, 1.00, 1.60, 1.25, 2.75,
    1.25, 1.25, 1.25, 3.00, 1.50, 2.00, 1.25, 1.40, 1.80, 1.50,
    2.20, 1.40, 1.50, 1.25, 2.00, 1.50, 1.25, 1.40, 0.60, 1.50
]

B = [
    1.0, 1.5, 1.0, 0.1, 1.5, 2.0, 1.0, 1.5, 3.0, 2.0,
    1.0, 3.0, 0.1, 1.5, 0.15, 2.0, 1.0, 0.1, 3.0, 0.1,
    1.2, 1.0, 0.1, 2.0, 1.2, 3.0, 1.5, 3.0, 2.0, 1.0,
    1.2, 2.0, 1.0
]

d = [
    5.0, 5.0, 5.0, 2.5, 6.0, 6.0, 5.0, 6.0, 10.0, 6.0,
    5.0, 9.0, 2.0, 7.0, 2.5, 6.0, 5.0, 2.0, 9.0, 2.0,
    5.0, 5.0, 2.5, 5.0, 6.0, 10.0, 7.0, 10.0, 6.0, 5.0,
    4.0, 4.0, 4.0
]

A_sets = [
    [31], [1], [2], [4], [6], [8], [10], [12], [11,13,14], [16],
    [9,18], [5,20,21], [19], [23], [7,25], [28], [29], [32], [3,33], [35],
    [36], [30,37], [38,39], [40], [41], [44], [46], [42,45,48,50], [26,34,43], [15,17,24,47],
    [49], [22], [27]
]

B_sets = [
    [1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15], [16,17], [18,19],
     [20], [], [22, 23, 24], [25,26], [27,28], [29,30], [31,32], [33,34], [35], [21,36],
    [37,38], [39], [40], [41,42], [43,44,50], [45,46,47], [48], [49], [], [],
    [], [], []

]


# Convertendo os conjuntos para indices 0-based. Dessa forma, o numero [50] agora é [49] (estava tendo problemas de indices)
A_sets = [[j-1 for j in s] for s in A_sets]
B_sets = [[j-1 for j in s] for s in B_sets]


# - Problema 5: QOR -------------------------------------------------------------------------
from typing import List, Sequence

def f_qor(
    x: Sequence[float],
    a: Sequence[float],
    B: Sequence[float],
    d: Sequence[float],
    A_sets: List[List[int]],
    B_sets: List[List[int]],
) -> float:
    """
    x: vetor de variáveis (índices 0-based)
    a: coeficientes para i=0..49  (50 itens)
    B, d: coeficientes para i=0..32 (33 itens)
    A_sets[i], B_sets[i]: listas de índices j (0-based) usados no i-ésimo termo da penalidade
    """
    # 1) sum_{i=1..50} a_i * x_i^2   -> i = 0..49 em Python
    term1 = sum(a[i] * (x[i] ** 2) for i in range(50))

    # 2) sum_{i=1..33} B_i * [ d_i - sum_{j in A(i)} x_j + sum_{j in B(i)} x_j ]^2
    term2 = 0.0
    for i in range(33):
        inner = d[i] \
              - sum(x[j] for j in A_sets[i]) \
              + sum(x[j] for j in B_sets[i])
        term2 += B[i] * (inner ** 2)

    return term1 + term2


# O ponto inicial é o vetor de zeros

def qor_objective(x):
    """Wrapper para QOR com parâmetros padrão"""
    return f_qor(x, a, B, d, A_sets, B_sets)

# - Problema 6: GOR -------------------------------------------------------------------------
def c_i(x_i, a_i):
    if x_i >= 0:
        return a_i * x_i * math.log(1 + x_i)
    else:
        return -a_i * x_i * math.log(1 - x_i)

def b_i(y_i, B_i):
    if y_i >= 0:
        return B_i * y_i**2 * math.log(1 + y_i)
    else:
        return B_i * y_i**2

def f_gor(x, a, B, d, A_sets, B_sets):
    term1 = sum(c_i(x[i], a[i]) for i in range(50))
    term2 = 0.0
    for i in range(33):
        y_i = d[i] - sum(x[j-1] for j in A_sets[i]) + sum(x[j-1] for j in B_sets[i])
        term2 += b_i(y_i, B[i])
    return term1 + term2

def gor_objective(x):
    """Wrapper para GOR com parâmetros padrão"""
    return f_gor(x, a, B, d, A_sets, B_sets)

# O ponto inicial é o vetor de zeros

# - Problema 7: PSP -------------------------------------------------------------------------

def h_i(y_i):
    if y_i >= 0.1:
        return 1.0 / y_i
    else:
        return 100.0 * (0.1 - y_i) + 10.0

def f_psp(x, a, B, d, A_sets, B_sets):
    # Termo 1: soma dos a_i * (x_i - 5)^2 para i = 0..49 ( Python 0-based )
    term1 = sum(a[i] * (x[i] - 5.0) ** 2 for i in range(50))

    # Termo 2: soma dos B_i * h_i(y_i) para i = 0..32
    term2 = 0.0
    for i in range(33):
        y_i = d[i] \
              - sum(x[j - 1] for j in A_sets[i]) \
              + sum(x[j - 1] for j in B_sets[i])
        term2 += B[i] * h_i(y_i)

    return term1 + term2

def psp_objective(x):
    """Wrapper para PSP com parâmetros padrão"""
    return f_psp(x, a, B, d, A_sets, B_sets)

# O ponto inicial é o vetor de zeros


# - Problema 9: TRIDIAGONAL -------------------------------------------------------------------------
def tridia_objective(x):

    n = len(x)
    
    # Primeiro termo: (x1 - 1)^2
    f = (x[0] - 1)**2
    
    # Termos subsequentes: (2*xi - x_{i-1})^2 para i = 2...n
    for i in range(1, n):
        f += (2*x[i] - x[i-1])**2
    
    return f


# Problema 10: LINEAR MINIMUM SURFACE ----------------------------------------------------------------
def lminsurf_objective(x):
    #x: vetor de variáveis (deve ser um quadrado perfeito, n = p^2)
        
    n = len(x)
    p = int(math.sqrt(n))
    
    if p*p != n:
        raise ValueError(f"n = {n} deve ser um quadrado perfeito")
    
    if n < 9:
        raise ValueError(f"n = {n} deve ser pelo menos 9")
    
    # Número de elementos
    nel = (p-1)**2
    
    f = 0
     # Para cada elemento (i,j) 
    for i in range(p-1):
        for j in range(p-1):
            # Índices dos 4 vértices do elemento
            idx1 = i*p + j           # (i,j)
            idx2 = i*p + (j+1)       # (i,j+1)
            idx3 = (i+1)*p + j       # (i+1,j)
            idx4 = (i+1)*p + (j+1)   # (i+1,j+1)
            
            # Cálculo da área do elemento
            a = x[idx1] - x[idx4]  # x(i,j) - x(i+1,j+1)
            b = x[idx3] - x[idx2]  # x(i+1,j) - x(i,j+1)
            
            ri = 1 + 0.5 * nel * (a**2 + b**2)
            f += math.sqrt(ri) / nel
    
    return f


def lminsurf_setup(n):

    p = int(math.sqrt(n))
    
    if p*p != n:
        raise ValueError(f"n = {n} deve ser um quadrado perfeito")
    
    if n < 9:
        raise ValueError(f"n = {n} deve ser pelo menos 9")
    
    h = 1.0 / (p-1)
    x0 = np.zeros(n)
    xlower = -np.inf * np.ones(n)
    xupper = np.inf * np.ones(n)
    
    # Configurar condições de contorno
    for iy in range(p):
        if iy == 0:  # Borda inferior
            for ix in range(p):
                t = ix * h
                x0[ix] = 1 + 8*t
                xlower[ix] = x0[ix]
                xupper[ix] = x0[ix]
        elif iy == p-1:  # Borda superior
            for ix in range(p):
                idx = ix + (p-1)*p
                t = ix * h
                x0[idx] = 5 + 8*t
                xlower[idx] = x0[idx]
                xupper[idx] = x0[idx]
        else:  # Bordas laterais
            # Borda esquerda
            idx = iy*p
            t = iy * h
            x0[idx] = 1 + 4*t
            xlower[idx] = x0[idx]
            xupper[idx] = x0[idx]
            
            # Borda direita
            idx = (iy+1)*p - 1
            x0[idx] = 9 + 4*t
            xlower[idx] = x0[idx]
            xupper[idx] = x0[idx]
    
    return x0, xlower, xupper


# - Problema 11: ENNGVAL1 ----------------------------------------------------------------
def engval1_objective(x):

    n = len(x)
    f = 0
    
    # Para cada par consecutivo (xi, xi+1)
    for i in range(n-1):
        t = x[i]**2 + x[i+1]**2
        f += t**2 - 4*x[i] + 3
    
    return f
    


# - Problema 12: SQUARE ROOT 1  ----------------------------------------------------------------
def msqrtals_objective(x):

    n = len(x)
    m = int(math.sqrt(n))
    
    if m*m != n:
        raise ValueError(f"n = {n} deve ser um quadrado perfeito")
    
    if n < 4:
        raise ValueError(f"n = {n} deve ser pelo menos 4")
    
    # Construir matriz B
    b = np.sin(np.arange(1, n+1)**2)
    B = b.reshape(m, m).T
    
    # Construir matriz A = B*B
    A = B @ B
    
    # Reshape x para matriz X
    X = x.reshape(m, m)
    
    # Calcular resíduos: A - X*X
    residual = A - X @ X
    
    # Soma dos quadrados dos resíduos
    f = np.sum(residual**2)
    
    return f

def msqrtals_setup(n=16):
    #n: número de variáveis (deve ser um quadrado perfeito, padrão 16)
    
    m = int(math.sqrt(n))
    
    if m*m != n:
        raise ValueError(f"n = {n} deve ser um quadrado perfeito")
    
    if n < 4:
        raise ValueError(f"n = {n} deve ser pelo menos 4")
    
    # Ponto inicial: 0.2*sin((1:n).^2)'
    x0 = 0.2 * np.sin(np.arange(1, n+1)**2)
    
    return x0



# - Problema 13: SQUARE ROOT  2 ----------------------------------------------------------------
def msqrtbls_objective(x):
    n = len(x)
    m = int(math.sqrt(n))
    
    if m*m != n:
        raise ValueError(f"n = {n} deve ser um quadrado perfeito")
    
    if n < 4:
        raise ValueError(f"n = {n} deve ser pelo menos 4")
    
    # Construir matriz B (Case 1: b(2*m+1) = 0)
    b = np.sin(np.arange(1, n+1)**2)
    b[2*m] = 0  # Defines Case 1
    B = b.reshape(m, m).T
    
    # Construir matriz A = B*B
    A = B @ B
    
    # Reshape x para matriz X
    X = x.reshape(m, m)
    
    # Calcular resíduos: A - X*X
    residual = A - X @ X
    
    # Soma dos quadrados dos resíduos
    f = np.sum(residual**2)
    
    return f

def msqrtbls_setup(n=16):

    m = int(math.sqrt(n))
    
    if m*m != n:
        raise ValueError(f"n = {n} deve ser um quadrado perfeito")
    
    if n < 4:
        raise ValueError(f"n = {n} deve ser pelo menos 4")
    
    # Ponto inicial: 0.2*sin((1:n).^2)'
    x0 = 0.2 * np.sin(np.arange(1, n+1)**2)
    
    return x0


# Problema 14: Extended Freudenthal and Roth ----------------------------------------------------------------

def freuroth_objective(x):

    n = len(x)
    f = 0
    
    # Para cada par consecutivo (xi, xi+1)
    for i in range(n-1):
        r1 = x[i] - 13 + 5*x[i+1]**2 - x[i+1]**3 - 2*x[i+1]
        r2 = x[i] - 29 + x[i+1]**3 + x[i+1]**2 - 14*x[i+1]
        f += r1**2 + r2**2
    
    return f


# Problema 15: Sparse Matrix Square Root ----------------------------------------------------------------
def spmsqrt_objective(x):

    n = len(x)
    m = (n + 2) // 3
    
    if 3*m - 2 != n:
        raise ValueError(f"n = {n} deve satisfazer n = 3m-2")
    
    if n < 4:
        raise ValueError(f"n = {n} deve ser pelo menos 4")
    
    # Construir matriz B tridiagonal
    B = np.zeros((m, m))
    b = np.sin(np.arange(1, 3*m-1)**2)
    
    ib = 0
    for j in range(m):
        if j == 0:
            B[0:2, 0] = b[0:2]
            ib = 2
        elif j == m-1:
            B[m-2:m, m-1] = b[ib:ib+2]
        else:
            B[j-1:j+2, j] = b[ib:ib+3]
            ib += 3
    
    # Construir matriz A = B*B
    A = B @ B
    
    # Extrair elementos não-zero de A (estrutura tridiagonal)
    a = []
    for j in range(m):
        start = max(0, j-2)
        end = min(m, j+3)
        a.extend(A[start:end, j])
    
    # Construir matriz X tridiagonal a partir de x
    X = np.zeros((m, m))
    ib = 0
    for j in range(m):
        if j == 0:
            X[0:2, 0] = x[0:2]
            ib = 2
        elif j == m-1:
            X[m-2:m, m-1] = x[n-2:n]
        else:
            X[j-1:j+2, j] = x[ib:ib+3]
            ib += 3
    
    # Calcular resíduos: A - X*X
    residual = A - X @ X
    
    # Extrair elementos não-zero do resíduo (estrutura tridiagonal)
    residual_elements = []
    for j in range(m):
        start = max(0, j-2)
        end = min(m, j+3)
        residual_elements.extend(residual[start:end, j])
    
    # Soma dos quadrados dos resíduos
    f = np.sum(np.array(residual_elements)**2)
    
    return f

def spmsqrt_setup(n):

    m = (n + 2) // 3
    
    if 3*m - 2 != n:
        raise ValueError(f"n = {n} deve satisfazer n = 3m-2")
    
    if n < 4:
        raise ValueError(f"n = {n} deve ser pelo menos 4")
    
    # Ponto inicial: 0.2*sin((1:n).^2)'
    x0 = 0.2 * np.sin(np.arange(1, n+1)**2)
    
    return x0


# - Problema 16: Ultsoc ----------------------------------------------------------------

def rho(magnitude_of_velocity):
    """
    Função rho com proteção contra valores que causam raiz quadrada negativa.
    """
    # Limitar a magnitude da velocidade para evitar problemas numéricos
    magnitude_of_velocity = np.clip(magnitude_of_velocity, 0, 0.99)
    
    denominator = 1.0 - magnitude_of_velocity**2
    return 1.0 / np.sqrt(denominator)

def compute_divergence(phi_values, h):
    """
    Calcula a divergência do termo 'div[rho * nabla(phi)]'
    usando diferenças finitas.
    """
    # Assumimos que phi_values é uma grade 2D (NxN)
    
    # 1. Calcule o gradiente de phi
    grad_phi_x = np.gradient(phi_values, h, axis=0)
    grad_phi_y = np.gradient(phi_values, h, axis=1)
    
    # 2. Calcule a magnitude do gradiente
    magnitude = np.sqrt(grad_phi_x**2 + grad_phi_y**2)
    
    # 3. Calcule o rho
    density = rho(magnitude)
    
    # 4. Calcule o vetor de fluxo 'rho * nabla(phi)'
    flux_x = density * grad_phi_x
    flux_y = density * grad_phi_y
    
    # 5. Calcule a divergência do fluxo
    div_flux_x = np.gradient(flux_x, h, axis=0)
    div_flux_y = np.gradient(flux_y, h, axis=1)
    
    divergence = div_flux_x + div_flux_y
    
    return divergence

def ults0_objective(phi_1d, grid_shape, h):

    try:
        # Remodelar o array 1D para a grade 2D
        phi_values = phi_1d.reshape(grid_shape)
        
        # Calcule o lado esquerdo da equação (3.1)
        left_side = compute_divergence(phi_values, h)
        
        # Verificar se há valores NaN ou infinitos
        if np.any(np.isnan(left_side)) or np.any(np.isinf(left_side)):
            return 1e10  # Retornar um valor alto se houver problemas numéricos
        
        # Retorne a soma dos quadrados dos resíduos
        return np.sum(left_side**2)
    
    except Exception as e:
        print(f"Erro na função objetivo: {e}")
        return 1e10