# Descida por Coordenadas Otimizada

Este arquivo implementa o algoritmo de **Descida por Coordenadas** otimizado para problemas de otimização não-linear de larga escala, baseado na coleção de problemas de Liu e Nocedal (1989).

## 📋 **Formulação Matemática**

### **Problema de Otimização**
Considere o problema irrestrito:
```
min f(x)
x ∈ ℝⁿ
```

onde `f: ℝⁿ → ℝ` é uma função diferenciável.

### **Algoritmo de Descida por Coordenadas**

O algoritmo resolve iterativamente subproblemas de dimensão reduzida:

```
Para k = 0, 1, 2, ...:
  1. Selecionar bloco de coordenadas Iₖ ⊆ {1, 2, ..., n}
  2. Resolver: x^{k+1} = argmin_{x_{Iₖ}} f(x_{Iₖ}, x^k_{¬Iₖ})
  3. Verificar convergência: ||x^{k+1} - x^k|| < ε
```

### **Seleção de Coordenadas**
- **Estratégia**: Seleção aleatória de blocos
- **Tamanho do bloco**: Configurável (padrão: 10)
- **Garantia**: Todas as coordenadas são eventualmente selecionadas

## 🔧 **Funções Implementadas**

### **1. `calcular_gradiente(f, x)`**
```python
def calcular_gradiente(f, x):
    """
    Calcula gradiente numericamente usando diferenças finitas centrais
    
    Args:
        f: função objetivo
        x: ponto atual
    
    Returns:
        grad: vetor gradiente
    """
```
**Fórmula**: `∇f_i ≈ [f(x + h·e_i) - f(x - h·e_i)] / (2h)`

### **2. `descida_por_coordenadas_otimizada(f, x0, max_iter, tol, block_size=10)`**
```python
def descida_por_coordenadas_otimizada(f, x0, max_iter, tol, block_size=10):
    """
    Algoritmo principal de descida por coordenadas
    
    Args:
        f: função objetivo
        x0: ponto inicial
        max_iter: máximo de iterações
        tol: tolerância para convergência
        block_size: tamanho do bloco de coordenadas
    
    Returns:
        x: ponto ótimo
        fo: valor da função objetivo
        k: número de iterações
    """
```

**Características**:
- Seleção aleatória de blocos de coordenadas
- Garantia de cobertura completa das coordenadas
- Critério de convergência baseado na mudança do ponto

### **3. `minimizar_bloco_coordenadas(f, x, indices)`**
```python
def minimizar_bloco_coordenadas(f, x, indices):
    """
    Minimiza f(x) em relação a um bloco de coordenadas
    
    Args:
        f: função objetivo
        x: ponto atual
        indices: lista de índices das coordenadas
    
    Returns:
        x: ponto atualizado
    """
```

**Algoritmo**:
```
Se |indices| = 1:
    x[i] = minimizar_coordenada(f, x, i)
Senão se |indices| ≤ 3:
    x[indices] = minimize(f_bloco, x0_bloco, method='Nelder-Mead')
Senão:
    Para cada i em indices:
        x[i] = minimizar_coordenada(f, x, i)
```

### **5. `minimizar_coordenada(f, x, i)`**
```python
def minimizar_coordenada(f, x, i):
    """
    Minimiza f(x) em relação à coordenada i
    
    Args:
        f: função objetivo
        x: ponto atual
        i: índice da coordenada
    
    Returns:
        x_i: valor otimizado da coordenada i
    """
```

**Implementação**:
- Define função 1D: `g(x_i) = f(x_1, ..., x_i, ..., x_n)`
- Usa `minimize_1d()` com método de Brent

### **6. `minimize_1d(f, x0)`**
```python
def minimize_1d(f, x0):
    """
    Minimização 1D usando método de Brent
    
    Args:
        f: função 1D
        x0: ponto inicial
    
    Returns:
        resultado: objeto com atributo .x
    """
```

**Método**: `scipy.optimize.minimize_scalar(method='brent')`

## 🏗️ **Classe Principal**

### **`CoordinateDescentSolver`**

```python
class CoordinateDescentSolver:
    def __init__(self, block_size=10):
        """
        Inicializa o solver
        
        Args:
            block_size: tamanho do bloco de coordenadas
        """
    
    def solve_problem(self, problem_name, max_iter=1000, tol=1e-6):
        """
        Resolve um problema específico
        
        Returns:
            dict: resultados da otimização
        """
    
    def solve_all_problems(self, max_iter=1000, tol=1e-6):
        """
        Resolve todos os problemas configurados
        """
    
    def print_summary(self):
        """
        Imprime resumo dos resultados
        """
```

## 📊 **Estrutura dos Resultados**

Cada resultado contém:
```python
{
    'problem': str,           # Nome do problema
    'success': bool,          # Sucesso da otimização
    'iterations': int,        # Número de iterações
    'function_value': float,  # Valor mínimo encontrado
    'x_value': array,         # Ponto ótimo
    'gradient_norm': float,   # Norma do gradiente final
    'message': str,           # Mensagem de status
    'execution_time': float,  # Tempo de execução
    'n_variables': int        # Número de variáveis
}
```

## 🚀 **Como Usar**

### **Exemplo Básico**
```python
from descent_coordinate_algorithm import CoordinateDescentSolver

# Criar solver
solver = CoordinateDescentSolver(block_size=10)

# Resolver um problema específico
result = solver.solve_problem('ROSENBROCK', max_iter=1000, tol=1e-6)

# Resolver todos os problemas
solver.solve_all_problems()
solver.print_summary()
```

### **Execução Completa**
```bash
cd liu_nocedal
python descent_coordinate_algorithm.py
```

## ⚙️ **Parâmetros Configuráveis**

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `block_size` | 10 | Tamanho do bloco de coordenadas |
| `max_iter` | 1000 | Máximo de iterações |
| `tol` | 1e-6 | Tolerância para convergência |

## 📈 **Vantagens do Algoritmo**

1. **Eficiência**: Minimização em subespaços de dimensão reduzida
2. **Paralelização**: Coordenadas independentes podem ser processadas em paralelo
3. **Memória**: Baixo uso de memória comparado a métodos de segunda ordem
4. **Robustez**: Funciona bem em problemas mal condicionados
5. **Flexibilidade**: Tamanho de bloco adaptável

## 🔍 **Complexidade Computacional**

- **Por iteração**: O(b·n) onde b é o tamanho do bloco
- **Total**: O(K·b·n) onde K é o número de iterações
- **Memória**: O(n) - apenas o vetor de variáveis

## 📋 **Problemas Suportados**

O algoritmo resolve todos os problemas da coleção Liu e Nocedal:
- Rosenbrock
- Extended Rosenbrock
- Extended Powell
- Freudenthal-Roth
- Engvall
- Trigonometric
- Penalty
- E outros...

## 📄 **Saídas Geradas**

- **Tabela resumida**: Estatísticas gerais dos problemas
- **Tabela detalhada**: Valores das variáveis em múltiplas colunas
- **PDFs**: Gerados automaticamente via LaTeX
- **Logs**: Progresso da otimização no console
