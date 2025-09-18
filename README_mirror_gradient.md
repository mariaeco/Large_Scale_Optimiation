# Gradiente Espelhado com Divergências de Bregman

Este arquivo implementa o algoritmo de **Gradiente Espelhado** (Mirror Descent) com divergências de Bregman para problemas de otimização não-linear de larga escala, baseado na coleção de problemas de Liu e Nocedal (1989).

## 📋 **Formulação Matemática**

### **Problema de Otimização**
Considere o problema com restrições:
```
min f(x)
x ∈ C
```

onde `f: ℝⁿ → ℝ` é uma função diferenciável e `C ⊆ ℝⁿ` é um conjunto convexo fechado.

### **Algoritmo de Gradiente Espelhado**

O algoritmo resolve iterativamente o problema:
```
x^{k+1} = argmin_{x ∈ C} {⟨∇f(x^k), x⟩ + (1/η_k) D_φ(x, x^k)}
```

onde:
- `∇f(x^k)` é o gradiente da função objetivo
- `η_k > 0` é o tamanho do passo
- `D_φ(x, y)` é a divergência de Bregman

### **Divergência de Bregman**
Para uma função estritamente convexa `φ: ℝⁿ → ℝ`:
```
D_φ(x, y) = φ(x) - φ(y) - ⟨∇φ(y), x - y⟩
```

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

### **2. Divergências de Bregman**

#### **`euclidean_phi(x)` e `euclidean_grad_phi(x)`**
```python
def euclidean_phi(x):
    """Função potencial: φ(x) = (1/2)||x||²"""
    return 0.5 * np.sum(x**2)

def euclidean_grad_phi(x):
    """Gradiente: ∇φ(x) = x"""
    return x
```

#### **`entropy_phi(x)` e `entropy_grad_phi(x)`**
```python
def entropy_phi(x):
    """Função potencial: φ(x) = Σᵢ xᵢ log(xᵢ)"""
    x_safe = np.maximum(x, 1e-10)
    return np.sum(x_safe * np.log(x_safe))

def entropy_grad_phi(x):
    """Gradiente: ∇φ(x) = 1 + log(x)"""
    x_safe = np.maximum(x, 1e-10)
    return 1 + np.log(x_safe)
```

#### **`p_norm_phi(x, p=2)` e `p_norm_grad_phi(x, p=2)`**
```python
def p_norm_phi(x, p=2):
    """Função potencial: φ(x) = (1/p)||x||ᵖᵖ"""
    return (1/p) * np.sum(np.abs(x)**p)

def p_norm_grad_phi(x, p=2):
    """Gradiente: ∇φ(x) = sign(x) * |x|^(p-1)"""
    return np.sign(x) * (np.abs(x) ** (p - 1))
```

### **3. `gradiente_espelhado(f, x0, eta_inicial, max_iter, tol, bregman, bounds, p)`**
```python
def gradiente_espelhado(f, x0, eta_inicial=0.01, max_iter=1000, tol=1e-6, 
                       bregman="euclidean", bounds=None, p=2):
    """
    Algoritmo de Gradiente Espelhado com passo adaptativo
    
    Args:
        f: função objetivo
        x0: ponto inicial
        eta_inicial: passo inicial
        max_iter: máximo de iterações
        tol: tolerância para convergência
        bregman: tipo de divergência ("euclidean", "entropy", "p_norm")
        bounds: tipo de restrição
        p: parâmetro para norma-p
    
    Returns:
        x: ponto ótimo
        fo: valor da função objetivo
        k: número de iterações
    """
```

**Características**:
- Passo adaptativo baseado na melhoria da função
- Múltiplas divergências de Bregman
- Projeções automáticas para diferentes restrições

### **4. Atualizações por Tipo de Divergência**

#### **Divergência Euclidiana**
```python
if bregman == "euclidean":
    x_new = x - eta * grad_f
    # Projeção na bola unitária se necessário
    if constraint_type == "ball":
        x_norm = np.linalg.norm(x_new)
        if x_norm > 1.0:
            x_new = x_new / x_norm * 0.99
```

#### **Divergência de Entropia**
```python
elif bregman == "entropy":
    x_safe = np.maximum(x, 1e-10)
    x_new = np.exp(np.log(x_safe) - eta * grad_f)
    # Normalização para simplex se necessário
    if constraint_type == "simplex":
        x_new = x_new / np.sum(x_new)
```

#### **Divergência Norma-p**
```python
elif bregman == "p_norm":
    if p == 2:
        x_new = x - eta * grad_f
    else:
        grad_phi_x = p_norm_grad_phi(x, p)
        grad_phi_new = grad_phi_x - eta * grad_f
        x_new = np.sign(grad_phi_new) * (np.abs(grad_phi_new) ** (1 / (p - 1)))
```

### **5. Controle Adaptativo do Passo**
```python
# Verificar melhoria
improvement = fo - fo_new

if improvement > 1e-8:
    # Boa melhoria: aceitar e aumentar passo
    x = x_new
    fo = fo_new
    eta = min(eta * 1.2, 2.0)
else:
    # Pouca melhoria: diminuir passo
    eta = eta * 0.15
    if k - last_improvement > 5:
        eta = max(eta, 1e-8)
```

## 🏗️ **Classe Principal**

### **`MirrorGradientOptimizedSolver`**

```python
class MirrorGradientOptimizedSolver:
    def __init__(self, eta=0.01, bregman="euclidean", p=2):
        """
        Inicializa o solver com gradiente espelhado adaptativo
        
        Args:
            eta: parâmetro de passo inicial
            bregman: tipo de divergência de Bregman
            p: parâmetro para norma-p
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
    
    def _determine_constraint_type(self, problem_name, bounds):
        """
        Determina o tipo de restrição apropriado para cada problema
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
    'n_variables': int,       # Número de variáveis
    'constraint_type': str    # Tipo de restrição aplicada
}
```

## 🚀 **Como Usar**

### **Exemplo Básico**
```python
from mirror_gradient_optimized import MirrorGradientOptimizedSolver

# Criar solver com divergência euclidiana
solver = MirrorGradientOptimizedSolver(eta=0.01, bregman="euclidean", p=2)

# Resolver um problema específico
result = solver.solve_problem('ROSENBROCK', max_iter=1000, tol=1e-6)

# Resolver todos os problemas
solver.solve_all_problems()
solver.print_summary()
```

### **Exemplos com Diferentes Divergências**
```python
# Divergência euclidiana
solver_euclidean = MirrorGradientOptimizedSolver(eta=0.01, bregman="euclidean", p=2)

# Divergência de entropia
solver_entropy = MirrorGradientOptimizedSolver(eta=0.1, bregman="entropy", p=2)

# Divergência norma-p
solver_pnorm = MirrorGradientOptimizedSolver(eta=0.01, bregman="p_norm", p=2)
```

### **Execução Completa**
```bash
cd liu_nocedal
python mirror_gradient_optimized.py
```

## ⚙️ **Parâmetros Configuráveis**

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `eta` | 0.01 | Passo inicial |
| `bregman` | "euclidean" | Tipo de divergência |
| `p` | 2 | Parâmetro para norma-p |
| `max_iter` | 1000 | Máximo de iterações |
| `tol` | 1e-6 | Tolerância para convergência |

## 🎯 **Tipos de Divergências**

### **Euclidiana** (`bregman="euclidean"`)
- **Função**: `φ(x) = (1/2)||x||²`
- **Uso**: Problemas gerais sem restrições específicas
- **Projeção**: Bola unitária

### **Entropia** (`bregman="entropy"`)
- **Função**: `φ(x) = Σᵢ xᵢ log(xᵢ)`
- **Uso**: Problemas no simplex ou com variáveis positivas
- **Projeção**: Simplex

### **Norma-p** (`bregman="p_norm"`)
- **Função**: `φ(x) = (1/p)||x||ᵖᵖ`
- **Uso**: Problemas com geometria específica
- **Projeção**: Bola p-norma

## 🔧 **Tipos de Restrições**

### **Bola Unitária** (`constraint_type="ball"`)
```python
x_norm = np.linalg.norm(x_new)
if x_norm > 1.0:
    x_new = x_new / x_norm * 0.99
```

### **Simplex** (`constraint_type="simplex"`)
```python
x_new = x_new / np.sum(x_new)
```

### **Caixa** (`constraint_type="box"`)
```python
for i, (lower, upper) in enumerate(bounds):
    x_new[i] = np.clip(x_new[i], lower, upper)
```

## 📈 **Vantagens do Algoritmo**

1. **Adaptabilidade**: Diferentes geometrias do espaço de otimização
2. **Controle Automático**: Passo adaptativo baseado na melhoria
3. **Eficiência**: Especializado para restrições específicas
4. **Robustez**: Funciona bem em problemas mal condicionados
5. **Flexibilidade**: Múltiplas divergências de Bregman

## 🔍 **Complexidade Computacional**

- **Por iteração**: O(n) - cálculo do gradiente e atualização
- **Total**: O(K·n) onde K é o número de iterações
- **Memória**: O(n) - apenas o vetor de variáveis

## 📋 **Problemas Suportados**

O algoritmo resolve todos os problemas da coleção Liu e Nocedal com restrições automáticas:
- **Rosenbrock**: Bola unitária
- **Extended Rosenbrock**: Bola unitária
- **Extended Powell**: Bola unitária
- **Freudenthal-Roth**: Bola unitária
- **Engvall**: Bola unitária
- **Trigonometric**: Simplex
- **Penalty**: Bola unitária
- **ULTS0**: Restrições de caixa

## 📄 **Saídas Geradas**

- **Tabela resumida**: Estatísticas gerais dos problemas
- **Tabela detalhada**: Valores das variáveis em múltiplas colunas
- **PDFs**: Gerados automaticamente via LaTeX
- **Logs**: Progresso da otimização no console

## 🎓 **Referências Teóricas**

- **Mirror Descent**: Nemirovski & Yudin (1983)
- **Divergências de Bregman**: Bregman (1967)
- **Problemas de Teste**: Liu & Nocedal (1989)
