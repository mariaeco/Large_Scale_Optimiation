
import numpy as np
from numpy import dot, zeros
from scipy.optimize import minimize_scalar, minimize
from math import log
from problems.setup_problems import*
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import random
from latex_to_pdf import salvar_pdf, generate_latex_table


def gradiente_descendente(f, x0, alpha, max_iter, tol):
    """
    Pseudo-algoritmo do Gradiente Descendente
    
    Parâmetros:
    - f: função objetivo
    - x0: ponto inicial
    - alpha: taxa de aprendizado
    - max_iter: máximo de iterações
    - tol: tolerância para convergência
    """
    
    x = x0
    
    for k in range(max_iter):
        # 1. Calcular gradiente
        grad = calcular_gradiente(f, x)
        
        # 2. Atualizar todas as coordenadas simultaneamente
        x_novo = x - alpha * grad
        
        # 3. Verificar convergência
        if np.linalg.norm(x_novo - x) < tol:
            break
            
        x = x_novo
    
    return x


def calcular_gradiente(f, x):
    """
    Calcula gradiente numericamente ou analiticamente
    """
    n = len(x)
    grad = zeros(n)
    
    for i in range(n):
        # Diferenças finitas centrais
        h = 1e-6
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad


def descida_por_coordenadas(f, x0, max_iter, tol):
    """
    Pseudo-algoritmo da Descida por Coordenadas
    
    Parâmetros:
    - f: função objetivo
    - x0: ponto inicial
    - max_iter: máximo de iterações
    - tol: tolerância para convergência
    """
    
    x = x0
    n = len(x)
    historico = [x.copy()]
    fo = f(x0)  # Inicializar fo com o valor inicial
    
    for k in range(max_iter):
        x_anterior = x.copy()
        
        # 1. Atualizar cada coordenada sequencialmente
        for i in range(n):
            # 2. Minimizar f em relação à coordenada i
            x[i] = minimizar_coordenada(f, x, i)
        
        # 3. Verificar convergência
        if np.linalg.norm(x - x_anterior) < tol:
            fo = f(x)  # Calcular valor final da função
            break
            
        fo = f(x)  # Calcular valor da função no novo ponto
        historico.append(x.copy())
    
    return x, fo, k 

def minimizar_coordenada(f, x, i):
    """
    Minimiza f(x) em relação à coordenada i, mantendo outras fixas
    """
    
    def funcao_1d(xi):
        # Criar cópia de x com coordenada i alterada
        x_temp = x.copy()
        x_temp[i] = xi
        return f(x_temp)
    
    # Resolver problema 1D usando busca linear ou método de Newton
    resultado = minimize_1d(funcao_1d, x0=x[i])
    return resultado.x

def minimize_1d(f, x0, method='brent'):
    """
    Minimização 1D usando diferentes métodos
    """
    if method == 'brent':
        # Método de Brent (busca linear)
        resultado = minimize_scalar(f, method='brent')
    elif method == 'newton':
        # Método de Newton 1D
        resultado = newton_1d(f, x0)
    else:
        # Fallback para busca linear simples
        resultado = minimize_scalar(f, method='brent')
    
    return resultado

def newton_1d(f, x0, max_iter=100, tol=1e-6):
    """
    Método de Newton 1D
    """
    x = x0
    
    for k in range(max_iter):
        # Calcular primeira e segunda derivadas
        fx = f(x)
        fx_prime = derivada_primeira(f, x)
        fx_double_prime = derivada_segunda(f, x)
        
        if abs(fx_double_prime) < 1e-12:
            break
            
        # Atualização de Newton
        x_novo = x - fx_prime / fx_double_prime
        
        if abs(x_novo - x) < tol:
            break
            
        x = x_novo
    
    return x

def derivada_primeira(f, x, h=1e-6):
    """Calcula primeira derivada numericamente"""
    return (f(x + h) - f(x - h)) / (2 * h)

def derivada_segunda(f, x, h=1e-6):
    """Calcula segunda derivada numericamente"""
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)


def descida_por_coordenadas_otimizada(f, x0, max_iter, tol, block_size=None, strategy='random'):
    """
    Algoritmo otimizado da Descida por Coordenadas com blocos de coordenadas
    
    Parâmetros:
    - f: função objetivo
    - x0: ponto inicial
    - max_iter: máximo de iterações
    - tol: tolerância para convergência
    - block_size: tamanho do bloco de coordenadas (padrão: min(10, n//4))
    - strategy: estratégia de seleção ('random', 'cyclic', 'greedy')
    """
    
    x = x0.copy()
    n = len(x)
    
    # Definir tamanho do bloco se não especificado
    if block_size is None:
        block_size = min(max(1, n // 4), 10)
    
    fo = f(x0)
    
    for k in range(max_iter):
        x_anterior = x.copy()
        
        # Selecionar coordenadas para atualizar
        if strategy == 'random':
            # Seleção aleatória de coordenadas
            indices = random.sample(range(n), min(block_size, n))
        elif strategy == 'cyclic':
            # Seleção cíclica
            start_idx = k % n
            indices = [(start_idx + i) % n for i in range(min(block_size, n))]
        elif strategy == 'greedy':
            # Seleção baseada no gradiente (maior magnitude)
            grad = calcular_gradiente(f, x)
            grad_magnitudes = np.abs(grad)
            indices = np.argsort(grad_magnitudes)[-min(block_size, n):].tolist()
        else:
            # Fallback para todas as coordenadas
            indices = list(range(n))
        
        # Minimizar em relação ao bloco de coordenadas selecionadas
        x = minimizar_bloco_coordenadas(f, x, indices)
        
        # Verificar convergência
        if np.linalg.norm(x - x_anterior) < tol:
            fo = f(x)
            break
            
        fo = f(x)
    
    return x, fo, k


def minimizar_bloco_coordenadas(f, x, indices):
    """
    Minimiza f(x) em relação a um bloco de coordenadas simultaneamente
    
    Args:
        f: função objetivo
        x: ponto atual
        indices: lista de índices das coordenadas a serem otimizadas
    
    Returns:
        x atualizado
    """
    if len(indices) == 1:
        # Caso especial: uma coordenada (usar método 1D)
        i = indices[0]
        x[i] = minimizar_coordenada(f, x, i)
        return x
    
    # Criar função objetivo para o bloco de coordenadas
    def funcao_bloco(coords_bloco):
        x_temp = x.copy()
        for idx, coord_idx in enumerate(indices):
            x_temp[coord_idx] = coords_bloco[idx]
        return f(x_temp)
    
    # Ponto inicial para o bloco
    x0_bloco = np.array([x[i] for i in indices])
    
    # Minimizar usando scipy.optimize.minimize
    try:
        resultado = minimize(funcao_bloco, x0_bloco, method='BFGS', 
                           options={'gtol': 1e-6, 'maxiter': 50})
        
        if resultado.success:
            # Atualizar coordenadas com os valores otimizados
            for idx, coord_idx in enumerate(indices):
                x[coord_idx] = resultado.x[idx]
        else:
            # Fallback: minimizar coordenadas individualmente
            for i in indices:
                x[i] = minimizar_coordenada(f, x, i)
                
    except Exception:
        # Fallback: minimizar coordenadas individualmente
        for i in indices:
            x[i] = minimizar_coordenada(f, x, i)
    
    return x


def descida_por_coordenadas_paralela(f, x0, max_iter, tol, block_size=None, n_workers=4):
    """
    Algoritmo de Descida por Coordenadas com processamento paralelo
    
    Parâmetros:
    - f: função objetivo
    - x0: ponto inicial
    - max_iter: máximo de iterações
    - tol: tolerância para convergência
    - block_size: tamanho do bloco de coordenadas
    - n_workers: número de workers para processamento paralelo
    """
    
    x = x0.copy()
    n = len(x)
    
    if block_size is None:
        block_size = min(max(1, n // 4), 10)
    
    fo = f(x0)
    
    for k in range(max_iter):
        x_anterior = x.copy()
        
        # Dividir coordenadas em blocos para processamento paralelo
        blocks = []
        for i in range(0, n, block_size):
            block_indices = list(range(i, min(i + block_size, n)))
            blocks.append(block_indices)
        
        # Processar blocos em paralelo
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Criar funções para cada bloco
            futures = []
            for block_indices in blocks:
                future = executor.submit(minimizar_bloco_coordenadas, f, x.copy(), block_indices)
                futures.append((future, block_indices))
            
            # Coletar resultados e atualizar x
            for future, block_indices in futures:
                try:
                    x_block = future.result(timeout=30)  # timeout de 30s
                    # Atualizar apenas as coordenadas deste bloco
                    for i in block_indices:
                        x[i] = x_block[i]
                except Exception:
                    # Fallback: minimizar coordenadas individualmente
                    for i in block_indices:
                        x[i] = minimizar_coordenada(f, x, i)
        
        # Verificar convergência
        if np.linalg.norm(x - x_anterior) < tol:
            fo = f(x)
            break
            
        fo = f(x)
    
    return x, fo, k


def descida_por_coordenadas_adaptativa(f, x0, max_iter, tol):
    """
    Algoritmo de Descida por Coordenadas com tamanho de bloco adaptativo
    
    Parâmetros:
    - f: função objetivo
    - x0: ponto inicial
    - max_iter: máximo de iterações
    - tol: tolerância para convergência
    """
    
    x = x0.copy()
    n = len(x)
    
    # Tamanho inicial do bloco
    block_size = max(1, n // 8)
    min_block_size = 1
    max_block_size = min(n, 20)
    
    fo = f(x0)
    last_improvement = 0
    
    for k in range(max_iter):
        x_anterior = x.copy()
        fo_anterior = fo
        
        # Seleção aleatória de coordenadas
        indices = random.sample(range(n), min(block_size, n))
        
        # Minimizar bloco
        x = minimizar_bloco_coordenadas(f, x, indices)
        fo = f(x)
        
        # Verificar melhoria
        improvement = fo_anterior - fo
        
        if improvement > 1e-8:
            # Boa melhoria: manter ou aumentar tamanho do bloco
            last_improvement = k
            if improvement > 1e-6 and block_size < max_block_size:
                block_size = min(block_size + 1, max_block_size)
        else:
            # Pouca melhoria: diminuir tamanho do bloco
            if k - last_improvement > 5 and block_size > min_block_size:
                block_size = max(block_size - 1, min_block_size)
        
        # Verificar convergência
        if np.linalg.norm(x - x_anterior) < tol:
            break
    
    return x, fo, k


    
class CoordinateDescentSolver:
    """
    Classe para resolver problemas de otimização usando o método da Descida por Coordenadas
    e gerar tabelas de resultados em formato LaTeX.
    """
    
    def __init__(self, algorithm='optimized', block_size=None, strategy='random'):
        """
        Inicializa o solver com diferentes algoritmos de descida por coordenadas.
        
        Args:
            algorithm (str): Tipo de algoritmo ('original', 'optimized', 'parallel', 'adaptive')
            block_size (int): Tamanho do bloco de coordenadas (None para automático)
            strategy (str): Estratégia de seleção ('random', 'cyclic', 'greedy')
        """
        self.results = []
        self.problems_config = setup_problems()
        self.algorithm = algorithm
        self.block_size = block_size
        self.strategy = strategy
    
    
    def solve_problem(self, problem_name, max_iter=500, tol=1e-3):
        """
        Resolve um problema específico usando Descida por Coordenadas.
        
        Args:
            problem_name (str): Nome do problema
            max_iter (int): Número máximo de iterações
            tol (float): Tolerância para convergência
            
        Returns:
            dict: Resultados da otimização
        """
        if problem_name not in self.problems_config:
            raise ValueError(f"Problema '{problem_name}' não encontrado")
        
        config = self.problems_config[problem_name]
        
        # Configurar o problema
        if problem_name == 'ULTS0':
            n, x0, grid_shape, h = config['setup']()
            args = (grid_shape, h)
            
            # Função objetivo com argumentos
            def objective_with_args(x):
                return config['objective'](x, grid_shape, h)
        else:
            n, x0 = config['setup']()
            args = config['args']
            objective_with_args = config['objective']
        
        # Executar otimização
        start_time = time.time()
        
        try:
            # Escolher algoritmo baseado na configuração
            if self.algorithm == 'original':
                x_cd, fo, iterations = descida_por_coordenadas(objective_with_args, x0, max_iter, tol)
            elif self.algorithm == 'optimized':
                x_cd, fo, iterations = descida_por_coordenadas_otimizada(
                    objective_with_args, x0, max_iter, tol, 
                    block_size=self.block_size, strategy=self.strategy
                )
            elif self.algorithm == 'parallel':
                x_cd, fo, iterations = descida_por_coordenadas_paralela(
                    objective_with_args, x0, max_iter, tol, 
                    block_size=self.block_size, n_workers=4
                )
            elif self.algorithm == 'adaptive':
                x_cd, fo, iterations = descida_por_coordenadas_adaptativa(
                    objective_with_args, x0, max_iter, tol
                )
            else:
                # Fallback para algoritmo otimizado
                x_cd, fo, iterations = descida_por_coordenadas_otimizada(
                    objective_with_args, x0, max_iter, tol, 
                    block_size=self.block_size, strategy=self.strategy
                )
            
            end_time = time.time()
            
            # Calcular norma do gradiente final
            grad_norm_final = calcular_gradiente(objective_with_args, x_cd)
            grad_norm = np.linalg.norm(grad_norm_final)
            
            # Armazenar resultados
            result_dict = {
                'problem': problem_name,
                'success': True,
                'iterations': iterations+1,
                'function_value': fo,
                'x_value': x_cd,
                'gradient_norm': grad_norm,
                'message': "Convergência atingida",
                'execution_time': end_time - start_time,
                'n_variables': n
            }
            
        except Exception as e:
            end_time = time.time()
            result_dict = {
                'problem': problem_name,
                'success': False,
                'iterations': 0,
                'function_value': float('inf'),
                'x_value': None,
                'gradient_norm': float('inf'),
                'message': f"Erro: {str(e)}",
                'execution_time': end_time - start_time,
                'n_variables': n
            }
        
        return result_dict
    
    def solve_all_problems(self, max_iter=1000, tol=1e-6):
        """
        Resolve todos os problemas configurados.
        
        Args:
            max_iter (int): Número máximo de iterações
            tol (float): Tolerância para convergência
        """
        print(f"Iniciando resolução de todos os problemas com Descida por Coordenadas ({self.algorithm})...")
        if self.algorithm == 'optimized':
            print(f"  - Estratégia: {self.strategy}")
            print(f"  - Tamanho do bloco: {self.block_size if self.block_size else 'automático'}")
        print("=" * 70)
        
        self.results = []
        
        for i, problem_name in enumerate(self.problems_config.keys(), 1):
            print(f"\n[{i}/{len(self.problems_config)}] Resolvendo: {problem_name}")
            
            result = self.solve_problem(problem_name, max_iter, tol)
            self.results.append(result)
            
            # Imprimir resultado
            if result['success']:
                print(f"  ✓ Sucesso: {result['iterations']} iterações, f* = {result['function_value']:.6e}")
            else:
                print(f"  ✗ Falhou: {result['message']}")
        
        print("\n" + "=" * 70)
        print("Resolução concluída!")
    
    
    def print_summary(self):
        """
        Imprime um resumo dos resultados.
        """
        if not self.results:
            print("Nenhum resultado disponível.")
            return
        
        print("\n" + "=" * 80)
        print("RESUMO DOS RESULTADOS - DESCIDA POR COORDENADAS")
        print("=" * 80)
        
        successful = sum(1 for r in self.results if r['success'])
        total = len(self.results)
        
        print(f"Problemas resolvidos com sucesso: {successful}/{total}")
        print(f"Taxa de sucesso: {successful/total*100:.1f}%")
        
        if successful > 0:
            successful_results = [r for r in self.results if r['success']]
            avg_iterations = np.mean([r['iterations'] for r in successful_results])
            avg_time = np.mean([r['execution_time'] for r in successful_results])
            total_time = sum([r['execution_time'] for r in successful_results])
            print(f"Número médio de iterações: {avg_iterations:.1f}")
            print(f"Tempo médio por problema: {avg_time:.3f}s")
            print(f"Tempo total de execução: {total_time:.3f}s")
        
        print("\nDetalhes por problema:")
        print("-" * 95)
        print(f"{'Problema':<25} {'Status':<8} {'Iterações':<10} {'Valor Mínimo':<15} {'Tempo (s)':<10}")
        print("-" * 95)
        
        for result in self.results:
            status = "✓" if result['success'] else "✗"
            iterations = str(result['iterations']) if result['success'] else "---"
            value = f"{result['function_value']:.6e}" if result['success'] else "Falhou"
            time_str = f"{result['execution_time']:.3f}" if result['success'] else "---"
            
            print(f"{result['problem']:<25} {status:<8} {iterations:<10} {value:<15} {time_str:<10}")


def compare_with_lbfgsb(problem_name, max_iter=1000, tol=1e-6):
    """
    Compara diretamente o algoritmo otimizado de descida por coordenadas com L-BFGS-B
    
    Args:
        problem_name (str): Nome do problema a ser testado
        max_iter (int): Número máximo de iterações
        tol (float): Tolerância para convergência
    """
    from scipy.optimize import minimize
    
    print(f"\nComparação: Descida por Coordenadas vs L-BFGS-B - {problem_name}")
    print("=" * 70)
    
    # Configurar problema
    solver = CoordinateDescentSolver(algorithm='optimized', strategy='greedy', block_size=5)
    
    if problem_name not in solver.problems_config:
        print(f"Problema '{problem_name}' não encontrado!")
        return
    
    config = solver.problems_config[problem_name]
    
    if problem_name == 'ULTS0':
        n, x0, grid_shape, h = config['setup']()
        args = (grid_shape, h)
        
        def objective_with_args(x):
            return config['objective'](x, grid_shape, h)
    else:
        n, x0 = config['setup']()
        args = config['args']
        objective_with_args = config['objective']
    
    # Teste 1: Descida por Coordenadas Otimizada
    print("1. Descida por Coordenadas Otimizada:")
    start_time = time.time()
    try:
        x_cd, fo_cd, iterations_cd = descida_por_coordenadas_otimizada(
            objective_with_args, x0, max_iter, tol, 
            block_size=5, strategy='greedy'
        )
        time_cd = time.time() - start_time
        grad_cd = calcular_gradiente(objective_with_args, x_cd)
        grad_norm_cd = np.linalg.norm(grad_cd)
        
        print(f"   ✓ Sucesso: {iterations_cd} iterações, f* = {fo_cd:.6e}")
        print(f"   ✓ Tempo: {time_cd:.3f}s, ||∇f|| = {grad_norm_cd:.6e}")
        success_cd = True
    except Exception as e:
        time_cd = time.time() - start_time
        print(f"   ✗ Falhou: {str(e)}")
        print(f"   ✗ Tempo: {time_cd:.3f}s")
        success_cd = False
    
    # Teste 2: L-BFGS-B
    print("\n2. L-BFGS-B:")
    start_time = time.time()
    try:
        result_lbfgsb = minimize(objective_with_args, x0, method='L-BFGS-B', 
                                options={'maxiter': max_iter, 'gtol': tol})
        time_lbfgsb = time.time() - start_time
        
        if result_lbfgsb.success:
            print(f"   ✓ Sucesso: {result_lbfgsb.nit} iterações, f* = {result_lbfgsb.fun:.6e}")
            print(f"   ✓ Tempo: {time_lbfgsb:.3f}s, ||∇f|| = {result_lbfgsb.jac:.6e if hasattr(result_lbfgsb, 'jac') else 'N/A'}")
            success_lbfgsb = True
        else:
            print(f"   ✗ Falhou: {result_lbfgsb.message}")
            print(f"   ✗ Tempo: {time_lbfgsb:.3f}s")
            success_lbfgsb = False
    except Exception as e:
        time_lbfgsb = time.time() - start_time
        print(f"   ✗ Falhou: {str(e)}")
        print(f"   ✗ Tempo: {time_lbfgsb:.3f}s")
        success_lbfgsb = False
    
    # Comparação
    print("\n" + "=" * 70)
    print("COMPARAÇÃO:")
    if success_cd and success_lbfgsb:
        speedup = time_lbfgsb / time_cd if time_cd > 0 else float('inf')
        print(f"Speedup da Descida por Coordenadas: {speedup:.2f}x")
        if time_cd < time_lbfgsb:
            print("✓ Descida por Coordenadas foi mais rápida!")
        else:
            print("✗ L-BFGS-B foi mais rápido")
    else:
        print("Não foi possível comparar (um dos algoritmos falhou)")


def main():
    """
    Função principal para executar a análise.
    """
    print("=" * 80)
    print("COMPARAÇÃO DE ALGORITMOS DE DESCIDA POR COORDENADAS")
    print("=" * 80)
    
    algorithms = [
        ('original', 'Algoritmo Original (1 coordenada por vez)'),
        ('optimized', 'Algoritmo Otimizado (blocos de coordenadas)')
        ('adaptive', 'Algoritmo Adaptativo (tamanho de bloco variável)')
        ('parallel', 'Algoritmo Paralelo (processamento paralelo)')
    ]
    
    all_results = {}
    
    for algorithm, description in algorithms:
        print(f"\n{description}")
        print("-" * 60)
        
        # Criar solver com algoritmo específico
        if algorithm == 'optimized':
            solver = CoordinateDescentSolver(algorithm='optimized', strategy='greedy', block_size=5)
        else:
            solver = CoordinateDescentSolver(algorithm=algorithm)
        
        # Resolver todos os problemas
        solver.solve_all_problems(max_iter=500, tol=1e-4)
        
        # Imprimir resumo
        solver.print_summary()
        
        # Armazenar resultados
        all_results[algorithm] = solver.results
        
        # Gerar tabela LaTeX específica
        filename = f'liu_nocedal/latex_solution/resultados_coordinate_descent_{algorithm}.tex'
        method_name = f'Descida por Coordenadas ({algorithm})'
        generate_latex_table(solver.results, filename, method_name)
        salvar_pdf(filename, 'liu_nocedal/latex_solution/')
    
    # Comparação final
    print("\n" + "=" * 80)
    print("COMPARAÇÃO FINAL DE PERFORMANCE")
    print("=" * 80)
    
    for algorithm, results in all_results.items():
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        avg_time = np.mean([r['execution_time'] for r in results if r['success']])
        avg_iterations = np.mean([r['iterations'] for r in results if r['success']])
        
        print(f"{algorithm.upper():<12}: {successful}/{total} sucessos, "
              f"tempo médio: {avg_time:.3f}s, iterações médias: {avg_iterations:.1f}")
    
    print("\nAnálise concluída!")




if __name__ == "__main__":
    main()