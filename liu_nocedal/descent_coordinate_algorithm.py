
from re import I
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
from latex_to_pdf import salvar_pdf, generate_latex_table, generate_detailed_latex_table


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




def descida_por_coordenadas(f, x0, max_iter, tol, block_size=None, strategy='greedy'):
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
            # Garantir que todas as coordenadas sejam eventualmente selecionadas
            if k % (n // block_size + 1) == 0:
                # A cada n/block_size iterações, usar todas as coordenadas
                indices = list(range(n))
            else:
                indices = random.sample(range(n), min(block_size, n))
        elif strategy == 'cyclic':
            # Seleção cíclica
            start_idx = k % n
            indices = [(start_idx + i) % n for i in range(min(block_size, n))]
        elif strategy == 'greedy':
            # Para problemas grandes, evitar cálculo de gradiente completo
            if n > 500:
                # Usar seleção aleatória para problemas grandes
                indices = random.sample(range(n), min(block_size, n))
            else:
                # Seleção baseada no gradiente (maior magnitude) para problemas pequenos
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
    
    return x, fo, k+1



def minimizar_bloco_coordenadas(f, x, indices):
    """
    Método original usando scipy.optimize.minimize (fallback)
    
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
        resultado = minimize(funcao_bloco, x0_bloco, method='L-BFGS-B', 
                           options={'gtol': 1e-6, 'maxiter': 200})
        
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
    resultado = minimize_1d(funcao_1d, x0=x[i], method='newton')
    return resultado.x



def minimize_1d(f, x0, method='brent'):
    """
    Minimização 1D usando diferentes métodos
    """
    if method == 'brent':
        # Método de Brent (busca linear) com limites mais restritivos
        resultado = minimize_scalar(f, method='brent', options={'maxiter': 50})
    elif method == 'newton':
        # Método de Newton 1D
        resultado = newton_1d(f, x0)
    else:
        # Fallback para busca linear simples
        resultado = minimize_scalar(f, method='brent', options={'maxiter': 50})
    
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
    
    # Retornar objeto similar ao OptimizeResult
    class SimpleResult:
        def __init__(self, x):
            self.x = x
    
    return SimpleResult(x)


def derivada_primeira(f, x, h=1e-6):
    """Calcula primeira derivada numericamente"""
    return (f(x + h) - f(x - h)) / (2 * h)


def derivada_segunda(f, x, h=1e-6):
    """Calcula segunda derivada numericamente"""
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)


    
class CoordinateDescentSolver:
    """
    Classe para resolver problemas de otimização usando o método da Descida por Coordenadas
    """
    
    def __init__(self, algorithm='optimized', block_size=None, strategy='random'):
        """
        Inicializa o solver com diferentes algoritmos de descida por coordenadas.
        
        Args:
            algorithm (str): Tipo de algoritmo ('optimized', 'adaptive')
            block_size (int): Tamanho do bloco de coordenadas (None para automático)
            strategy (str): Estratégia de seleção ('random', 'cyclic', 'greedy')
        """
        self.results = []
        self.problems_config = setup_problems()
        self.algorithm = algorithm
        self.block_size = block_size
        self.strategy = strategy
    
    
    def solve_problem(self, problem_name, max_iter=1000, tol=1e-6):
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

            x_cd, fo, iterations = descida_por_coordenadas(
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




def main():
    """
    Função principal para executar a análise.
    """

    solver = CoordinateDescentSolver(algorithm='optimized', strategy='random', block_size=10)

    # Resolver todos os problemas
    solver.solve_all_problems(max_iter=1000, tol=1e-3)
    
    # Imprimir resumo
    solver.print_summary()

    
     # Gerar tabela LaTeX específica
    method_name = f'Gradiente Espelhado'
    detailed_filename = f'liu_nocedal/latex_solution/resultados_descent_coordinate.tex'
    generate_detailed_latex_table(solver.results, detailed_filename, method_name)
    salvar_pdf(detailed_filename, 'liu_nocedal/latex_solution/')
    

if __name__ == "__main__":
    main()