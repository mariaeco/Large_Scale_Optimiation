import numpy as np
from numpy import dot, zeros
from scipy.optimize import minimize_scalar, minimize
from math import log
from problems.setup_problems import*
import time
import os
from datetime import datetime
import random
from latex_to_pdf import salvar_pdf, generate_latex_table


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


def descida_por_coordenadas_ccd(f, x0, max_iter, tol):
    """
    Algoritmo otimizado de Descida Cíclica por Coordenadas (CCD)
    Baseado no algoritmo otimizado do DescendingByCoordinate_outro.py
    
    Parâmetros:
    - f: função objetivo
    - x0: ponto inicial
    - max_iter: máximo de iterações
    - tol: tolerância para convergência
    """
    
    x = x0.copy()
    n = len(x)
    fo = f(x0)
    
    # Calcular gradiente inicial
    grad = calcular_gradiente(f, x)
    
    for k in range(max_iter):
        x_anterior = x.copy()
        fo_anterior = fo
        
        # Atualizar cada coordenada sequencialmente
        for i in range(n):
            # Calcular gradiente da coordenada i
            grad_i = calcular_gradiente(f, x)
            grad_i_val = grad_i[i]
            
            # Fórmula fechada para minimização 1D (aproximação quadrática)
            # Para f(x) ≈ f(x_0) + grad_i * (x_i - x_0_i) + 0.5 * hess_ii * (x_i - x_0_i)²
            # A solução ótima é: x_i = x_0_i - grad_i / hess_ii
            
            # Estimar segunda derivada numericamente
            h = 1e-6
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            
            grad_plus = calcular_gradiente(f, x_plus)
            grad_minus = calcular_gradiente(f, x_minus)
            hess_ii = (grad_plus[i] - grad_minus[i]) / (2 * h)
            
            # Evitar divisão por zero
            if abs(hess_ii) > 1e-12:
                delta_x_i = -grad_i_val / hess_ii
                x[i] += delta_x_i
            else:
                # Fallback: usar passo fixo pequeno
                x[i] -= 1e-6 * grad_i_val
        
        # Calcular novo valor da função
        fo = f(x)
        
        # Verificar convergência
        if np.linalg.norm(x - x_anterior) < tol:
            break
    
    return x, fo, k


def descida_por_coordenadas_rcd(f, x0, max_iter, tol):
    """
    Algoritmo otimizado de Descida Aleatória por Coordenadas (RCD)
    
    Parâmetros:
    - f: função objetivo
    - x0: ponto inicial
    - max_iter: máximo de iterações
    - tol: tolerância para convergência
    """
    
    x = x0.copy()
    n = len(x)
    fo = f(x0)
    
    for k in range(max_iter):
        x_anterior = x.copy()
        fo_anterior = fo
        
        # Selecionar coordenada aleatória
        i = random.randint(0, n-1)
        
        # Calcular gradiente da coordenada i
        grad_i = calcular_gradiente(f, x)
        grad_i_val = grad_i[i]
        
        # Estimar segunda derivada numericamente
        h = 1e-6
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        
        grad_plus = calcular_gradiente(f, x_plus)
        grad_minus = calcular_gradiente(f, x_minus)
        hess_ii = (grad_plus[i] - grad_minus[i]) / (2 * h)
        
        # Evitar divisão por zero
        if abs(hess_ii) > 1e-12:
            delta_x_i = -grad_i_val / hess_ii
            x[i] += delta_x_i
        else:
            # Fallback: usar passo fixo pequeno
            x[i] -= 1e-6 * grad_i_val
        
        # Calcular novo valor da função
        fo = f(x)
        
        # Verificar convergência
        if np.linalg.norm(x - x_anterior) < tol:
            break
    
    return x, fo, k


def descida_por_coordenadas_mdcd(f, x0, max_iter, tol):
    """
    Algoritmo otimizado de Descida por Coordenadas de Máxima Descida (MDCD)
    
    Parâmetros:
    - f: função objetivo
    - x0: ponto inicial
    - max_iter: máximo de iterações
    - tol: tolerância para convergência
    """
    
    x = x0.copy()
    n = len(x)
    fo = f(x0)
    
    for k in range(max_iter):
        x_anterior = x.copy()
        fo_anterior = fo
        
        # Calcular gradiente completo
        grad = calcular_gradiente(f, x)
        
        # Calcular potencial de descida para cada coordenada
        potencial_descida = np.zeros(n)
        
        for i in range(n):
            # Estimar segunda derivada numericamente
            h = 1e-6
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            
            grad_plus = calcular_gradiente(f, x_plus)
            grad_minus = calcular_gradiente(f, x_minus)
            hess_ii = (grad_plus[i] - grad_minus[i]) / (2 * h)
            
            # Potencial de descida = grad_i² / hess_ii
            if abs(hess_ii) > 1e-12:
                potencial_descida[i] = (grad[i] ** 2) / hess_ii
            else:
                potencial_descida[i] = 0
        
        # Selecionar coordenada com maior potencial de descida
        if np.max(potencial_descida) < 1e-15:
            # Nenhuma coordenada oferece descida significativa
            break
        
        i_star = np.argmax(potencial_descida)
        grad_i_val = grad[i_star]
        
        # Estimar segunda derivada para a coordenada selecionada
        h = 1e-6
        x_plus = x.copy()
        x_plus[i_star] += h
        x_minus = x.copy()
        x_minus[i_star] -= h
        
        grad_plus = calcular_gradiente(f, x_plus)
        grad_minus = calcular_gradiente(f, x_minus)
        hess_ii = (grad_plus[i_star] - grad_minus[i_star]) / (2 * h)
        
        # Evitar divisão por zero
        if abs(hess_ii) > 1e-12:
            delta_x_i = -grad_i_val / hess_ii
            x[i_star] += delta_x_i
        else:
            # Fallback: usar passo fixo pequeno
            x[i_star] -= 1e-6 * grad_i_val
        
        # Calcular novo valor da função
        fo = f(x)
        
        # Verificar convergência
        if np.linalg.norm(x - x_anterior) < tol:
            break
    
    return x, fo, k


def descida_por_coordenadas_adaptativa(f, x0, max_iter, tol):
    """
    Algoritmo de Descida por Coordenadas com estratégia adaptativa
    Alterna entre CCD, RCD e MDCD baseado na performance
    
    Parâmetros:
    - f: função objetivo
    - x0: ponto inicial
    - max_iter: máximo de iterações
    - tol: tolerância para convergência
    """
    
    x = x0.copy()
    n = len(x)
    fo = f(x0)
    
    # Estratégias disponíveis
    strategies = ['ccd', 'rcd', 'mdcd']
    current_strategy = 0
    last_improvement = 0
    improvement_threshold = 1e-8
    
    for k in range(max_iter):
        x_anterior = x.copy()
        fo_anterior = fo
        
        # Aplicar estratégia atual
        if strategies[current_strategy] == 'ccd':
            # CCD: atualizar todas as coordenadas
            for i in range(n):
                grad_i = calcular_gradiente(f, x)
                grad_i_val = grad_i[i]
                
                h = 1e-6
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                
                grad_plus = calcular_gradiente(f, x_plus)
                grad_minus = calcular_gradiente(f, x_minus)
                hess_ii = (grad_plus[i] - grad_minus[i]) / (2 * h)
                
                if abs(hess_ii) > 1e-12:
                    delta_x_i = -grad_i_val / hess_ii
                    x[i] += delta_x_i
                else:
                    x[i] -= 1e-6 * grad_i_val
        
        elif strategies[current_strategy] == 'rcd':
            # RCD: atualizar coordenada aleatória
            i = random.randint(0, n-1)
            grad_i = calcular_gradiente(f, x)
            grad_i_val = grad_i[i]
            
            h = 1e-6
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            
            grad_plus = calcular_gradiente(f, x_plus)
            grad_minus = calcular_gradiente(f, x_minus)
            hess_ii = (grad_plus[i] - grad_minus[i]) / (2 * h)
            
            if abs(hess_ii) > 1e-12:
                delta_x_i = -grad_i_val / hess_ii
                x[i] += delta_x_i
            else:
                x[i] -= 1e-6 * grad_i_val
        
        elif strategies[current_strategy] == 'mdcd':
            # MDCD: atualizar coordenada com maior potencial de descida
            grad = calcular_gradiente(f, x)
            potencial_descida = np.zeros(n)
            
            for i in range(n):
                h = 1e-6
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                
                grad_plus = calcular_gradiente(f, x_plus)
                grad_minus = calcular_gradiente(f, x_minus)
                hess_ii = (grad_plus[i] - grad_minus[i]) / (2 * h)
                
                if abs(hess_ii) > 1e-12:
                    potencial_descida[i] = (grad[i] ** 2) / hess_ii
                else:
                    potencial_descida[i] = 0
            
            if np.max(potencial_descida) > 1e-15:
                i_star = np.argmax(potencial_descida)
                grad_i_val = grad[i_star]
                
                h = 1e-6
                x_plus = x.copy()
                x_plus[i_star] += h
                x_minus = x.copy()
                x_minus[i_star] -= h
                
                grad_plus = calcular_gradiente(f, x_plus)
                grad_minus = calcular_gradiente(f, x_minus)
                hess_ii = (grad_plus[i_star] - grad_minus[i_star]) / (2 * h)
                
                if abs(hess_ii) > 1e-12:
                    delta_x_i = -grad_i_val / hess_ii
                    x[i_star] += delta_x_i
                else:
                    x[i_star] -= 1e-6 * grad_i_val
        
        # Calcular novo valor da função
        fo = f(x)
        
        # Verificar melhoria
        improvement = fo_anterior - fo
        
        if improvement > improvement_threshold:
            # Boa melhoria: manter estratégia atual
            last_improvement = k
        else:
            # Pouca melhoria: trocar estratégia
            if k - last_improvement > 5:
                current_strategy = (current_strategy + 1) % len(strategies)
                last_improvement = k
        
        # Verificar convergência
        if np.linalg.norm(x - x_anterior) < tol:
            break
    
    return x, fo, k


class CoordinateDescentOptimizedSolver:
    """
    Classe para resolver problemas de otimização usando algoritmos otimizados de Descida por Coordenadas
    e gerar tabelas de resultados em formato LaTeX.
    """
    
    def __init__(self, algorithm='ccd'):
        """
        Inicializa o solver com diferentes algoritmos de descida por coordenadas.
        
        Args:
            algorithm (str): Tipo de algoritmo ('ccd', 'rcd', 'mdcd', 'adaptativo')
        """
        self.results = []
        self.problems_config = setup_problems()
        self.algorithm = algorithm
    
    def solve_problem(self, problem_name, max_iter=1000, tol=1e-3):
        """
        Resolve um problema específico usando Descida por Coordenadas Otimizada.
        
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
            if self.algorithm == 'ccd':
                x_cd, fo, iterations = descida_por_coordenadas_ccd(objective_with_args, x0, max_iter, tol)
            elif self.algorithm == 'rcd':
                x_cd, fo, iterations = descida_por_coordenadas_rcd(objective_with_args, x0, max_iter, tol)
            elif self.algorithm == 'mdcd':
                x_cd, fo, iterations = descida_por_coordenadas_mdcd(objective_with_args, x0, max_iter, tol)
            elif self.algorithm == 'adaptativo':
                x_cd, fo, iterations = descida_por_coordenadas_adaptativa(objective_with_args, x0, max_iter, tol)
            else:
                # Fallback para CCD
                x_cd, fo, iterations = descida_por_coordenadas_ccd(objective_with_args, x0, max_iter, tol)
            
            end_time = time.time()
            
            # Calcular norma do gradiente final
            grad_norm_final = calcular_gradiente(objective_with_args, x_cd)
            grad_norm = np.linalg.norm(grad_norm_final)
            
            # Armazenar resultados
            result_dict = {
                'problem': problem_name,
                'success': True,
                'iterations': iterations + 1,
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
    
    def solve_all_problems(self, max_iter=1000, tol=1e-3):
        """
        Resolve todos os problemas configurados.
        
        Args:
            max_iter (int): Número máximo de iterações
            tol (float): Tolerância para convergência
        """
        print(f"Iniciando resolução de todos os problemas com Descida por Coordenadas Otimizada ({self.algorithm.upper()})...")
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
        print("RESUMO DOS RESULTADOS - DESCIDA POR COORDENADAS OTIMIZADA")
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


def compare_coordinate_algorithms(problem_name, max_iter=500, tol=1e-4):
    """
    Compara diferentes versões do algoritmo de descida por coordenadas
    
    Args:
        problem_name (str): Nome do problema a ser testado
        max_iter (int): Número máximo de iterações
        tol (float): Tolerância para convergência
    """
    print(f"\nComparação: Diferentes versões da Descida por Coordenadas - {problem_name}")
    print("=" * 70)
    
    # Configurar problema
    solver_base = CoordinateDescentOptimizedSolver(algorithm='ccd')
    
    if problem_name not in solver_base.problems_config:
        print(f"Problema '{problem_name}' não encontrado!")
        return
    
    config = solver_base.problems_config[problem_name]
    
    if problem_name == 'ULTS0':
        n, x0, grid_shape, h = config['setup']()
        args = (grid_shape, h)
        
        def objective_with_args(x):
            return config['objective'](x, grid_shape, h)
    else:
        n, x0 = config['setup']()
        args = config['args']
        objective_with_args = config['objective']
    
    algorithms = [
        ("ccd", "Descida Cíclica por Coordenadas (CCD)"),
        ("rcd", "Descida Aleatória por Coordenadas (RCD)"),
        ("mdcd", "Descida por Coordenadas de Máxima Descida (MDCD)"),
        ("adaptativo", "Descida por Coordenadas Adaptativa")
    ]
    
    results = {}
    
    for algo_name, description in algorithms:
        print(f"\n{description}:")
        start_time = time.time()
        try:
            if algo_name == 'ccd':
                x_result, fo, iterations = descida_por_coordenadas_ccd(objective_with_args, x0, max_iter, tol)
            elif algo_name == 'rcd':
                x_result, fo, iterations = descida_por_coordenadas_rcd(objective_with_args, x0, max_iter, tol)
            elif algo_name == 'mdcd':
                x_result, fo, iterations = descida_por_coordenadas_mdcd(objective_with_args, x0, max_iter, tol)
            elif algo_name == 'adaptativo':
                x_result, fo, iterations = descida_por_coordenadas_adaptativa(objective_with_args, x0, max_iter, tol)
            
            time_taken = time.time() - start_time
            grad_norm = np.linalg.norm(calcular_gradiente(objective_with_args, x_result))
            
            print(f"   ✓ Sucesso: {iterations + 1} iterações, f* = {fo:.6e}")
            print(f"   ✓ Tempo: {time_taken:.3f}s, ||∇f|| = {grad_norm:.6e}")
            
            results[algo_name] = {
                'success': True,
                'iterations': iterations + 1,
                'function_value': fo,
                'time': time_taken,
                'gradient_norm': grad_norm
            }
            
        except Exception as e:
            time_taken = time.time() - start_time
            print(f"   ✗ Falhou: {str(e)}")
            print(f"   ✗ Tempo: {time_taken:.3f}s")
            
            results[algo_name] = {
                'success': False,
                'iterations': 0,
                'function_value': float('inf'),
                'time': time_taken,
                'gradient_norm': float('inf')
            }
    
    # Comparação final
    print("\n" + "=" * 70)
    print("COMPARAÇÃO FINAL:")
    print("-" * 70)
    print(f"{'Algoritmo':<35} {'Iterações':<10} {'Tempo (s)':<10} {'||∇f||':<15}")
    print("-" * 70)
    
    for algo_name, description in algorithms:
        if algo_name in results:
            r = results[algo_name]
            if r['success']:
                print(f"{description:<35} {r['iterations']:<10} {r['time']:<10.3f} {r['gradient_norm']:<15.6e}")
            else:
                print(f"{description:<35} {'Falhou':<10} {r['time']:<10.3f} {'---':<15}")


def main():
    """
    Função principal para executar a análise.
    """
    print("=" * 80)
    print("DESCIDA POR COORDENADAS OTIMIZADA - COMPARAÇÃO DE VERSÕES")
    print("=" * 80)
    
    versions = [
        ('ccd', 'Descida Cíclica por Coordenadas (CCD)'),
        # ('rcd', 'Descida Aleatória por Coordenadas (RCD)'),
        # ('mdcd', 'Descida por Coordenadas de Máxima Descida (MDCD)'),
        # ('adaptativo', 'Descida por Coordenadas Adaptativa')
    ]
    
    all_results = {}
    
    for version_name, description in versions:
        print(f"\n{description}")
        print("-" * 60)
        
        # Criar solver com versão específica
        solver = CoordinateDescentOptimizedSolver(algorithm=version_name)
        
        # Resolver todos os problemas
        solver.solve_all_problems(max_iter=500, tol=1e-4)
        
        # Imprimir resumo
        solver.print_summary()
        
        # Armazenar resultados
        all_results[version_name] = solver.results
        
        # Gerar tabela LaTeX específica
        filename = f'liu_nocedal/latex_solution/resultados_coordinate_descent_{version_name}.tex'
        method_name = f'Descida por Coordenadas Otimizada ({description})'
        generate_latex_table(solver.results, filename, method_name)
        salvar_pdf(filename, 'liu_nocedal/latex_solution/')
    
    # Comparação final
    print("\n" + "=" * 80)
    print("COMPARAÇÃO FINAL DE PERFORMANCE")
    print("=" * 80)
    
    for version_name, results in all_results.items():
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        avg_time = np.mean([r['execution_time'] for r in results if r['success']])
        avg_iterations = np.mean([r['iterations'] for r in results if r['success']])
        
        print(f"{version_name.upper():<12}: {successful}/{total} sucessos, "
              f"tempo médio: {avg_time:.3f}s, iterações médias: {avg_iterations:.1f}")
    
    print("\nAnálise concluída!")


if __name__ == "__main__":
    main()
