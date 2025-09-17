import numpy as np
from numpy import dot, zeros
from scipy.optimize import minimize_scalar, minimize
from math import log
from problems.setup_problems import*
import time
import os
from datetime import datetime
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


def gradiente_espelhado_otimizado(f, x0, eta, max_iter, tol, versao="norma_p", p=2):
    """
    Algoritmo otimizado do Gradiente Espelhado (Mirror Descent)
    
    Parâmetros:
    - f: função objetivo
    - x0: ponto inicial
    - eta: parâmetro de passo
    - max_iter: máximo de iterações
    - tol: tolerância para convergência
    - versao: tipo de função de distância ("norma_p" ou "entropia_negativa")
    - p: parâmetro da norma-p (padrão 2)
    """
    
    x = x0.copy()
    fo = f(x0)
    
    for k in range(max_iter):
        # 1. Calcular gradiente da função objetivo
        grad_f = calcular_gradiente(f, x)
        
        # 2. Verificar convergência pela norma do gradiente
        if np.linalg.norm(grad_f) <= tol:
            break
        
        # 3. Atualizar usando fórmula fechada
        if versao == "entropia_negativa":
            # Para entropia negativa: x_new = exp(log(x) - eta * grad_f)
            x_new = np.exp(np.log(np.maximum(x, 1e-10)) - eta * grad_f)
        elif versao == "norma_p":
            # Para norma-p: x_new = sign(y) * |y|^(1/(p-1))
            # onde y = sign(x) * |x|^(p-1) - eta * grad_f
            y = np.sign(x) * (np.abs(x) ** (p - 1)) - eta * grad_f
            x_new = np.sign(y) * (np.abs(y) ** (1 / (p - 1)))
        else:
            # Fallback para euclidiana (p=2)
            x_new = x - eta * grad_f
        
        # 4. Verificar convergência pela mudança no ponto
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            fo = f(x)
            break
        
        x = x_new
        fo = f(x)
    
    return x, fo, k


def gradiente_espelhado_adaptativo(f, x0, eta_inicial=0.01, max_iter=1000, tol=1e-3, versao="norma_p"):
    """
    Algoritmo de Gradiente Espelhado com passo adaptativo
    
    Parâmetros:
    - f: função objetivo
    - x0: ponto inicial
    - eta_inicial: passo inicial
    - max_iter: máximo de iterações
    - tol: tolerância para convergência
    - versao: tipo de função de distância
    """
    
    x = x0.copy()
    eta = eta_inicial
    fo = f(x0)
    last_improvement = 0
    
    for k in range(max_iter):
        # 1. Calcular gradiente da função objetivo
        grad_f = calcular_gradiente(f, x)
        
        # 2. Verificar convergência
        if np.linalg.norm(grad_f) <= tol:
            break
        
        # 3. Tentar atualização com passo atual
        if versao == "entropia_negativa":
            x_new = np.exp(np.log(np.maximum(x, 1e-10)) - eta * grad_f)
        else:  # norma_p
            y = np.sign(x) * (np.abs(x) ** 1) - eta * grad_f  # p=2
            x_new = y
        
        # 4. Verificar se houve melhoria
        fo_new = f(x_new)
        improvement = fo - fo_new
        
        if improvement > 1e-8:
            # Boa melhoria: aceitar e aumentar passo
            x = x_new
            fo = fo_new
            last_improvement = k
            eta = min(eta * 1.1, 1.0)  # Aumentar passo, mas limitar
        else:
            # Pouca melhoria: diminuir passo
            eta = eta * 0.9
            if k - last_improvement > 10:
                eta = max(eta, 1e-6)  # Evitar passo muito pequeno
        
        # 5. Verificar convergência pela mudança no ponto
        if np.linalg.norm(x_new - x) < tol:
            break
    
    return x, fo, k


class MirrorGradientOptimizedSolver:
    """
    Classe para resolver problemas de otimização usando o método otimizado do Gradiente Espelhado
    e gerar tabelas de resultados em formato LaTeX.
    """
    
    def __init__(self, versao="norma_p", eta=0.01, adaptativo=False):
        """
        Inicializa o solver com diferentes versões do gradiente espelhado.
        
        Args:
            versao (str): Tipo de função de distância ('norma_p', 'entropia_negativa')
            eta (float): Parâmetro de passo
            adaptativo (bool): Se True, usa passo adaptativo
        """
        self.results = []
        self.problems_config = setup_problems()
        self.versao = versao
        self.eta = eta
        self.adaptativo = adaptativo
    
    def solve_problem(self, problem_name, max_iter=1000, tol=1e-3):
        """
        Resolve um problema específico usando Gradiente Espelhado Otimizado.
        
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
            if self.adaptativo:
                x_md, fo, iterations = gradiente_espelhado_adaptativo(
                    objective_with_args, x0, self.eta, max_iter, tol, self.versao
                )
            else:
                x_md, fo, iterations = gradiente_espelhado_otimizado(
                    objective_with_args, x0, self.eta, max_iter, tol, self.versao
                )
            
            end_time = time.time()
            
            # Calcular norma do gradiente final
            grad_norm_final = calcular_gradiente(objective_with_args, x_md)
            grad_norm = np.linalg.norm(grad_norm_final)
            
            # Armazenar resultados
            result_dict = {
                'problem': problem_name,
                'success': True,
                'iterations': iterations + 1,
                'function_value': fo,
                'x_value': x_md,
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
        print(f"Iniciando resolução de todos os problemas com Gradiente Espelhado Otimizado ({self.versao})...")
        if self.adaptativo:
            print(f"  - Modo: Adaptativo")
        else:
            print(f"  - Passo fixo: {self.eta}")
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
        print("RESUMO DOS RESULTADOS - GRADIENTE ESPELHADO OTIMIZADO")
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


def compare_mirror_algorithms(problem_name, max_iter=1000, tol=1e-3):
    """
    Compara diferentes versões do algoritmo de gradiente espelhado
    
    Args:
        problem_name (str): Nome do problema a ser testado
        max_iter (int): Número máximo de iterações
        tol (float): Tolerância para convergência
    """
    print(f"\nComparação: Diferentes versões do Gradiente Espelhado - {problem_name}")
    print("=" * 70)
    
    # Configurar problema
    solver_base = MirrorGradientOptimizedSolver(versao="norma_p", eta=0.01, adaptativo=False)
    
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
        # ("norma_p_fixo", "Norma-p (passo fixo)", "norma_p", 0.01, False),
        ("norma_p_adaptativo", "Norma-p (adaptativo)", "norma_p", 0.01, True)
        # ("entropia_fixo", "Entropia negativa (passo fixo)", "entropia_negativa", 0.01, False)
        # ("entropia_adaptativo", "Entropia negativa (adaptativo)", "entropia_negativa", 0.01, True)
    ]
    
    results = {}
    
    for algo_name, description, versao, eta, adaptativo in algorithms:
        print(f"\n{description}:")
        start_time = time.time()
        try:
            if adaptativo:
                x_result, fo, iterations = gradiente_espelhado_adaptativo(
                    objective_with_args, x0, eta, max_iter, tol, versao
                )
            else:
                x_result, fo, iterations = gradiente_espelhado_otimizado(
                    objective_with_args, x0, eta, max_iter, tol, versao
                )
            
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
    print(f"{'Algoritmo':<25} {'Iterações':<10} {'Tempo (s)':<10} {'||∇f||':<15}")
    print("-" * 70)
    
    for algo_name, description, _, _, _ in algorithms:
        if algo_name in results:
            r = results[algo_name]
            if r['success']:
                print(f"{description:<25} {r['iterations']:<10} {r['time']:<10.3f} {r['gradient_norm']:<15.6e}")
            else:
                print(f"{description:<25} {'Falhou':<10} {r['time']:<10.3f} {'---':<15}")


def main():
    """
    Função principal para executar a análise.
    """
    print("=" * 80)
    print("GRADIENTE ESPELHADO OTIMIZADO - COMPARAÇÃO DE VERSÕES")
    print("=" * 80)
    
    versions = [
        ('norma_p_fixo', 'Norma-p (passo fixo)', 'norma_p', 0.01, False),
        ('norma_p_adaptativo', 'Norma-p (adaptativo)', 'norma_p', 0.01, True),
        ('entropia_adaptativo', 'Entropia negativa (adaptativo)', 'entropia_negativa', 0.01, True)
    ]
    
    all_results = {}
    
    for version_name, description, versao, eta, adaptativo in versions:
        print(f"\n{description}")
        print("-" * 60)
        
        # Criar solver com versão específica
        solver = MirrorGradientOptimizedSolver(versao=versao, eta=eta, adaptativo=adaptativo)
        
        # Resolver todos os problemas
        solver.solve_all_problems(max_iter=500, tol=1e-4)
        
        # Imprimir resumo
        solver.print_summary()
        
        # Armazenar resultados
        all_results[version_name] = solver.results
        
        # Gerar tabela LaTeX específica
        filename = f'liu_nocedal/latex_solution/resultados_mirror_gradient_{version_name}.tex'
        method_name = f'Gradiente Espelhado Otimizado ({description})'
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
        
        print(f"{version_name.upper():<20}: {successful}/{total} sucessos, "
              f"tempo médio: {avg_time:.3f}s, iterações médias: {avg_iterations:.1f}")
    
    print("\nAnálise concluída!")


if __name__ == "__main__":
    main()
