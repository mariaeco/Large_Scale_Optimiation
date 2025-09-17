
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

def gradiente_espelhado(f, x0, eta, max_iter, tol, phi, phi_grad):
    """
    Pseudo-algoritmo do Gradiente Espelhado (Mirror Descent)
    
    Parâmetros:
    - f: função objetivo
    - x0: ponto inicial
    - eta: parâmetro de passo
    - max_iter: máximo de iterações
    - tol: tolerância para convergência
    - phi: função de distância (ex: ||x||²/2)
    - phi_grad: gradiente de phi
    """
    
    x = x0
    historico = [x]
    fo = f(x0)  # Inicializar fo com o valor inicial
    
    for k in range(max_iter):
        # 1. Calcular gradiente da função objetivo
        grad_f = calcular_gradiente(f, x)
        
        # 2. Atualizar usando operador proximal
        # x_{k+1} = argmin_x {⟨grad_f, x⟩ + (1/eta) * D_phi(x, x_k)}
        x_novo, f_novo = resolver_subproblema(grad_f, x, eta, phi, phi_grad)
        
        # 3. Verificar convergência
        if np.linalg.norm(x_novo - x) < tol:
            x = x_novo
            fo = f(x)  # Calcular valor final da função
            break
   
        x = x_novo
        fo = f(x)  # Calcular valor da função no novo ponto
        historico.append(x)
    
    return x, fo, k

def resolver_subproblema(grad_f, x_atual, eta, phi, phi_grad):
    """
    Resolve o subproblema de minimização
    min_x {⟨grad_f, x⟩ + (1/eta) * D_phi(x, x_atual)}
    """
    
    def funcao_subproblema(x):
        # Divergência de Bregman: D_phi(x, y) = phi(x) - phi(y) - ⟨∇phi(y), x-y⟩
        bregman_div = phi(x) - phi(x_atual) - dot(phi_grad(x_atual), x - x_atual)
        return dot(grad_f, x) + (1/eta) * bregman_div
    
    # Resolver usando otimizador (ex: scipy.optimize.minimize)
    resultado = minimize(funcao_subproblema, x0=x_atual, method='L-BFGS-B')
    
    # resultado.
    return resultado.x, resultado.fun


    # Exemplos de funções de distância
def phi_euclidiana(x):
    """Função de distância euclidiana: ||x||²/2"""
    return 0.5 * sum(x[i]**2 for i in range(len(x)))

def phi_grad_euclidiana(x):
    """Gradiente da função euclidiana: x"""
    return x

def phi_entropia(x):
    """Função de entropia: sum(x[i] * log(x[i]))"""
    return sum(x[i] * log(x[i]) for i in range(len(x)))

def phi_grad_entropia(x):
    """Gradiente da entropia: log(x) + 1"""
    return [np.log(x[i]) + 1 for i in range(len(x))]


    
class MirrorGradientSolver:
    """
    Classe para resolver problemas de otimização usando o método do Gradiente Espelhado (Mirror Descent)
    e gerar tabelas de resultados em formato LaTeX.
    """
    
    def __init__(self):
        self.results = []
        self.problems_config = setup_problems()

    
    def solve_problem(self, problem_name, max_iter=1000, eta=0.01, tol=1e-3):
        """
        Resolve um problema específico usando Gradiente Espelhado.
        
        Args:
            problem_name (str): Nome do problema
            max_iter (int): Número máximo de iterações
            eta (float): Parâmetro de passo
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
            x_md, fo, iterations = gradiente_espelhado(objective_with_args, x0, eta, max_iter, tol,
                                         phi_euclidiana, phi_grad_euclidiana)
            end_time = time.time()
            
            # Calcular norma do gradiente final
            grad_norm_final = calcular_gradiente(objective_with_args, x_md)
            grad_norm = np.linalg.norm(grad_norm_final)
            
            
            # Armazenar resultados
            result_dict = {
                'problem': problem_name,
                'success': True,
                'iterations': iterations+1,
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
    
    def solve_all_problems(self, max_iter=1000, eta=0.01, tol=1e-3):
        """
        Resolve todos os problemas configurados.
        
        Args:
            max_iter (int): Número máximo de iterações
            eta (float): Parâmetro de passo
            tol (float): Tolerância para convergência
        """
        print("Iniciando resolução de todos os problemas com Gradiente Espelhado...")
        print("=" * 70)
        
        self.results = []
        
        for i, problem_name in enumerate(self.problems_config.keys(), 1):
            print(f"\n[{i}/{len(self.problems_config)}] Resolvendo: {problem_name}")
            
            result = self.solve_problem(problem_name, max_iter, eta, tol)
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
        print("RESUMO DOS RESULTADOS - GRADIENTE ESPELHADO")
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
    # Criar solver
    solver = MirrorGradientSolver()
    
    # Resolver todos os problemas
    solver.solve_all_problems()
    
    # Imprimir resumo
    solver.print_summary()
    
    # Gerar tabela LaTeX
    filename = f'liu_nocedal/latex_solution/resultados_mirror_gradient.tex'
    generate_latex_table(solver.results, filename, 'Gradiente Espelhado')
    salvar_pdf(filename, 'liu_nocedal/latex_solution/')
    
    print("\nAnálise concluída!")

if __name__ == "__main__":
    main()