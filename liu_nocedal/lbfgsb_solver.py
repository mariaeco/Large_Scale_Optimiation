# Classe para resolver problemas de otimização usando L-BFGS-B e gerar tabela LaTeX para problemas de Liu e Nocedal
import numpy as np
from scipy.optimize import minimize
from problems.setup_problems import*
import time
import os
from datetime import datetime
from latex_to_pdf import salvar_pdf, generate_latex_table, generate_detailed_latex_table

class LBFGSBSolver:
    """
    Classe para resolver problemas de otimização usando o método L-BFGS-B
    e gerar tabelas de resultados em formato LaTeX.
    """
    
    def __init__(self):
        self.results = []
        self.problems_config = setup_problems()
 
    
    def solve_problem(self, problem_name, max_iter=1000, ftol=1e-6):
        """
        Resolve um problema específico usando L-BFGS-B.
        
        Args:
            problem_name (str): Nome do problema
            max_iter (int): Número máximo de iterações
            ftol (float): Tolerância para convergência
            
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
        else:
            n, x0 = config['setup']()
            args = config['args']

        
        # Configurar opções
        options = {
            'maxiter': max_iter,
            'ftol': ftol,
            'disp': False
        }
        
        # Executar otimização
        start_time = time.time()
        
        if problem_name == 'ULTS0':
            result = minimize(
                fun=config['objective'],
                x0=x0,
                args=args,
                method='L-BFGS-B',
                bounds=config['bounds'],
                options=options
            )
        else:
            result = minimize(
                fun=config['objective'],
                x0=x0,
                method='L-BFGS-B',
                bounds=config['bounds'],
                options=options
            )
        
        end_time = time.time()
        
        # Armazenar resultados
        result_dict = {
            'problem': problem_name,
            'success': result.success,
            'iterations': result.nit+1,
            'function_value': result.fun,
            'x_value': result.x,
            'gradient_norm': self._compute_gradient_norm(problem_name ,config['objective'], result.x),
            'message': result.message,
            'execution_time': end_time - start_time,
            'n_variables': n
        }
        
        return result_dict
    
    def _compute_gradient_norm(self, problem_name, objective_func, x, h=1e-8):
        """
        Calcula a norma do gradiente numericamente para medir a precisão.
        
        Args:
            objective_func: função objetivo
            x: ponto onde calcular o gradiente
            h: passo para diferenças finitas
            
        Returns:
            norma do gradiente
        """
        try:
            n = len(x)
            grad = np.zeros(n)
            
            for i in range(n):
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                
                grad[i] = (objective_func(x_plus) - objective_func(x_minus)) / (2 * h)
            
            return np.linalg.norm(grad)
        except:
            return float('inf')
    
    def solve_all_problems(self, max_iter=1000, ftol=1e-6):
        """
        Resolve todos os problemas configurados.
        
        Args:
            max_iter (int): Número máximo de iterações
            ftol (float): Tolerância para convergência
        """
        print("Iniciando resolução de todos os problemas...")
        print("=" * 60)
        
        self.results = []
        
        for i, problem_name in enumerate(self.problems_config.keys(), 1):
            print(f"\n[{i}/{len(self.problems_config)}] Resolvendo: {problem_name}")
            
            result = self.solve_problem(problem_name, max_iter, ftol)
            self.results.append(result)
            
            # Imprimir resultado
            if result['success']:
                print(f"  ✓ Sucesso: {result['iterations']} iterações, f* = {result['function_value']:.6e}")
            else:
                print(f"  ✗ Falhou: {result['message']}")

        print("\n" + "=" * 60)
        print("Resolução concluída!")

     
    

    
    def print_summary(self):
        """
        Imprime um resumo dos resultados.
        """
        if not self.results:
            print("Nenhum resultado disponível.")
            return
        
        print("\n" + "=" * 80)
        print("RESUMO DOS RESULTADOS")
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
        print(f"{'Problema':<25} {'N Variáveis':<15} {'Iterações':<10} {'Valor Mínimo':<15} {'Tempo (s)':<10}")
        print("-" * 95)
        
        for result in self.results:
            status = "✓" if result['success'] else "✗"
            iterations = str(result['iterations']) if result['success'] else "---"
            n_variables = str(result['n_variables'])
            value = f"{result['function_value']:.6e}" if result['success'] else "Falhou"
            time_str = f"{result['execution_time']:.3f}" if result['success'] else "---"
            
            print(f"{result['problem']:<25} {n_variables:<15} {iterations:<10} {value:<15} {time_str:<10}")


def main():
    """
    Função principal para executar a análise.
    """
    # Criar solver
    solver = LBFGSBSolver()
    
    # Resolver todos os problemas
    solver.solve_all_problems()
    
    # Imprimir resumo
    solver.print_summary()
    
    # Gerar tabela LaTeX
    filename = f'liu_nocedal/latex_solution/resultados_lbfgsb.tex'
    # generate_latex_table(solver.results, filename, 'L-BFGS-B')
    # salvar_pdf(filename, 'liu_nocedal/latex_solution/')
    
    # Gerar tabela detalhada com valores das variáveis
    detailed_filename = f'liu_nocedal/latex_solution/resultados_lbfgsb.tex'
    generate_detailed_latex_table(solver.results, detailed_filename, 'L-BFGS-B')
    salvar_pdf(detailed_filename, 'liu_nocedal/latex_solution/')
    
    print("\nAnálise concluída!")


if __name__ == "__main__":
    main()
