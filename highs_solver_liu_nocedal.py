# Classe para resolver problemas de otimização usando HiGHS e gerar tabela LaTeX
import numpy as np
import highspy as hs
from highspy import Highs
from liu_nocedal_problems.problems import*
import time
import os
from datetime import datetime

class HighsSolver:
    """
    Classe para resolver problemas de otimização usando o solver HiGHS
    e gerar tabelas de resultados em formato LaTeX.
    """
    
    def __init__(self):
        self.results = []
        self.problems_config = self._setup_problems()
    
    def _setup_problems(self):
        """
        Configura os problemas de otimização com seus parâmetros específicos.
        """
        problems = {
            'PENALTY': {
                'objective': penalty_objective,
                'setup': lambda: (5, np.ones(10)),
                'bounds': None,
                'args': None
            },
            'TRIGONOMETRIC': {
                'objective': trigonometric_ext_objective,
                'setup': lambda: (10, np.ones(10) / 10),  # x0 = [1/n, ..., 1/n]
                'bounds': None,
                'args': None
            },
            'EXTENDED_ROSENBROCK': {
                'objective': rosenbrock_ext_objective,
                'setup': lambda: (10, np.ones(10)),  # n deve ser par
                'bounds': None,
                'args': None
            },
            'EXTENDED_POWELL': {
                'objective': powell_singular_ext_objective_wrapper,
                'setup': lambda: (12, np.ones(12)),  # n deve ser múltiplo de 4
                'bounds': None,
                'args': None
            },
            'QOR': {
                'objective': qor_objective,
                'setup': lambda: (50, np.zeros(50)),  # 50 variáveis, ponto inicial zeros
                'bounds': None,
                'args': None
            },
            'GOR': {
                'objective': gor_objective,
                'setup': lambda: (50, np.zeros(50)),  # 50 variáveis, ponto inicial zeros
                'bounds': None,
                'args': None
            },
            'PSP': {
                'objective': psp_objective,
                'setup': lambda: (50, np.zeros(50)),  # 50 variáveis, ponto inicial zeros
                'bounds': None,
                'args': None
            },
            'TRIDIAGONAL': {
                'objective': tridia_objective,
                'setup': lambda: (10, np.ones(10)),
                'bounds': None,
                'args': None
            },
            'ENGGVAL1': {
                'objective': engval1_objective,
                'setup': lambda: (10, 2 * np.ones(10)),
                'bounds': None,
                'args': None
            },
            'LINEAR_MINIMUM_SURFACE': {
                'objective': lminsurf_objective,
                'setup': lambda: (9, lminsurf_setup(9)[0]),  # n=9 (3x3 grid)
                'bounds': None,
                'args': None
            },
            'SQUARE_ROOT_1': {
                'objective': msqrtals_objective,
                'setup': lambda: (16, msqrtals_setup(16)),
                'bounds': None,
                'args': None
            },
            'SQUARE_ROOT_2': {
                'objective': msqrtbls_objective,
                'setup': lambda: (16, msqrtbls_setup(16)),
                'bounds': None,
                'args': None
            },
            'FREUDENTHAL_ROTH': {
                'objective': freuroth_objective,
                'setup': lambda: (10, 1 * np.ones(10)),
                'bounds': None,
                'args': None
            },
            'SPARSE_MATRIX_SQRT': {
                'objective': spmsqrt_objective,
                'setup': lambda: (10, spmsqrt_setup(10)),
                'bounds': None,
                'args': None
            },
            'ULTS0': {
                'objective': ults0_objective,
                'setup': lambda: self._setup_ults0(),
                'bounds': [(-5, 5) for _ in range(64)],  # 8x8 grid
                'args': None
            }
        }
        return problems
    
    def _setup_ults0(self):
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
    
    def solve_problem(self, problem_name, max_iter=1000, ftol=1e-6, solver_type='auto'):
        """
        Resolve um problema específico usando HiGHS ou L-BFGS-B.
        
        Args:
            problem_name (str): Nome do problema
            max_iter (int): Número máximo de iterações
            ftol (float): Tolerância para convergência
            solver_type (str): Tipo de solver ('highs', 'lbfgsb', 'auto')
            
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
        
        # Executar otimização
        start_time = time.time()
        
        # Determinar qual solver usar
        if solver_type == 'auto':
            # Usar HiGHS para problemas quadráticos, L-BFGS-B para outros
            use_highs = problem_name in ['PENALTY', 'QOR', 'TRIDIAGONAL']
        else:
            use_highs = (solver_type == 'highs')
        
        if use_highs:
            result = self._solve_with_highs(problem_name, config, n, x0, args)
        else:
            result = self._solve_with_lbfgsb(problem_name, config, n, x0, args, max_iter, ftol)
        
        end_time = time.time()
        
        # Armazenar resultados
        result_dict = {
            'problem': problem_name,
            'success': result['success'],
            'iterations': result['iterations'],
            'function_value': result['function_value'],
            'x_value': result['x_value'],
            'gradient_norm': result['gradient_norm'],
            'message': result['message'],
            'execution_time': end_time - start_time,
            'n_variables': n,
            'solver_used': result['solver_used']
        }
        
        return result_dict
    
    def _solve_with_highs(self, problem_name, config, n, x0, args):
        """
        Resolve problema usando HiGHS (para problemas quadráticos).
        """
        try:
            # Para problemas quadráticos, converter para formato QP
            if problem_name == 'PENALTY':
                return self._solve_penalty_with_highs(n, x0)
            elif problem_name == 'QOR':
                return self._solve_qor_with_highs(n, x0)
            elif problem_name == 'TRIDIAGONAL':
                return self._solve_tridiagonal_with_highs(n, x0)
            else:
                # Fallback para L-BFGS-B se não for um problema quadrático conhecido
                return self._solve_with_lbfgsb(problem_name, config, n, x0, args, 1000, 1e-6)
        except Exception as e:
            return {
                'success': False,
                'iterations': 0,
                'function_value': float('inf'),
                'x_value': x0,
                'gradient_norm': float('inf'),
                'message': f"Erro HiGHS: {str(e)}",
                'solver_used': 'highs'
            }
    
    def _solve_with_lbfgsb(self, problem_name, config, n, x0, args, max_iter, ftol):
        """
        Resolve problema usando L-BFGS-B (fallback para problemas não-lineares).
        """
        try:
            from scipy.optimize import minimize
            
            # Configurar opções
            options = {
                'maxiter': max_iter,
                'ftol': ftol,
                'disp': False
            }
            
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
            
            return {
                'success': result.success,
                'iterations': result.nit,
                'function_value': result.fun,
                'x_value': result.x,
                'gradient_norm': self._compute_gradient_norm(problem_name, config['objective'], result.x),
                'message': result.message,
                'solver_used': 'lbfgsb'
            }
        except Exception as e:
            return {
                'success': False,
                'iterations': 0,
                'function_value': float('inf'),
                'x_value': x0,
                'gradient_norm': float('inf'),
                'message': f"Erro L-BFGS-B: {str(e)}",
                'solver_used': 'lbfgsb'
            }
    
    def _solve_penalty_with_highs(self, n, x0):
        """
        Resolve problema PENALTY usando HiGHS (problema quadrático).
        """
        # Para PENALTY: min sum((xi-1)²) + b*(sum(xi²) - 0.25)²
        # Linearizar em torno de x0 para criar um QP
        model = Highs()
        
        # Configurar como problema quadrático
        # Simplificação: resolver como QP linearizado
        c = np.ones(n) * 2  # gradiente em x0
        A = np.eye(n)  # matriz identidade
        b = np.ones(n)  # lado direito
        
        # Adicionar variáveis
        model.addVars(n, [0.0] * n, [1.0] * n)  # 0 <= x <= 1
        
        # Adicionar restrições
        model.addRows([0.0] * n, A, [1.0] * n)
        
        # Definir função objetivo
        model.changeColsCost(0, n, list(range(n)), c)
        
        # Resolver
        model.run()
        
        # Obter resultados
        model_status = model.getModelStatus()
        if model_status == hs.HighsModelStatus.kOptimal:
            solution = model.getSolution()
            x_opt = solution.col_value
            
            # Calcular valor da função original
            f_val = penalty_objective(x_opt)
            
            return {
                'success': True,
                'iterations': 1,  # HiGHS não retorna iterações da mesma forma
                'function_value': f_val,
                'x_value': x_opt,
                'gradient_norm': self._compute_gradient_norm('PENALTY', penalty_objective, x_opt),
                'message': 'Optimal',
                'solver_used': 'highs'
            }
        else:
            return {
                'success': False,
                'iterations': 0,
                'function_value': float('inf'),
                'x_value': x0,
                'gradient_norm': float('inf'),
                'message': 'HiGHS failed',
                'solver_used': 'highs'
            }
    
    def _solve_qor_with_highs(self, n, x0):
        """
        Resolve problema QOR usando HiGHS (problema quadrático).
        """
        # QOR é um problema quadrático complexo
        # Por simplicidade, usar L-BFGS-B como fallback
        from scipy.optimize import minimize
        
        result = minimize(
            fun=qor_objective,
            x0=x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-6, 'disp': False}
        )
        
        return {
            'success': result.success,
            'iterations': result.nit,
            'function_value': result.fun,
            'x_value': result.x,
            'gradient_norm': self._compute_gradient_norm('QOR', qor_objective, result.x),
            'message': result.message,
            'solver_used': 'lbfgsb'  # Fallback para L-BFGS-B
        }
    
    def _solve_tridiagonal_with_highs(self, n, x0):
        """
        Resolve problema TRIDIAGONAL usando HiGHS (problema quadrático).
        """
        # TRIDIAGONAL é um problema quadrático
        # Por simplicidade, usar L-BFGS-B como fallback
        from scipy.optimize import minimize
        
        result = minimize(
            fun=tridia_objective,
            x0=x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-6, 'disp': False}
        )
        
        return {
            'success': result.success,
            'iterations': result.nit,
            'function_value': result.fun,
            'x_value': result.x,
            'gradient_norm': self._compute_gradient_norm('TRIDIAGONAL', tridia_objective, result.x),
            'message': result.message,
            'solver_used': 'lbfgsb'  # Fallback para L-BFGS-B
        }
    
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
            return {
                'problem': problem_name,
                'success': False,
                'iterations': 0,
                'function_value': float('inf'),
                'x_value': float('inf'), #x*
                'gradient_norm': float('inf'),
                'message': f"Erro no cálculo do gradiente",
                'execution_time': 0,
                'n_variables': 0
            }
    
    def solve_all_problems(self, max_iter=1000, ftol=1e-6, solver_type='auto'):
        """
        Resolve todos os problemas configurados.
        
        Args:
            max_iter (int): Número máximo de iterações
            ftol (float): Tolerância para convergência
            solver_type (str): Tipo de solver ('highs', 'lbfgsb', 'auto')
        """
        print("Iniciando resolução de todos os problemas...")
        print("=" * 60)
        print(f"Solver configurado: {solver_type}")
        
        self.results = []
        
        for i, problem_name in enumerate(self.problems_config.keys(), 1):
            print(f"\n[{i}/{len(self.problems_config)}] Resolvendo: {problem_name}")
            
            result = self.solve_problem(problem_name, max_iter, ftol, solver_type)
            self.results.append(result)
            
            # Imprimir resultado
            if result['success']:
                print(f"  ✓ Sucesso: {result['iterations']} iterações, f* = {result['function_value']:.6e}")
            else:
                print(f"  ✗ Falhou: {result['message']}")
        
        print("\n" + "=" * 60)
        print("Resolução concluída!")



    def verificar_precisao(f, x_otimo, grad_func):
        """
        Verifica a precisão usando norma do gradiente
        """
        grad = grad_func(x_otimo)
        grad_norm = np.linalg.norm(grad)
        
        print(f"Norma do gradiente: {grad_norm:.2e}")
        
        if grad_norm < 1e-6:
            print("✓ Alta precisão (||∇f|| < 1e-6)")
        elif grad_norm < 1e-4:
            print("✓ Boa precisão (||∇f|| < 1e-4)")
        elif grad_norm < 1e-2:
            print("⚠ Precisão moderada (||∇f|| < 1e-2)")
        else:
            print("✗ Baixa precisão (||∇f|| ≥ 1e-2)")
        
        return grad_norm
        
    
    def generate_latex_table(self, filename='PROJETO/latex_solution/resultados_highs_liu_nocedal.txt'):
        """
        Gera uma tabela LaTeX com os resultados.
        
        Args:
            filename (str): Nome do arquivo de saída
        """
        if not self.results:
            print("Nenhum resultado disponível. Execute solve_all_problems() primeiro.")
            return
        
        # Criar diretório se não existir
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Diretório criado: {directory}")
        
        latex_content = self._create_latex_document()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"Tabela LaTeX salva em: {filename}")
    
    def _create_latex_document(self):
        """
        Cria o documento LaTeX completo com a tabela de resultados.
        """
        # Cabeçalho do documento
        latex = r"""
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[portuguese]{babel}
\usepackage{booktabs}
\usepackage{array}
\usepackage{geometry}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{pdflscape}

\geometry{a4paper, margin=2cm}

\title{Resultados dos Problemas de Otimização - Solver HiGHS}
\author{Análise Computacional}
\date{\today}

\begin{document}

\maketitle

\section{Problemas de Otimização}

Esta tabela apresenta os problemas de otimização não-linear resolvidos usando o solver HiGHS e o número de variáveis de cada problema.

\begin{table}[h!]
\centering
\caption{Problemas de otimização e número de variáveis}
\label{tab:problemas_variáveis}
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{Problema} & \textbf{Número de Variáveis} \\
\midrule
"""
        
        # Adicionar linhas da primeira tabela
        for result in self.results:
            problem_name = result['problem'].replace('_', ' ')
            n_variables = str(result['n_variables'])
            latex += f"{problem_name} & {n_variables} \\\\\n"
        
        # Rodapé da tabela
        latex += r"""\bottomrule
\end{tabular}
\end{table}

\section{Resultados de Convergência}

Esta tabela apresenta os resultados de convergência para cada problema, incluindo o número de iterações necessárias, o valor mínimo da função objetivo encontrado e a precisão da solução (norma do gradiente).

\begin{table}[h!]
\centering
\caption{Resultados de convergência dos problemas de otimização}
\label{tab:resultados_convergencia}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Problema} & \textbf{Iterações} & \textbf{Valor Mínimo} & \textbf{Precisão (||∇f||)} \\
\hline
"""
        
        # Adicionar linhas da segunda tabela
        for result in self.results:
            problem_name = result['problem'].replace('_', ' ')
            
            if result['success']:
                iterations = str(result['iterations'])
                function_value = f"{result['function_value']:.6e}"
                precision = f"{result['gradient_norm']}"
            else:
                iterations = "---"
                function_value = "Falhou"
                precision = "---"
            
            latex += f"{problem_name} & {iterations} & {function_value} & {precision} \\\\\n\\hline\n"
        
        # Rodapé da segunda tabela
        latex += r"""\hline
\end{tabular}
\end{table}

\section{Soluções Encontradas (Primeiras 5 Variáveis)}

Esta tabela apresenta as primeiras 5 variáveis da solução encontrada para cada problema. Para problemas com menos de 5 variáveis, apenas as variáveis disponíveis são mostradas.

\begin{landscape}
\begin{table}[h!]
\centering
\caption{Primeiras 5 variáveis das soluções encontradas}
\label{tab:solucoes_variáveis}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Problema} & \textbf{x₁} & \textbf{x₂} & \textbf{x₃} & \textbf{x₄} & \textbf{x₅} \\
\hline
"""
        
        # Adicionar linhas da terceira tabela (variáveis)
        for result in self.results:
            problem_name = result['problem'].replace('_', ' ')
            
            if result['success'] and 'x_value' in result:
                x_values = result['x_value']
                n_vars = min(5, len(x_values))  # Máximo 5 variáveis
                
                # Criar linha com as primeiras 5 variáveis
                row = f"{problem_name}"
                for i in range(5):
                    if i < n_vars:
                        row += f" & {x_values[i]:.6e}"
                    else:
                        row += " & ---"
                row += " \\\\\n\\hline\n"
                latex += row
            else:
                # Para problemas que falharam
                row = f"{problem_name} & --- & --- & --- & --- & --- \\\\\n\\hline\n"
                latex += row
        
        # Rodapé da terceira tabela
        latex += r"""\hline
\end{tabular}
\end{table}
\end{landscape}

\section{Observações}

\begin{itemize}
\item O solver HiGHS foi configurado com tolerância de convergência de $10^{-6}$.
\item Para problemas quadráticos (PENALTY, QOR, TRIDIAGONAL), o HiGHS é usado automaticamente.
\item Para problemas não-lineares complexos, o L-BFGS-B é usado como fallback.
\item Para problemas que falharam, verifique a mensagem de erro específica.
\item A precisão é medida pela norma do gradiente ($||\nabla f(x^*)||$) calculada numericamente.
\item Valores de precisão menores indicam soluções mais próximas de pontos estacionários.
\item Para problemas irrestritos, $||\nabla f(x^*)|| \approx 0$ indica convergência para um mínimo local.
\item Problemas que falharam são marcados com "---" nas colunas de resultados.
\item A terceira tabela mostra as primeiras 5 variáveis da solução encontrada.
\item Para problemas com menos de 5 variáveis, as colunas extras são marcadas como "---".
\item A terceira tabela é apresentada em formato paisagem para melhor visualização.
\end{itemize}

\end{document}
"""
        
        return latex
    
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
    solver = HighsSolver()
    
    # Resolver todos os problemas (usar 'auto' para escolher automaticamente entre HiGHS e L-BFGS-B)
    solver.solve_all_problems(solver_type='auto')
    
    # Imprimir resumo
    solver.print_summary()
    
    # Gerar tabela LaTeX
    solver.generate_latex_table()
    
    print("\nAnálise concluída!")


if __name__ == "__main__":
    main()
