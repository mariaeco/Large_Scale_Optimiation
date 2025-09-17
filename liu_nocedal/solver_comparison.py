"""
Comparador de Métodos de Otimização
Compara L-BFGS-B, Mirror Gradient e Coordinate Descent nos problemas de Liu e Nocedal
"""

import numpy as np
import time
import os
from datetime import datetime

# Importar os solvers
from lbfgsb_solver_liu_nocedal import LBFGSBSolver
from mirror_gradient_algorith import MirrorGradientSolver
from descent_coordinate_algorithm import CoordinateDescentSolver


class ComparisonSolver:
    """
    Classe para comparar os três métodos de otimização e gerar tabelas comparativas.
    """
    
    def __init__(self):
        self.lbfgs_solver = LBFGSBSolver()
        self.mirror_solver = MirrorGradientSolver()
        self.coordinate_solver = CoordinateDescentSolver()
        self.comparison_results = []
    
    def run_all_methods(self, max_iter=1000, tol=1e-6):
        """
        Executa todos os três métodos nos problemas de Liu e Nocedal.
        
        Args:
            max_iter (int): Número máximo de iterações
            tol (float): Tolerância para convergência
        """
        print("=" * 80)
        print("COMPARAÇÃO DE MÉTODOS DE OTIMIZAÇÃO")
        print("=" * 80)
        print("Métodos: L-BFGS-B, Mirror Gradient, Coordinate Descent")
        print("Problemas: Liu e Nocedal (16 problemas)")
        print("=" * 80)
        
        # Obter lista de problemas (usar a mesma ordem para todos)
        problem_names = list(self.lbfgs_solver.problems_config.keys())
        
        self.comparison_results = []
        
        for i, problem_name in enumerate(problem_names, 1):
            print(f"\n[{i}/{len(problem_names)}] Comparando problema: {problem_name}")
            print("-" * 60)
            
            problem_results = {
                'problem': problem_name,
                'lbfgs': None,
                'mirror': None,
                'coordinate': None
            }
            
            # Executar L-BFGS-B
            print("  Executando L-BFGS-B...")
            try:
                lbfgs_result = self.lbfgs_solver.solve_problem(problem_name, max_iter, tol)
                problem_results['lbfgs'] = lbfgs_result
                status = "✓" if lbfgs_result['success'] else "✗"
                print(f"    {status} L-BFGS-B: {lbfgs_result['iterations']} iter, f* = {lbfgs_result['function_value']:.6e}")
            except Exception as e:
                print(f"    ✗ L-BFGS-B falhou: {str(e)}")
                problem_results['lbfgs'] = {
                    'success': False,
                    'iterations': 0,
                    'function_value': float('inf'),
                    'execution_time': 0,
                    'gradient_norm': float('inf'),
                    'message': f"Erro: {str(e)}"
                }
            
            # Executar Mirror Gradient
            print("  Executando Mirror Gradient...")
            try:
                mirror_result = self.mirror_solver.solve_problem(problem_name, max_iter, eta=0.01, tol=tol)
                problem_results['mirror'] = mirror_result
                status = "✓" if mirror_result['success'] else "✗"
                print(f"    {status} Mirror: {mirror_result['iterations']} iter, f* = {mirror_result['function_value']:.6e}")
            except Exception as e:
                print(f"    ✗ Mirror Gradient falhou: {str(e)}")
                problem_results['mirror'] = {
                    'success': False,
                    'iterations': 0,
                    'function_value': float('inf'),
                    'execution_time': 0,
                    'gradient_norm': float('inf'),
                    'message': f"Erro: {str(e)}"
                }
            
            # Executar Coordinate Descent
            print("  Executando Coordinate Descent...")
            try:
                coord_result = self.coordinate_solver.solve_problem(problem_name, max_iter, tol)
                problem_results['coordinate'] = coord_result
                status = "✓" if coord_result['success'] else "✗"
                print(f"    {status} Coordinate: {coord_result['iterations']} iter, f* = {coord_result['function_value']:.6e}")
            except Exception as e:
                print(f"    ✗ Coordinate Descent falhou: {str(e)}")
                problem_results['coordinate'] = {
                    'success': False,
                    'iterations': 0,
                    'function_value': float('inf'),
                    'execution_time': 0,
                    'gradient_norm': float('inf'),
                    'message': f"Erro: {str(e)}"
                }
            
            self.comparison_results.append(problem_results)
        
        print("\n" + "=" * 80)
        print("COMPARAÇÃO CONCLUÍDA!")
        print("=" * 80)
    
    def generate_comparison_latex(self, filename='PROJETO/latex_solution/comparacao_metodos.tex'):
        """
        Gera uma tabela LaTeX comparando os três métodos.
        
        Args:
            filename (str): Nome do arquivo de saída
        """
        if not self.comparison_results:
            print("Nenhum resultado disponível. Execute run_all_methods() primeiro.")
            return
        
        # Criar diretório se não existir
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Diretório criado: {directory}")
        
        latex_content = self._create_comparison_latex_document()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"Tabela comparativa LaTeX salva em: {filename}")
    
    def _create_comparison_latex_document(self):
        """
        Cria o documento LaTeX completo com as tabelas comparativas.
        """
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
\usepackage{multirow}
\usepackage{xcolor}

\geometry{a4paper, margin=1.5cm}

\title{Comparação de Métodos de Otimização - Liu e Nocedal}
\author{Análise Computacional Comparativa}
\date{\today}

\begin{document}

\maketitle

\section{Resumo Executivo}

Este documento apresenta uma comparação abrangente entre três métodos de otimização aplicados aos problemas de Liu e Nocedal:

\begin{itemize}
    \item \textbf{L-BFGS-B:} Método quasi-Newton com restrições de caixa
    \item \textbf{Mirror Gradient:} Método do gradiente espelhado com divergência de Bregman
    \item \textbf{Coordinate Descent:} Método de descida por coordenadas
\end{itemize}

\section{Comparação de Convergência}

A tabela abaixo apresenta os resultados de convergência para cada método, incluindo número de iterações, valor mínimo da função objetivo, precisão (norma do gradiente) e tempo de execução.

\begin{landscape}
\begin{table}[h!]
\centering
\caption{Comparação de convergência dos métodos de otimização}
\label{tab:comparacao_convergencia}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline
\multirow{2}{*}{\textbf{Problema}} & \multicolumn{4}{c|}{\textbf{L-BFGS-B}} & \multicolumn{4}{c|}{\textbf{Mirror Gradient}} & \multicolumn{4}{c|}{\textbf{Coordinate Descent}} \\
\cline{2-13}
& \textbf{Iter} & \textbf{f*} & \textbf{Precisão} & \textbf{Tempo} & \textbf{Iter} & \textbf{f*} & \textbf{Precisão} & \textbf{Tempo} & \textbf{Iter} & \textbf{f*} & \textbf{Precisão} & \textbf{Tempo} \\
\hline
"""
        
        # Adicionar linhas da tabela comparativa
        for result in self.comparison_results:
            problem_name = result['problem'].replace('_', ' ')
            
            # L-BFGS-B
            lbfgs = result['lbfgs']
            if lbfgs and lbfgs['success']:
                lbfgs_iter = str(lbfgs['iterations'])
                lbfgs_f = f"{lbfgs['function_value']:.2e}"
                lbfgs_prec = f"{lbfgs['gradient_norm']:.2e}"
                lbfgs_time = f"{lbfgs['execution_time']:.3f}"
            else:
                lbfgs_iter = "---"
                lbfgs_f = "Falhou"
                lbfgs_prec = "---"
                lbfgs_time = "---"
            
            # Mirror Gradient
            mirror = result['mirror']
            if mirror and mirror['success']:
                mirror_iter = str(mirror['iterations'])
                mirror_f = f"{mirror['function_value']:.2e}"
                mirror_prec = f"{mirror['gradient_norm']:.2e}"
                mirror_time = f"{mirror['execution_time']:.3f}"
            else:
                mirror_iter = "---"
                mirror_f = "Falhou"
                mirror_prec = "---"
                mirror_time = "---"
            
            # Coordinate Descent
            coord = result['coordinate']
            if coord and coord['success']:
                coord_iter = str(coord['iterations'])
                coord_f = f"{coord['function_value']:.2e}"
                coord_prec = f"{coord['gradient_norm']:.2e}"
                coord_time = f"{coord['execution_time']:.3f}"
            else:
                coord_iter = "---"
                coord_f = "Falhou"
                coord_prec = "---"
                coord_time = "---"
            
            latex += f"{problem_name} & {lbfgs_iter} & {lbfgs_f} & {lbfgs_prec} & {lbfgs_time} & {mirror_iter} & {mirror_f} & {mirror_prec} & {mirror_time} & {coord_iter} & {coord_f} & {coord_prec} & {coord_time} \\\\\n\\hline\n"
        
        latex += r"""\hline
\end{tabular}%
}
\end{table}
\end{landscape}

\section{Análise de Performance}

\subsection{Taxa de Sucesso}

A tabela abaixo apresenta a taxa de sucesso de cada método:

\begin{table}[h!]
\centering
\caption{Taxa de sucesso por método}
\label{tab:taxa_sucesso}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Método} & \textbf{Problemas Resolvidos} & \textbf{Taxa de Sucesso} \\
\hline
"""
        
        # Calcular taxas de sucesso
        total_problems = len(self.comparison_results)
        
        lbfgs_success = sum(1 for r in self.comparison_results if r['lbfgs'] and r['lbfgs']['success'])
        mirror_success = sum(1 for r in self.comparison_results if r['mirror'] and r['mirror']['success'])
        coord_success = sum(1 for r in self.comparison_results if r['coordinate'] and r['coordinate']['success'])
        
        lbfgs_rate = (lbfgs_success / total_problems) * 100
        mirror_rate = (mirror_success / total_problems) * 100
        coord_rate = (coord_success / total_problems) * 100
        
        latex += f"L-BFGS-B & {lbfgs_success}/{total_problems} & {lbfgs_rate:.1f}\\% \\\\\n"
        latex += f"Mirror Gradient & {mirror_success}/{total_problems} & {mirror_rate:.1f}\\% \\\\\n"
        latex += f"Coordinate Descent & {coord_success}/{total_problems} & {coord_rate:.1f}\\% \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\end{table}

\subsection{Estatísticas de Performance}

\begin{table}[h!]
\centering
\caption{Estatísticas de performance dos métodos}
\label{tab:estatisticas}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Método} & \textbf{Iterações Médias} & \textbf{Tempo Médio (s)} & \textbf{Precisão Média} \\
\hline
"""
        
        # Calcular estatísticas para métodos bem-sucedidos
        lbfgs_successful = [r['lbfgs'] for r in self.comparison_results if r['lbfgs'] and r['lbfgs']['success']]
        mirror_successful = [r['mirror'] for r in self.comparison_results if r['mirror'] and r['mirror']['success']]
        coord_successful = [r['coordinate'] for r in self.comparison_results if r['coordinate'] and r['coordinate']['success']]
        
        if lbfgs_successful:
            lbfgs_avg_iter = np.mean([r['iterations'] for r in lbfgs_successful])
            lbfgs_avg_time = np.mean([r['execution_time'] for r in lbfgs_successful])
            lbfgs_avg_prec = np.mean([r['gradient_norm'] for r in lbfgs_successful])
            latex += f"L-BFGS-B & {lbfgs_avg_iter:.1f} & {lbfgs_avg_time:.3f} & {lbfgs_avg_prec:.2e} \\\\\n"
        else:
            latex += "L-BFGS-B & --- & --- & --- \\\\\n"
        
        if mirror_successful:
            mirror_avg_iter = np.mean([r['iterations'] for r in mirror_successful])
            mirror_avg_time = np.mean([r['execution_time'] for r in mirror_successful])
            mirror_avg_prec = np.mean([r['gradient_norm'] for r in mirror_successful])
            latex += f"Mirror Gradient & {mirror_avg_iter:.1f} & {mirror_avg_time:.3f} & {mirror_avg_prec:.2e} \\\\\n"
        else:
            latex += "Mirror Gradient & --- & --- & --- \\\\\n"
        
        if coord_successful:
            coord_avg_iter = np.mean([r['iterations'] for r in coord_successful])
            coord_avg_time = np.mean([r['execution_time'] for r in coord_successful])
            coord_avg_prec = np.mean([r['gradient_norm'] for r in coord_successful])
            latex += f"Coordinate Descent & {coord_avg_iter:.1f} & {coord_avg_time:.3f} & {coord_avg_prec:.2e} \\\\\n"
        else:
            latex += "Coordinate Descent & --- & --- & --- \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\end{table}

\section{Observações}

\begin{itemize}
\item \textbf{L-BFGS-B:} Método quasi-Newton robusto, geralmente converge rapidamente para problemas bem condicionados.
\item \textbf{Mirror Gradient:} Usa divergência de Bregman, pode ser mais eficaz para problemas com estrutura específica.
\item \textbf{Coordinate Descent:} Minimiza uma coordenada por vez, pode ser mais lento mas mais estável para alguns problemas.
\item A precisão é medida pela norma do gradiente ($||\nabla f(x^*)||$) calculada numericamente.
\item Tempos de execução incluem o cálculo do gradiente para verificação de precisão.
\item Problemas que falharam são marcados com "---" nas colunas de resultados.
\item A tolerância de convergência foi configurada em $10^{-6}$ para todos os métodos.
\end{itemize}

\section{Conclusões}

Com base nos resultados apresentados, podemos observar:

\begin{enumerate}
\item \textbf{Eficiência:} L-BFGS-B geralmente apresenta menor número de iterações.
\item \textbf{Robustez:} A taxa de sucesso varia entre os métodos dependendo do tipo de problema.
\item \textbf{Tempo:} O tempo de execução depende da complexidade do problema e do método.
\item \textbf{Precisão:} Todos os métodos convergem para soluções com alta precisão quando bem-sucedidos.
\end{enumerate}

\end{document}
"""
        
        return latex
    
    def print_comparison_summary(self):
        """
        Imprime um resumo da comparação.
        """
        if not self.comparison_results:
            print("Nenhum resultado disponível.")
            return
        
        print("\n" + "=" * 100)
        print("RESUMO COMPARATIVO DOS MÉTODOS")
        print("=" * 100)
        
        total_problems = len(self.comparison_results)
        
        # Calcular estatísticas
        lbfgs_success = sum(1 for r in self.comparison_results if r['lbfgs'] and r['lbfgs']['success'])
        mirror_success = sum(1 for r in self.comparison_results if r['mirror'] and r['mirror']['success'])
        coord_success = sum(1 for r in self.comparison_results if r['coordinate'] and r['coordinate']['success'])
        
        print(f"Total de problemas: {total_problems}")
        print(f"L-BFGS-B: {lbfgs_success}/{total_problems} ({lbfgs_success/total_problems*100:.1f}%)")
        print(f"Mirror Gradient: {mirror_success}/{total_problems} ({mirror_success/total_problems*100:.1f}%)")
        print(f"Coordinate Descent: {coord_success}/{total_problems} ({coord_success/total_problems*100:.1f}%)")
        
        # Estatísticas de performance
        lbfgs_successful = [r['lbfgs'] for r in self.comparison_results if r['lbfgs'] and r['lbfgs']['success']]
        mirror_successful = [r['mirror'] for r in self.comparison_results if r['mirror'] and r['mirror']['success']]
        coord_successful = [r['coordinate'] for r in self.comparison_results if r['coordinate'] and r['coordinate']['success']]
        
        print("\nEstatísticas de Performance:")
        print("-" * 50)
        
        if lbfgs_successful:
            lbfgs_avg_iter = np.mean([r['iterations'] for r in lbfgs_successful])
            lbfgs_avg_time = np.mean([r['execution_time'] for r in lbfgs_successful])
            print(f"L-BFGS-B: {lbfgs_avg_iter:.1f} iter médias, {lbfgs_avg_time:.3f}s médio")
        
        if mirror_successful:
            mirror_avg_iter = np.mean([r['iterations'] for r in mirror_successful])
            mirror_avg_time = np.mean([r['execution_time'] for r in mirror_successful])
            print(f"Mirror Gradient: {mirror_avg_iter:.1f} iter médias, {mirror_avg_time:.3f}s médio")
        
        if coord_successful:
            coord_avg_iter = np.mean([r['iterations'] for r in coord_successful])
            coord_avg_time = np.mean([r['execution_time'] for r in coord_successful])
            print(f"Coordinate Descent: {coord_avg_iter:.1f} iter médias, {coord_avg_time:.3f}s médio")


def main():
    """
    Função principal para executar a comparação.
    """
    # Criar comparador
    comparator = ComparisonSolver()
    
    # Executar comparação
    comparator.run_all_methods()
    
    # Imprimir resumo
    comparator.print_comparison_summary()
    
    # Gerar tabela LaTeX
    comparator.generate_comparison_latex()
    
    print("\nComparação concluída!")


if __name__ == "__main__":
    main()
