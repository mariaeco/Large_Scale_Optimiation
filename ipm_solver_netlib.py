# Solver em lote para problemas NETLIB usando HiGHS
import os
import highspy as hs
import pandas as pd
from time import perf_counter as pc
from highspy import Highs
import glob

class NetlibSolver:
    """
    Classe para resolver todos os problemas NETLIB em lote e gerar tabela de resultados.
    """
    
    def __init__(self, problems_dir='problems_mps', solver='ipm'):
        """
        Inicializa o solver em lote.
        
        Args:
            problems_dir (str): Diretório contendo os arquivos .mps
            solver (str): Solver a ser usado ('ipm', 'simplex', 'pdlp')
        """
        self.problems_dir = problems_dir
        self.solver = solver
        self.results = []
        

    
    def get_problem_files(self):
        """
        Obtém lista de arquivos .mps no diretório de problemas.
        
        Returns:
            list: Lista de caminhos para arquivos .mps
        """
        pattern = os.path.join(self.problems_dir, '*.mps')
        mps_files = glob.glob(pattern)
        
        # Também incluir arquivos .MPS (maiúsculo)
        pattern_upper = os.path.join(self.problems_dir, '*.MPS')
        mps_files.extend(glob.glob(pattern_upper))
        
        # Remover duplicatas baseado no nome do problema (sem extensão)
        seen_problems = set()
        unique_files = []
        
        for file_path in mps_files:
            problem_name = os.path.basename(file_path).split('.')[0].lower()
            if problem_name not in seen_problems:
                seen_problems.add(problem_name)
                unique_files.append(file_path)
        
        return sorted(unique_files)
    
    def solve_single_problem(self, mps_file):
        """
        Resolve um único problema NETLIB.
        
        Args:
            mps_file (str): Caminho para o arquivo .mps
            
        Returns:
            dict: Resultados da otimização ou None se falhou
        """
    
        # Criar instância do solver
        model = Highs()
        
        # Configurar solver
        model.setOptionValue("solver", self.solver)
        
        # Carregar modelo
        status = model.readModel(mps_file)
        if status != hs.HighsStatus.kOk:
            print(f"Erro ao carregar {mps_file}")
            return None
        
        # Resolver problema
        start_time = pc()
        model.run()
        end_time = pc()
        
        # Obter resultados
        model_status = model.getModelStatus()
        status_str = model.modelStatusToString(model_status)
        
        # Obter informações básicas
        info = model.getInfo()
        execution_time = end_time - start_time
        
        problem_name = os.path.basename(mps_file).split('.')[0]
        
        # Extrair número de variáveis e restrições da solução
        n_vars = 0
        n_constraints = 0
        
        solution = model.getSolution()
        if solution:
            if hasattr(solution, 'col_value') and solution.col_value is not None:
                n_vars = len(solution.col_value)
            if hasattr(solution, 'row_value') and solution.row_value is not None:
                n_constraints = len(solution.row_value)
        
        # Inicializar resultado
        result = {
            'PROBLEMA': problem_name,
            'N_VAR': n_vars,
            'N_RESTRICOES': n_constraints,
            'STATUS': status_str,
            'ITERAÇÕES': info.ipm_iteration_count if hasattr(info, 'ipm_iteration_count') else 'N/A',
            'TEMPO(SEG.)': f"{execution_time:.3f}",
            'VALOR ÓTIMO': 'N/A',
            'VALOR ÓTIMO PRIMAL': 'N/A',
            'VALOR ÓTIMO DUAL': 'N/A',
            'INVIABILIDADE PRIMAL': 'N/A',
            'INVIABILIDADE DUAL': 'N/A',
            'GAP ABSOLUTO': 'N/A',
            'GAP RELATIVO': 'N/A',
            'x_value': 'N/A'
        }
        
        # Se convergiu, obter valores adicionais
        if model_status == hs.HighsModelStatus.kOptimal:
            try:
                # Valor ótimo
                primal_value = model.getObjectiveValue()
                result['VALOR ÓTIMO'] = f"{primal_value:.3e}"
                result['VALOR ÓTIMO PRIMAL'] = f"{primal_value:.3e}"
                

                # Obter variáveis duais das restrições
          
                # Calcular valor dual usando: b^T * y (onde b é o lado direito e y são as variáveis duais)
                lp_model = model.getLp()
                row_lower = lp_model.row_lower_
                row_upper = lp_model.row_upper_
                row_dual = solution.row_dual
                
                # Construir vetor b (lado direito das restrições)
                b = []
                for i in range(len(row_lower)):
                    if row_lower[i] == row_upper[i]:
                        b.append(row_lower[i])
                    elif row_upper[i] < 1e20:  # Se não é infinito
                        b.append(row_upper[i])
                    else:
                        b.append(row_lower[i])
                
                # Calcular valor dual: b^T * y
                dual_value = sum(b[i] * row_dual[i] for i in range(len(b)))
                result['VALOR ÓTIMO DUAL'] = f"{dual_value:.3e}"


                result['x_value'] = solution.col_value

                # Obter informações básicas do modelo
                solution = model.getSolution()


                if solution:
                    # Viabilidade primal (norma dos resíduos)
                    if hasattr(solution, 'row_value'):
                        primal_inf = model.getInfo().max_primal_infeasibility
                        result['INVIABILIDADE PRIMAL'] = f"{primal_inf:.3e}"
                    
                    # Viabilidade dual (norma dos custos reduzidos)
                    if hasattr(solution, 'col_dual'):
                        dual_inf = model.getInfo().max_dual_infeasibility
                        result['INVIABILIDADE DUAL'] = f"{dual_inf:.3e}"
                        


                info = model.getInfo()
        
                # Verificar se temos informações sobre gaps
                print("Atributos disponíveis em info:")
                for attr in dir(info):
                        value = getattr(info, attr)
                        print(f"----- {attr}: {value}")
                        if 'primal_dual_objective_error' in attr.lower():
                            gap_abs = value
                            # Gap absoluto
                    
                # result['GAP ABSOLUTO'] = f"{gap_abs:.3e}"


                # Calcular gap absoluto usando primal e dual corretos
                # gap_abs = abs(primal_value - dual_value)
                result['GAP ABSOLUTO'] = f"{gap_abs:.3e}"

                info = model.getInfo()
                #wuero os atributos do info para pegar o Relative P-D gap
                rel_gap = gap_abs/primal_value

                result['GAP RELATIVO'] = f"{rel_gap:.3e}"
                    



            except Exception as e:
                print(f"Erro ao extrair informações detalhadas para {problem_name}: {e}")
        
        return result
        

    
    
    def solve_all_problems(self):
        """
        Resolve todos os problemas NETLIB no diretório especificado.
        """
        mps_files = self.get_problem_files()
        
        if not mps_files:
            print(f"Nenhum arquivo .mps encontrado em {self.problems_dir}")
            return
        
        print(f"Encontrados {len(mps_files)} problemas para resolver")
        print(f"Usando solver: {self.solver}")
        
        self.results = []
        
        for i, mps_file in enumerate(mps_files, 1):
            problem_name = os.path.basename(mps_file).split('.')[0]
            print("--------------------------------------------------------------------")
            print(f"[{i}/{len(mps_files)}] Resolvendo: {problem_name}")
            
            result = self.solve_single_problem(mps_file)
            if result:
                self.results.append(result)
                
                # Imprimir resultado resumido
                status = result['STATUS']
                iterations = result['ITERAÇÕES']
                time = result['TEMPO(SEG.)']
                value = result['VALOR ÓTIMO']
                
                if status == 'Optimal':
                    print(f"  ✓ {status}: {iterations} iterações, {time}s, valor = {value}")
                else:
                    print(f"  ✗ {status}: {iterations} iterações, {time}s")
        
        print(f"Resolução concluída! {len(self.results)} problemas processados.")
    
    def generate_results_table(self):
        """
        Gera tabela de resultados em formato pandas DataFrame.
        
        Returns:
            pd.DataFrame: Tabela com todos os resultados
        """
        if not self.results:
            print("Nenhum resultado disponível. Execute solve_all_problems() primeiro.")
            return None
        
        df = pd.DataFrame(self.results)
        return df
    
    def print_summary(self):
        """
        Imprime resumo dos resultados.
        """
        if not self.results:
            print("Nenhum resultado disponível.")
            return
        
        df = self.generate_results_table()
        
        # Estatísticas gerais
        total_problems = len(df)
        optimal_count = len(df[df['STATUS'] == 'Optimal'])
        success_rate = (optimal_count / total_problems) * 100
        
        print("\n" + "="*80)
        print("RESUMO DOS RESULTADOS NETLIB")
        print("="*80)
        print(f"Total de problemas: {total_problems}")
        print(f"Problemas resolvidos com sucesso: {optimal_count}")
        print(f"Taxa de sucesso: {success_rate:.1f}%")
        
        if optimal_count > 0:
            optimal_df = df[df['STATUS'] == 'Optimal']
            avg_iterations = optimal_df['ITERAÇÕES'].astype(float).mean()
            avg_time = optimal_df['TEMPO(SEG.)'].astype(float).mean()
            print(f"Número médio de iterações: {avg_iterations:.1f}")
            print(f"Tempo médio por problema: {avg_time:.3f}s")
        
        # Problemas que falharam
        failed_df = df[df['STATUS'] != 'Optimal']
        if len(failed_df) > 0:
            print(f"\nProblemas que falharam ({len(failed_df)}):")
            for _, row in failed_df.iterrows():
                print(f"  {row['PROBLEMA']}: {row['STATUS']}")
    
    def save_results_to_csv(self, filename='resultados_netlib.csv'):
        """
        Salva resultados em arquivo CSV.
        
        Args:
            filename (str): Nome do arquivo CSV
        """
        if not self.results:
            print("Nenhum resultado disponível.")
            return
        
        df = self.generate_results_table()
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Resultados salvos em: {filename}")
    
    def generate_latex_table(self, filename='resultados_netlib.tex'):
        """
        Gera tabela LaTeX com os resultados.
        
        Args:
            filename (str): Nome do arquivo LaTeX
        """
        if not self.results:
            print("Nenhum resultado disponível.")
            return
        
        df = self.generate_results_table()
        print(df)
        
        # Criar documento LaTeX
        latex_content = self._create_latex_document(df)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"Tabela LaTeX salva em: {filename}")
    
    def _create_latex_document(self, df):
        """
        Cria documento LaTeX completo com as 3 tabelas de resultados.
        
        Args:
            df (pd.DataFrame): DataFrame com os resultados
            
        Returns:
            str: Conteúdo do documento LaTeX
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
\usepackage{longtable}

\geometry{a4paper, margin=1.5cm}

\title{Resultados dos Problemas NETLIB - Solver HiGHS}
\author{Análise Computacional}
\date{\today}

\begin{document}

\maketitle

\section{Informações dos Problemas}

Esta tabela apresenta informações básicas sobre cada problema da coleção NETLIB, incluindo métricas de viabilidade.

\scriptsize
\begin{longtable}{@{}l|cccc@{}}
\caption{Informações dos problemas NETLIB} \label{tab:info_problemas} \\
\toprule
\textbf{Problema} & \textbf{Nº de Variáveis} & \textbf{Nº de Restrições} & \textbf{Inviab. Primal} & \textbf{Inviab. Dual} \\
\midrule
\endfirsthead

\toprule
\textbf{Problema} & \textbf{Nº de Variáveis} & \textbf{Nº de Restrições} & \textbf{Inviab. Primal} & \textbf{Inviab. Dual} \\
\midrule
\endhead

\midrule \multicolumn{5}{r}{{Continua na próxima página}} \\ \midrule
\endfoot

\bottomrule
\endlastfoot
"""
        
        # Adicionar linhas da primeira tabela
        for _, row in df.iterrows():
            problem = row['PROBLEMA']
            N_VAR = row['N_VAR']    
            N_RESTRICOES = row['N_RESTRICOES']
            primal_inf = row['INVIABILIDADE PRIMAL']
            dual_inf = row['INVIABILIDADE DUAL']
            
            latex += f"{problem} & {N_VAR} & {N_RESTRICOES} & {primal_inf} & {dual_inf} \\\\\n"
        
        # Rodapé da primeira tabela
        latex += r"""
\bottomrule
\end{longtable}

\section{Resultados de Convergência}

Esta tabela apresenta os resultados de convergência para cada problema, incluindo o número de iterações, valor da função objetivo e o gap relativo.

\scriptsize
\begin{longtable}{@{}l|cccccc@{}}
\caption{Resultados de convergência dos problemas NETLIB} \label{tab:resultados_convergencia} \\
\toprule
\textbf{Problema} & \textbf{Iterações} & \textbf{Valor Ótimo} & \textbf{Valor Ótimo Primal} & \textbf{Valor Ótimo Dual} & \textbf{Gap Absoluto} & \textbf{Gap Relativo} \\
\midrule
\endfirsthead


\multicolumn{7}{c}%
{{\bfseries \tablename\ \thetable{} -- continuação da página anterior}} \\
\toprule
\textbf{Problema} & \textbf{Iterações} & \textbf{Valor Ótimo} & \textbf{Valor Ótimo Primal} & \textbf{Valor Ótimo Dual} & \textbf{Gap Absoluto} & \textbf{Gap Relativo} \\
\midrule
\endhead

\midrule \multicolumn{7}{r}{{Continua na próxima página}} \\ \midrule
\endfoot

\bottomrule
\endlastfoot
"""
        
        # Adicionar linhas da segunda tabela
        for _, row in df.iterrows():
            problem = row['PROBLEMA']
            iterations = str(row['ITERAÇÕES'])
            function_value = row['VALOR ÓTIMO']
            primal_value = row['VALOR ÓTIMO PRIMAL']
            dual_value = row['VALOR ÓTIMO DUAL']
            gap_abs = row['GAP ABSOLUTO']
            gap_rel = row['GAP RELATIVO']
            
            # # Formatar valores para LaTeX com 3 casas decimais
            # if function_value == 'N/A':
            #     function_formatted = 'N/A'
            # else:
            #     # Converter para float e formatar com 3 casas decimais
            #     try:
            #         val = float(function_value)
            #         function_formatted = f"${val:.3f}$"
            #     except:
            #         function_formatted = f"${function_value}$"
                
            # if primal_value == 'N/A':
            #     primal_formatted = 'N/A'
            # else:
            #     try:
            #         val = float(primal_value)
            #         primal_formatted = f"${val:.3f}$"
            #     except:
            #         primal_formatted = f"${primal_value}$"
                
            # if dual_value == 'N/A':
            #     dual_formatted = 'N/A'
            # else:
            #     try:
            #         val = float(dual_value)
            #         dual_formatted = f"${val:.3f}$"
            #     except:
            #         dual_formatted = f"${dual_value}$"
            
            latex += f"{problem} & {iterations} & {function_value} & {primal_value} & {dual_value} & {gap_abs} & {gap_rel} \\\\\n"
        
        # Rodapé da segunda tabela
        latex += r"""
\bottomrule
\end{longtable}

\section{Soluções das Variáveis (Primeiras 5)}

Esta tabela apresenta as primeiras 5 variáveis da solução encontrada para cada problema. Para problemas com menos de 5 variáveis, apenas as variáveis disponíveis são mostradas.

\scriptsize % ou \footnotesize, escolha conforme achar melhor
\begin{longtable}{|l|ccccc|}
\caption{Primeiras 5 variáveis das soluções encontradas\label{tab:solucoes_variaveis}} \\
\hline
\textbf{Problema} & \textbf{x1} & \textbf{x2} & \textbf{x3} & \textbf{x4} & \textbf{x5} \\
\hline
\endfirsthead

\hline
\textbf{Problema} & \textbf{x1} & \textbf{x2} & \textbf{x3} & \textbf{x4} & \textbf{x5} \\
\hline
\endhead

\hline
\multicolumn{6}{r}{{Continua na próxima página}} \\
\endfoot

\hline
\endlastfoot
"""
        
        # Adicionar linhas da terceira tabela (variáveis)
        for _, row in df.iterrows():
            problem = row['PROBLEMA']
            
            if row['STATUS'] == 'Optimal' and row['x_value'] != 'N/A':
                x_values = row['x_value']
                n_vars = min(5, len(x_values))  # Máximo 5 variáveis
                
                # Criar linha com as primeiras 5 variáveis
                row_latex = f"{problem}"
                for i in range(5):
                    if i < n_vars:
                        row_latex += f" & {x_values[i]:.3f}"
                    else:
                        row_latex += " & ---"
                row_latex += " \\\\\n"
                latex += row_latex
            else:
                # Para problemas que falharam
                row_latex = f"{problem} & --- & --- & --- & --- & --- \\\\\n"
                latex += row_latex
        
        # Rodapé da terceira tabela
        latex += r"""

\hline
\end{longtable}

\section{Observações}

\begin{itemize}
\item O solver HiGHS foi configurado com o método IPM (Interior Point Method).
\item Problemas com status "Optimal" convergiram com sucesso.
\item A primeira tabela mostra informações básicas dos problemas e métricas de viabilidade.
\item A segunda tabela apresenta métricas de convergência e qualidade da solução.
\item A terceira tabela mostra valores simulados das primeiras 5 variáveis (para demonstração).
\item A terceira tabela é apresentada em formato paisagem para melhor visualização.
\end{itemize}

\end{document}
"""
        
        return latex


def main():
    """
    Função principal para executar a análise em lote.
    """
    # Criar solver
    solver = NetlibSolver(problems_dir='PROJETO/netlib_problems', solver='ipm')
    
    # Resolver todos os problemas
    solver.solve_all_problems()
    
    # Imprimir resumo
    solver.print_summary()
    
    # Salvar resultados
    solver.save_results_to_csv('PROJETO/latex_solution/resultados_netlib.csv')
    solver.generate_latex_table('PROJETO/latex_solution/resultados_netlib.tex')
    
    print("\nAnálise NETLIB concluída!")


if __name__ == "__main__":
    main()
