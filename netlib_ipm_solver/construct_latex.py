


def create_individual_problem_document(row):
    """
    Cria documento LaTeX para um problema individual.
    
    Args:
        row: Linha do DataFrame com os dados do problema
        
    Returns:
        str: Conteúdo do documento LaTeX
    """
    problem = row['PROBLEMA']
    
    latex = f"""
\\documentclass[12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[portuguese]{{babel}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{geometry}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{longtable}}

\\geometry{{a4paper, margin=1.5cm}}

\\title{{Solução do Problema {problem} - Solver HiGHS}}
\\author{{Análise Computacional}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Informações do Problema}}

\\textbf{{Informações do Problema:}}
\\begin{{itemize}}
\\item Nome: {problem}
\\item Número de Variáveis: {row['N_VAR']}
\\item Número de Restrições: {row['N_RESTRICOES']}
\\item Inviabilidade Primal: {row['INVIABILIDADE PRIMAL']}
\\item Inviabilidade Dual: {row['INVIABILIDADE DUAL']}
\\item Valor Primal: {row['VALOR ÓTIMO PRIMAL']}
\\item Valor Dual: {row['VALOR ÓTIMO DUAL']}
\\item Gap: {row['GAP RELATIVO']}
\\item Número de Iterações: {row['ITERAÇÕES']}
\\end{{itemize}}

"""
    
    if row['STATUS'] == 'Optimal':
        # Adicionar tabela de variáveis primais e custos reduzidos
        latex += f"""
\\section{{Variáveis Primais e Custos Reduzidos}}

\\begin{{longtable}}{{@{{}}cccc@{{}}}}
\\caption{{Variáveis primais e custos reduzidos do problema {problem}}} \\\\
\\toprule
\\textbf{{Coordenada x}} & \\textbf{{Valor x}} & \\textbf{{Coordenada z}} & \\textbf{{Valor z}} \\\\
\\midrule
\\endfirsthead

\\toprule
\\textbf{{Coordenada x}} & \\textbf{{Valor x}} & \\textbf{{Coordenada z}} & \\textbf{{Valor z}} \\\\
\\midrule
\\endhead

\\midrule \\multicolumn{{4}}{{r}}{{{{Continua na próxima página}}}} \\\\ \\midrule
\\endfoot

\\bottomrule
\\endlastfoot
"""
        
        # Adicionar variáveis primais e custos reduzidos
        if row['x_value'] != 'N/A':
            x_values = row['x_value']
            z_values = row['z_value']
            
            # Determinar o número máximo de variáveis
            max_vars = max(len(x_values), len(z_values))
            
            for i in range(max_vars):
                # Coordenada x e valor x
                val_x = x_values[i] if i < len(x_values) else 0.0
                
                # # Só imprimir se o valor x for maior que 10e-12
                # if abs(val_x) <= 10e-12:
                coord_x = i + 1
                
                # Coordenada z e valor z
                coord_z = i + 1
                val_z = z_values[i] if i < len(z_values) else 0.0
                
                latex += f"{coord_x} & {val_x:.6f} & {coord_z} & {val_z:.6f} \\\\\n"
            
        latex += r"""
\end{longtable}

\section{Variáveis Duais (Multiplicadores de Lagrange)}

\begin{longtable}{@{}cc@{}}
\caption{Variáveis duais do problema """ + problem + r"""} \\
\toprule
\textbf{Coordenada y} & \textbf{Valor y} \\
\midrule
\endfirsthead

\toprule
\textbf{Coordenada y} & \textbf{Valor y} \\
\midrule
\endhead

\midrule \multicolumn{2}{r}{{Continua na próxima página}} \\ \midrule
\endfoot

\bottomrule
\endlastfoot
"""
        
        # Adicionar variáveis duais
        if row['y_value'] != 'N/A':
            y_values = row['y_value']
            for i, value in enumerate(y_values):
                latex += f"{i+1} & {value:.6f} \\\\\n"
        
        latex += r"""
\end{longtable}
"""
    else:
        # Para problemas que falharam
        latex += f"""
\\section{{Status}}

\\textbf{{Status:}} {row['STATUS']} - Problema não convergiu.
"""
    
    # Adicionar observações
    latex += r"""

\section{Observações}

\begin{itemize}
\item O solver HiGHS foi configurado com o método IPM (Interior Point Method).
\item Este arquivo contém a solução detalhada para o problema """ + problem + r""".
\item As variáveis duais representam os multiplicadores de Lagrange das restrições.
\item Os custos reduzidos (z) indicam o impacto de forçar variáveis não-básicas na base.
\end{itemize}

\end{document}
"""
    
    return latex

def create_latex_document(df):
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
\begin{longtable}{@{}l|ccccc@{}}
\caption{Resultados de convergência dos problemas NETLIB} \label{tab:resultados_convergencia} \\
\toprule
\textbf{Problema} & \textbf{Iterações} & \textbf{Primal} & \textbf{Dual} & \textbf{Gap Absoluto} & \textbf{Gap Relativo} \\
\midrule
\endfirsthead


\multicolumn{6}{c}%
{{\bfseries \tablename\ \thetable{} -- continuação da página anterior}} \\
\toprule
\textbf{Problema} & \textbf{Iterações}  \textbf{Primal} & \textbf{Dual} & \textbf{Gap Absoluto} & \textbf{Gap Relativo} \\
\midrule
\endhead

\midrule \multicolumn{6}{r}{{Continua na próxima página}} \\ \midrule
\endfoot

\bottomrule
\endlastfoot
"""
    
    # Adicionar linhas da segunda tabela
    for _, row in df.iterrows():
        problem = row['PROBLEMA']
        iterations = str(row['ITERAÇÕES'])
        primal_value = row['VALOR ÓTIMO PRIMAL']
        dual_value = row['VALOR ÓTIMO DUAL']
        gap_abs = row['GAP ABSOLUTO']
        gap_rel = row['GAP RELATIVO']
        
        
        latex += f"{problem} & {iterations} & {primal_value} & {dual_value} & {gap_abs} & {gap_rel} \\\\\n"
    
    # Rodapé da segunda tabela
    latex += r"""
\bottomrule
\end{longtable}

\section{Observações}

\begin{itemize}
\item O solver HiGHS foi configurado com o método IPM (Interior Point Method).
\item Problemas com status "Optimal" convergiram com sucesso.
\item A primeira tabela mostra informações básicas dos problemas e métricas de viabilidade.
\item A segunda tabela apresenta métricas de convergência e qualidade da solução.
\item As soluções detalhadas de cada problema (variáveis primais e duais) são salvas em arquivos individuais.
\item As variáveis duais representam os multiplicadores de Lagrange das restrições.
\end{itemize}



\end{document}
"""
    
    return latex