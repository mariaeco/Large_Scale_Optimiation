"""
Funções para construir documentos LaTeX para resultados NETLIB
"""
import os
import ast
import numpy as np


def create_latex_document(df):
    """
    Cria o documento LaTeX completo para os resultados NETLIB.
    
    Args:
        df: DataFrame com os resultados dos problemas
        
    Returns:
        str: Conteúdo LaTeX completo
    """
    if df is None or df.empty:
        return ""
    
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
    
    # Adicionar dados dos problemas
    for _, row in df.iterrows():
        problem_name = row['PROBLEMA']
        n_vars = int(row['N_VAR'])
        n_constraints = int(row['N_RESTRICOES'])
        primal_infeas = format_float(row['INVIABILIDADE PRIMAL'])
        dual_infeas = format_float(row['INVIABILIDADE DUAL'])
        
        latex += f"{problem_name} & {n_vars} & {n_constraints} & {primal_infeas} & {dual_infeas} \\\\\n"
    
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
\textbf{Problema} & \textbf{Iterações} & \textbf{Primal} & \textbf{Dual} & \textbf{Gap Absoluto} & \textbf{Gap Relativo} \\
\midrule
\endhead

\midrule \multicolumn{6}{r}{{Continua na próxima página}} \\ \midrule
\endfoot

\bottomrule
\endlastfoot
"""
    
    # Adicionar dados de convergência
    for _, row in df.iterrows():
        problem_name = row['PROBLEMA']
        iterations = int(row['ITERAÇÕES']) if row['ITERAÇÕES'] is not None and row['ITERAÇÕES'] != '' else 0
        primal_value = format_float(row['VALOR ÓTIMO PRIMAL'])
        dual_value = format_float(row['VALOR ÓTIMO DUAL'])
        gap_abs = format_float(row['GAP ABSOLUTO'])
        gap_rel = format_float(row['GAP RELATIVO'])
        
        latex += f"{problem_name} & {iterations} & {primal_value} & {dual_value} & {gap_abs} & {gap_rel} \\\\\n"
    
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


def create_individual_problem_document(row):
    """
    Cria documento LaTeX individual para um problema específico.
    
    Args:
        row: Linha do DataFrame com dados do problema
        
    Returns:
        str: Conteúdo LaTeX do problema individual
    """
    problem_name = row['PROBLEMA']
    n_vars = int(row['N_VAR'])
    n_constraints = int(row['N_RESTRICOES'])
    primal_infeas = format_float(row['INVIABILIDADE PRIMAL'])
    dual_infeas = format_float(row['INVIABILIDADE DUAL'])
    primal_value = format_float(row['VALOR ÓTIMO PRIMAL'])
    dual_value = format_float(row['VALOR ÓTIMO DUAL'])
    gap = format_float(row['GAP RELATIVO'])
    iterations = int(row['ITERAÇÕES']) if row['ITERAÇÕES'] is not None and row['ITERAÇÕES'] != '' else 0
    
    # Cabeçalho do documento
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

\\title{{Solução do Problema {problem_name} - Solver HiGHS}}
\\author{{Análise Computacional}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Informações do Problema}}

\\textbf{{Informações do Problema:}}
\\begin{{itemize}}
\\item Nome: {problem_name}
\\item Número de Variáveis: {n_vars}
\\item Número de Restrições: {n_constraints}
\\item Inviabilidade Primal: {primal_infeas}
\\item Inviabilidade Dual: {dual_infeas}
\\item Valor Primal: {primal_value}
\\item Valor Dual: {dual_value}
\\item Gap: {gap}
\\item Número de Iterações: {iterations}
\\end{{itemize}}


\\section{{Variáveis Primais e Custos Reduzidos (x > 0 e z > 0)}}

\\begin{{longtable}}{{@{{}}cccc@{{}}}}
\\caption{{Variáveis primais (x) e custos reduzidos (z) com valores > 0 do problema {problem_name}}} \\\\
\\toprule
\\textbf{{Coordenada x}} & \\textbf{{Valor x (Primal)}} & \\textbf{{Coordenada z}} & \\textbf{{Valor z (Custo Reduzido)}} \\\\
\\midrule
\\endfirsthead

\\toprule
\\textbf{{Coordenada x}} & \\textbf{{Valor x (Primal)}} & \\textbf{{Coordenada z}} & \\textbf{{Valor z (Custo Reduzido)}} \\\\
\\midrule
\\endhead

\\midrule \\multicolumn{{4}}{{r}}{{{{Continua na próxima página}}}} \\\\ \\midrule
\\endfoot

\\bottomrule
\\endlastfoot
"""
    
    # Adicionar dados das variáveis primais (x) e custos reduzidos (z) com valores > 0
    try:
        # Verificar se as colunas existem
        if 'x_value' not in row or 'z_value' not in row:
            latex += f"Dados de variáveis não encontrados para {problem_name} \\\\\n"
        else:
            x_values = parse_list_from_string(row['x_value'])  # Variáveis primais
            z_values = parse_list_from_string(row['z_value'])   # Custos reduzidos
            
            if x_values is not None and z_values is not None:
                # Filtrar valores x > 0
                x_filtered_entries = []
                for i in range(len(x_values)):
                    x_val = x_values[i]
                    if x_val > 0:
                        x_filtered_entries.append((i+1, x_val))
                
                # Filtrar valores z > 0
                z_filtered_entries = []
                for i in range(len(z_values)):
                    z_val = z_values[i]
                    if z_val > 0:
                        z_filtered_entries.append((i+1, z_val))
                
                # Combinar as entradas x e z lado a lado
                max_combined = max(len(x_filtered_entries), len(z_filtered_entries))
                
                # Limitar a 500 linhas para não tornar o arquivo muito grande
                max_lines = min(500, max_combined)
                
                for i in range(max_lines):
                    # Obter valores x e z para a linha i
                    x_coord = ""
                    x_val = ""
                    if i < len(x_filtered_entries):
                        x_coord, x_val = x_filtered_entries[i]
                    
                    z_coord = ""
                    z_val = ""
                    if i < len(z_filtered_entries):
                        z_coord, z_val = z_filtered_entries[i]
                    
                    # Formatar valores
                    x_formatted = format_float(x_val) if x_val != "" else ""
                    z_formatted = format_float(z_val) if z_val != "" else ""
                    
                    latex += f"{x_coord} & {x_formatted} & {z_coord} & {z_formatted} \\\\\n"
            else:
                latex += f"Dados de variáveis não puderam ser processados para {problem_name} \\\\\n"
            
    except Exception as e:
        latex += f"Dados não disponíveis: {str(e)} \\\\\n"
    
    latex += r"""
\bottomrule
\end{longtable}

"""
    
    # Adicionar seção de variáveis duais (y) se disponível
    try:
        if 'y_value' in row:
            y_values = parse_list_from_string(row['y_value'])
            if y_values is not None and len(y_values) > 0:
                latex += f"""
\\section{{Variáveis Duais (Multiplicadores de Lagrange - Valores > 0)}}

\\begin{{longtable}}{{@{{}}cc@{{}}}}
\\caption{{Variáveis duais (y) com valores > 0 do problema {problem_name}}} \\\\
\\toprule
\\textbf{{Coordenada y}} & \\textbf{{Valor y (Dual)}} \\\\
\\midrule
\\endfirsthead

\\toprule
\\textbf{{Coordenada y}} & \\textbf{{Valor y (Dual)}} \\\\
\\midrule
\\endhead

\\midrule \\multicolumn{{2}}{{r}}{{{{Continua na próxima página}}}} \\\\ \\midrule
\\endfoot

\\bottomrule
\\endlastfoot
"""
                
                # Filtrar apenas valores y > 0
                filtered_dual_entries = []
                for i in range(len(y_values)):
                    y_val = y_values[i]
                    if y_val > 0:
                        filtered_dual_entries.append((i+1, y_val))
                
                # Limitar a 500 entradas para não tornar o arquivo muito grande
                max_dual_entries = min(500, len(filtered_dual_entries))
                
                for i in range(max_dual_entries):
                    coord, y_val = filtered_dual_entries[i]
                    y_formatted = format_float(y_val)
                    latex += f"{coord} & {y_formatted} \\\\\n"
                
                latex += r"""
\bottomrule
\end{longtable}

"""
    except Exception as e:
        print(f"Erro ao processar variáveis duais para {problem_name}: {e}")
    
    latex += r"""
\end{document}
"""
    
    return latex


def parse_list_from_string(value_str):
    """
    Converte string de lista para lista Python.
    
    Args:
        value_str: String contendo uma lista
        
    Returns:
        list: Lista Python ou None se erro
    """
    # Verificar se é None ou N/A
    if value_str is None or value_str == 'N/A':
        return None
    
    # Verificar se é pandas NaN
    try:
        import pandas as pd
        if pd.isna(value_str):
            return None
    except:
        pass
    
    try:
        # Se já é uma lista (não string), retornar diretamente
        if isinstance(value_str, list):
            return value_str
        
        # Se é numpy array, converter para lista
        if hasattr(value_str, 'tolist'):
            return value_str.tolist()
            
        # Se é string, tentar converter
        if isinstance(value_str, str):
            # Remover aspas externas se existirem
            if value_str.startswith('"') and value_str.endswith('"'):
                value_str = value_str[1:-1]
            
            # Verificar se é uma string que representa uma lista
            if value_str.startswith('[') and value_str.endswith(']'):
                # Usar ast.literal_eval para converter string para lista
                return ast.literal_eval(value_str)
            else:
                print(f"String não parece ser uma lista: {value_str[:50]}...")
                return None
        else:
            print(f"Tipo não reconhecido: {type(value_str)}")
            return None
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Erro ao converter lista: {e}")
        return None


def format_float(value):
    """
    Formata um valor float para exibição no LaTeX.
    
    Args:
        value: Valor a ser formatado
        
    Returns:
        str: Valor formatado
    """
    # Verificar se é None ou N/A
    if value is None or value == 'N/A':
        return 'N/A'
    
    # Verificar se é pandas NaN
    try:
        import pandas as pd
        if pd.isna(value):
            return 'N/A'
    except:
        pass
    
    try:
        float_val = float(value)
        if float_val == 0.0:
            return '0.000e+00'
        else:
            return f"{float_val:.3e}"
    except (ValueError, TypeError):
        return 'N/A'


def generate_individual_problem_files(df, output_dir='netlib_ipm_solver/latex_solution/relatorio_individual_problems_tex'):
    """
    Gera arquivos LaTeX individuais para cada problema.
    
    Args:
        df: DataFrame com os resultados
        output_dir: Diretório de saída
    """
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        problem_name = row['PROBLEMA']
        filename = os.path.join(output_dir, f"{problem_name}.tex")
        
        # Criar conteúdo LaTeX para o problema individual
        latex_content = create_individual_problem_document(row)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"Arquivo individual salvo: {filename}")


def generate_general_report(df, output_file='netlib_ipm_solver/latex_solution/relatorio_geral_netlib.tex'):
    """
    Gera relatório geral LaTeX.
    
    Args:
        df: DataFrame com os resultados
        output_file: Arquivo de saída
    """
    latex_content = create_latex_document(df)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"Relatório geral salvo: {output_file}")


# Adicionar import do pandas se não estiver disponível
try:
    import pandas as pd
except ImportError:
    print("Pandas não está disponível. Instale com: pip install pandas")
    pd = None
