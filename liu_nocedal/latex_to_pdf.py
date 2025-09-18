"""
Script para converter salvar arquivos LaTeX e salvar em PDF
"""
from pdflatex import PDFLaTeX
import os
import subprocess
import tempfile
import shutil


# Adicionar MiKTeX ao PATH temporariamente
miktex_path = r"C:\Program Files\MiKTeX\miktex\bin\x64"
os.environ["PATH"] = miktex_path + os.pathsep + os.environ["PATH"]


def generate_latex_table(results, filename='liu_nocedal/latex_solution/results.txt', method_name='L-BFGS-B'):
    """
    Gera uma tabela LaTeX com os resultados.
    
    Args:
        results: Lista de resultados dos problemas
        filename (str): Nome do arquivo de saída
        method_name (str): Nome do método de otimização
    """
    if not results:
        print("Nenhum resultado disponível. Execute solve_all_problems() primeiro.")
        return
    
    # Criar diretório se não existir
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Diretório criado: {directory}")
    
    latex_content = create_latex_document(results, method_name)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"Tabela LaTeX salva em: {filename}")

def create_latex_document(results, method_name='L-BFGS-B'):
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

\title{Resultados dos Problemas de Otimização - Método """ + method_name + r"""}
\author{Maria Marcolina Lima Cardoso}
\date{\today}

\begin{document}

\maketitle

\section{Problemas de Otimização}

A tabela 1 apresenta os problemas de otimização não-linear resolvidos usando o método """ + method_name + r""" e o número de variáveis de cada problema.

\begin{table}[h!]
\centering
\caption{Problemas de otimização e número de variáveis}
\label{tab:problemas_variáveis}
\begin{tabular}{@{}|lc|@{}}
\toprule
\textbf{Problema} & \textbf{Número de Variáveis} \\
\midrule
"""
    
    # Adicionar linhas da primeira tabela
    for result in results:
        problem_name = result['problem'].replace('_', ' ')
        n_variables = str(result['n_variables'])
        latex += f"{problem_name} & {n_variables} \\\\\n"
    
    # Rodapé da tabela
    latex += r"""\bottomrule
\end{tabular}
\end{table}

\section{Resultados de Convergência}

A tabela 2 apresenta os resultados de convergência para cada problema, incluindo o número de iterações necessárias, o valor mínimo da função objetivo encontrado e a precisão da solução (norma do gradiente).


\begin{table}[h!]
\small
\centering
\caption{Resultados de convergência dos problemas de otimização}
\label{tab:resultados_convergencia}
\small
\begin{tabular}{|l|cccc|}
\hline
\textbf{Problema} & \textbf{Iterações} & \textbf{Valor Mínimo} & \textbf{Precisão ($||\nabla f(x^*)||$)} & \textbf{Tempo (s)}\\
\hline
"""
    
    # Adicionar linhas da segunda tabela
    for result in results:
        problem_name = result['problem'].replace('_', ' ')
        
        if result['success']:
            iterations = str(result['iterations'])
            function_value = f"{result['function_value']:.3e}"
            precision = f"{result['gradient_norm']:.3e}"
        else:
            iterations = "---"
            function_value = "Falhou"
            precision = "---"
        
        latex += f"{problem_name} & {iterations} & {function_value} & {precision} & {result['execution_time']:.3f}s \\\\\n"
    
    # Rodapé da segunda tabela
    latex += r"""\hline
\end{tabular}
\end{table}


\section{Soluções Encontradas (Primeiras 5 Variáveis)}

A tabela 3 apresenta as primeiras 5 variáveis da solução encontrada para cada problema. Para problemas com menos de 5 variáveis, apenas as variáveis disponíveis são mostradas.

\begin{landscape}
\begin{table}[h!]
\centering
\caption{Primeiras 5 variáveis das soluções encontradas}
\label{tab:solucoes_variáveis}
\begin{tabular}{|l|ccccc|}
\hline
\textbf{Problema} & \textbf{x1} & \textbf{x2} & \textbf{x3} & \textbf{x4} & \textbf{x5} \\
\hline
"""
    
    # Adicionar linhas da terceira tabela (variáveis)
    for result in results:
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
            row += " \\\\\n"
            latex += row
        else:
            # Para problemas que falharam
            row = f"{problem_name} & --- & --- & --- & --- & --- \\\\\n"
            latex += row
    
    # Rodapé da terceira tabela
    latex += r"""\hline
\hline
\end{tabular}
\end{table}
\end{landscape}

\section{Observações}

\begin{itemize}
\item O método L-BFGS-B foi configurado com tolerância de convergência de $10^{-6}$.
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

def generate_detailed_latex_table(results, filename='liu_nocedal/latex_solution/detailed_results.tex', method_name='Gradiente Espelhado'):
    """
    Gera uma tabela LaTeX detalhada com informações do problema e valores das variáveis em múltiplas colunas.
    
    Args:
        results: Lista de resultados dos problemas
        filename (str): Nome do arquivo de saída
        method_name (str): Nome do método de otimização
    """
    if not results:
        print("Nenhum resultado disponível. Execute solve_all_problems() primeiro.")
        return
    
    # Criar diretório se não existir
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Diretório criado: {directory}")
    
    latex_content = create_detailed_latex_document(results, method_name)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"Tabela LaTeX detalhada salva em: {filename}")

def create_detailed_latex_document(results, method_name='Gradiente Espelhado'):
    """
    Cria o documento LaTeX completo com informações detalhadas e valores das variáveis.
    """
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
\\usepackage{{pdflscape}}

\\geometry{{a4paper, margin=1.5cm}}

\\title{{Resultados Detalhados - Método {method_name} - Otimização de Grande Porte}}
\\author{{Maria Marcolina Lima Cardoso}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Resumo dos Resultados}}

A tabela abaixo apresenta um resumo dos resultados obtidos com o método {method_name}.

\\begin{{table}}[h!]
\\centering
\\caption{{Resumo dos resultados}}
\\label{{tab:resumo}}
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
\\textbf{{Problema}} & \\textbf{{Nº Variáveis}} & \\textbf{{Iterações}} & \\textbf{{Valor Mínimo}} & \\textbf{{Tempo (s)}} \\\\
\\midrule
"""
    
    # Adicionar resumo dos resultados
    for result in results:
        problem_name = result['problem'].replace('_', ' ')
        n_variables = str(result['n_variables'])
        
        if result['success']:
            iterations = str(result['iterations'])
            function_value = f"{result['function_value']:.6e}"
            time_str = f"{result['execution_time']:.3f}"
        else:
            iterations = "---"
            function_value = "Falhou"
            time_str = "---"
        
        latex += f"{problem_name} & {n_variables} & {iterations} & {function_value} & {time_str} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}

\section{Valores das Variáveis}

As tabelas abaixo apresentam os valores de cada variável para cada problema resolvido com sucesso.

"""
    
    # Criar tabela para cada problema
    for result in results:
        if result['success'] and 'x_value' in result and result['x_value'] is not None:
            problem_name = result['problem'].replace('_', ' ')
            x_values = result['x_value']
            n_vars = len(x_values)
            
            latex += f"""
\\newpage            
\\subsection{{Problema: {problem_name}}}

\\textbf{{Informações:}}
\\begin{{itemize}}
\\item Número de variáveis: {n_vars}
\\item Iterações: {result['iterations']}
\\item Valor mínimo: {result['function_value']:.6e}
\\item Tempo de execução: {result['execution_time']:.3f}s
\\end{{itemize}}

\\small
\\begin{{longtable}}{{@{{}}cc|cc|cc@{{}}}}
\\caption{{Valores das variáveis do problema {problem_name}}} \\\\
\\toprule
\\textbf{{Coordenada}} & \\textbf{{Valor}} & \\textbf{{Coordenada}} & \\textbf{{Valor}} & \\textbf{{Coordenada}} & \\textbf{{Valor}} \\\\
\\midrule
\\endfirsthead

\\toprule
\\textbf{{Coordenada}} & \\textbf{{Valor}} & \\textbf{{Coordenada}} & \\textbf{{Valor}} & \\textbf{{Coordenada}} & \\textbf{{Valor}} \\\\
\\midrule
\\endhead

\\midrule \\multicolumn{{6}}{{r}}{{{{Continua na próxima página}}}} \\\\ \\midrule
\\endfoot

\\bottomrule
\\endlastfoot
"""
            
            # Organizar variáveis em grupos de 3 colunas
            for i in range(0, n_vars, 3):
                row = ""
                for j in range(3):
                    coord_idx = i + j + 1
                    if coord_idx <= n_vars:
                        value = x_values[coord_idx - 1]
                        row += f"{coord_idx} & {value:.6f}"
                        if j < 2:
                            row += " & "
                    else:
                        row += " & "
                        if j < 2:
                            row += " & "
                
                # Completar a linha se necessário
                while row.count('&') < 5:
                    row += " & "
                
                row += " \\\\\n"
                latex += row
            
            latex += r"""
\end{longtable}

"""
    
    latex += r"""



\end{document}
"""
    
    return latex

def salvar_pdf(filename, output_dir):
    """
    Função principal
    """
    print("=" * 60)
    print("CONVERSOR LaTeX PARA PDF")
    print("=" * 60)
    
    convert_with_subprocess(filename, output_dir)

def convert_with_subprocess(tex_file, output_dir):
    try:
        # Criar diretório temporário para arquivos auxiliares
        temp_dir = tempfile.mkdtemp()
        
        # Executar pdflatex com arquivos auxiliares em diretório temporário
        result = subprocess.run([
            'pdflatex',
            '-interaction=nonstopmode',
            '-output-directory', output_dir,
            '-aux-directory', temp_dir,
            tex_file
        ], capture_output=True, text=True)
        
        # Limpar diretório temporário
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if result.returncode == 0:
            print("✓ PDF gerado com sucesso!")
        else:
            print(f"✗ Erro: {result.stderr}")
            
    except FileNotFoundError:
        print("✗ pdflatex não encontrado no PATH")        
