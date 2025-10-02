"""
Script para gerar tabela de comparação dos três métodos de otimização
L-BFGS-B, Descent Coordinate e Mirror Gradient
"""

import os
import subprocess
import tempfile
import shutil
import re
from typing import List, Dict, Any

def format_scientific_number(value_str: str) -> str:
    """
    Converte número em formato científico para formato normal com 4 casas decimais.
    
    Args:
        value_str: String com número em formato científico (ex: "1.090881e+02")
        
    Returns:
        String com número formatado (ex: "109.0881")
    """
    if value_str == "---" or value_str == "Falhou":
        return value_str
    
    try:
        # Converter string para float
        value = float(value_str)
        # Formatar com 4 casas decimais
        return f"{value:.4f}"
    except (ValueError, TypeError):
        return value_str

def extract_data_from_latex(latex_file: str) -> List[Dict[str, Any]]:
    """
    Extrai dados das tabelas de resumo dos arquivos LaTeX.
    
    Args:
        latex_file: Caminho para o arquivo LaTeX
        
    Returns:
        Lista de dicionários com os dados extraídos
    """
    if not os.path.exists(latex_file):
        print(f"Arquivo não encontrado: {latex_file}")
        return []
    
    with open(latex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Encontrar a tabela de resumo
    table_pattern = r'\\begin\{tabular\}.*?\\end\{tabular\}'
    table_match = re.search(table_pattern, content, re.DOTALL)
    
    if not table_match:
        print(f"Não foi possível encontrar tabela em {latex_file}")
        return []
    
    table_content = table_match.group(0)
    # Extrair linhas da tabela (ignorando cabeçalho)
    lines = table_content.split('\\')
    data_lines = []
    
    for line in lines:
        if '&' in line and not 'textbf' in line and not 'toprule' in line and not 'bottomrule' in line:
            # Limpar a linha
            line = line.strip()
            if line and not line.startswith('\\'):
                data_lines.append(line)
    
    results = []
    for line in data_lines:
        if '&' in line:
            parts = line.split('&')
            if len(parts) >= 5:
                problem = parts[0].strip()
                # Limpar o nome do problema (remover midrule, etc.)
                problem = problem.replace('midrule', '').replace('\n', '').strip()
                n_vars = parts[1].strip()
                iterations = parts[2].strip()
                min_value = parts[3].strip()
                time = parts[4].strip().replace('\\\\', '').strip()
                
                # Formatar valor mínimo
                min_value_formatted = format_scientific_number(min_value)
                
                results.append({
                    'problem': problem,
                    'n_variables': n_vars,
                    'iterations': iterations,
                    'min_value': min_value_formatted,
                    'time': time
                })
    
    return results

def create_comparison_latex(lbfgs_data: List[Dict], descent_data: List[Dict], mirror_data: List[Dict]) -> str:
    """
    Cria o documento LaTeX com a tabela de comparação.
    
    Args:
        lbfgs_data: Dados do método L-BFGS-B
        descent_data: Dados do método Descent Coordinate
        mirror_data: Dados do método Mirror Gradient
        
    Returns:
        String com o conteúdo LaTeX
    """
    
    # Obter lista única de problemas
    all_problems = set()
    for data in [lbfgs_data, descent_data, mirror_data]:
        for item in data:
            all_problems.add(item['problem'])
    
    all_problems = sorted(list(all_problems))
    
    latex_content = r"""
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[portuguese]{babel}
\usepackage{booktabs}
\usepackage{array}
\usepackage{geometry}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{longtable}
\usepackage{pdflscape}
\usepackage{multirow}

\geometry{a4paper, margin=1.5cm}

\title{Comparação dos Métodos de Otimização - L-BFGS-B, Descent Coordinate e Mirror Gradient}
\author{Maria Marcolina Lima Cardoso}
\date{\today}

\begin{document}

\maketitle

\section{Comparação dos Resultados}

A tabela abaixo apresenta uma comparação dos resultados obtidos com os três métodos de otimização: L-BFGS-B, Descent Coordinate e Mirror Gradient.

\begin{landscape}
\begin{table}[h!]
\centering
\caption{Comparação dos métodos de otimização}
\label{tab:comparacao}
\footnotesize
\begin{tabular}{@{}ll|ccc|ccc|ccc@{}}
\toprule
\multirow{2}{*}{\textbf{Problema}} & \multirow{2}{*}{\textbf{Vars}} & \multicolumn{3}{c|}{\textbf{L-BFGS-B}} & \multicolumn{3}{c|}{\textbf{Descent Coordinate}} & \multicolumn{3}{c}{\textbf{Mirror Gradient}} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}
& & \textbf{Iter.} & \textbf{Valor Mín.} & \textbf{Tempo (s)} & \textbf{Iter.} & \textbf{Valor Mín.} & \textbf{Tempo (s)} & \textbf{Iter.} & \textbf{Valor Mín.} & \textbf{Tempo (s)} \\
\midrule
"""
    
    # Adicionar dados para cada problema
    for problem in all_problems:
        # Buscar dados de cada método para este problema
        lbfgs_item = next((item for item in lbfgs_data if item['problem'] == problem), None)
        descent_item = next((item for item in descent_data if item['problem'] == problem), None)
        mirror_item = next((item for item in mirror_data if item['problem'] == problem), None)
        
        # Obter número de variáveis (pegar de qualquer método que tenha o problema)
        n_vars = "---"
        if lbfgs_item:
            n_vars = lbfgs_item['n_variables']
        elif descent_item:
            n_vars = descent_item['n_variables']
        elif mirror_item:
            n_vars = mirror_item['n_variables']
        
        # L-BFGS-B
        if lbfgs_item:
            lbfgs_iter = lbfgs_item['iterations']
            lbfgs_value = lbfgs_item['min_value']
            lbfgs_time = lbfgs_item['time']
        else:
            lbfgs_iter = lbfgs_value = lbfgs_time = "---"
        
        # Descent Coordinate
        if descent_item:
            descent_iter = descent_item['iterations']
            descent_value = descent_item['min_value']
            descent_time = descent_item['time']
        else:
            descent_iter = descent_value = descent_time = "---"
        
        # Mirror Gradient
        if mirror_item:
            mirror_iter = mirror_item['iterations']
            mirror_value = mirror_item['min_value']
            mirror_time = mirror_item['time']
        else:
            mirror_iter = mirror_value = mirror_time = "---"
        
        # Adicionar linha à tabela
        latex_content += f"{problem} & {n_vars} & {lbfgs_iter} & {lbfgs_value} & {lbfgs_time} & {descent_iter} & {descent_value} & {descent_time} & {mirror_iter} & {mirror_value} & {mirror_time} \\\\\n"
    
    latex_content += r"""
\bottomrule
\end{tabular}
\end{table}
\end{landscape}

\section{Análise dos Resultados}

\subsection{Observações Gerais}

\begin{itemize}
\item \textbf{L-BFGS-B}: Apresenta boa convergência para a maioria dos problemas, com número moderado de iterações e tempos de execução razoáveis.
\item \textbf{Descent Coordinate}: Demonstra convergência muito rápida (3 iterações para a maioria dos problemas), mas com tempos de execução variáveis.
\item \textbf{Mirror Gradient}: Utiliza 1000 iterações para a maioria dos problemas, indicando possível necessidade de ajuste de parâmetros ou critério de parada.
\end{itemize}

\subsection{Problemas de Destaque}

\begin{itemize}
\item \textbf{EXTENDED POWELL}: Todos os métodos convergem rapidamente para o valor ótimo (0.000000e+00).
\item \textbf{ULTS0}: O Mirror Gradient apresenta melhor valor mínimo final comparado aos outros métodos.
\item \textbf{FREUDENTHAL ROTH}: L-BFGS-B e Descent Coordinate apresentam resultados similares, enquanto Mirror Gradient diverge significativamente.
\end{itemize}

\subsection{Performance por Método}

\begin{itemize}
\item \textbf{Eficiência em Iterações}: Descent Coordinate > L-BFGS-B > Mirror Gradient
\item \textbf{Tempo de Execução}: L-BFGS-B apresenta os menores tempos na maioria dos casos
\item \textbf{Precisão}: L-BFGS-B e Descent Coordinate apresentam valores mínimos mais precisos para a maioria dos problemas
\end{itemize}

\end{document}
"""
    
    return latex_content

def generate_comparison_table():
    """
    Função principal para gerar a tabela de comparação.
    """
    print("=" * 60)
    print("GERADOR DE TABELA DE COMPARAÇÃO DOS MÉTODOS")
    print("=" * 60)
    
    # Caminhos dos arquivos LaTeX
    lbfgs_file = "liu_nocedal/latex_solution/resultados_lbfgsb.tex"
    descent_file = "liu_nocedal/latex_solution/resultados_descent_coordinate.tex"
    mirror_file = "liu_nocedal/latex_solution/resultados_mirror_gradient.tex"
    
    print("Extraindo dados dos arquivos LaTeX...")
    
    # Extrair dados de cada arquivo
    lbfgs_data = extract_data_from_latex(lbfgs_file)
    descent_data = extract_data_from_latex(descent_file)
    mirror_data = extract_data_from_latex(mirror_file)
    
    print(f"L-BFGS-B: {len(lbfgs_data)} problemas encontrados")
    print(f"Descent Coordinate: {len(descent_data)} problemas encontrados")
    print(f"Mirror Gradient: {len(mirror_data)} problemas encontrados")
    
    if not lbfgs_data and not descent_data and not mirror_data:
        print("Nenhum dado encontrado. Verifique os arquivos de entrada.")
        return
    
    # Criar documento LaTeX
    print("Gerando documento LaTeX...")
    latex_content = create_comparison_latex(lbfgs_data, descent_data, mirror_data)
    
    # Salvar arquivo LaTeX
    output_file = "liu_nocedal/latex_solution/resultados_comparacao_metodos.tex"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"Arquivo LaTeX salvo em: {output_file}")
    
    # Gerar PDF
    print("Gerando PDF...")
    try:
        # Criar diretório temporário para arquivos auxiliares
        temp_dir = tempfile.mkdtemp()
        
        # Executar pdflatex
        result = subprocess.run([
            'pdflatex',
            '-interaction=nonstopmode',
            '-output-directory', 'liu_nocedal/latex_solution',
            '-aux-directory', temp_dir,
            'liu_nocedal/latex_solution/resultados_comparacao_metodos.tex'
        ], capture_output=True, text=True)
        
        # Limpar diretório temporário
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if result.returncode == 0:
            print("✓ PDF gerado com sucesso!")
            print(f"Arquivo PDF: {output_file.replace('.tex', '.pdf')}")
        else:
            print(f"✗ Erro ao gerar PDF: {result.stderr}")
            
    except FileNotFoundError:
        print("✗ pdflatex não encontrado no PATH")
        print("Instale o MiKTeX ou LaTeX para gerar o PDF")

def main():
    """
    Função principal.
    """
    generate_comparison_table()

if __name__ == "__main__":
    main()
