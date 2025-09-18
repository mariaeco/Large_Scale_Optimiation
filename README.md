# Large Scale Optimization

Este repositório implementa e compara diferentes métodos de otimização para problemas de larga escala, incluindo problemas da coleção NETLIB e problemas de Liu e Nocedal. O projeto foca em métodos eficientes para otimização não-linear e programação linear.

## 📁 Estrutura do Projeto

### 🔧 **Arquivos Principais**

#### `netlib_ipm_solver/netlib_ipm_solver.py`
- **Função**: Resolve problemas da coleção NETLIB usando o solver HiGHS
- **Método**: Interior Point Method (IPM)
- **Entrada**: Arquivos .mps da coleção NETLIB (82 problemas)
- **Saída**: 
  - Arquivo principal com tabelas gerais (`resultados_netlib.tex` e `pdf`)
  - Arquivos individuais para cada problema (`individual_problems/*.tex` e `pdf`)
  - Extração de variáveis primais e duais
- **Uso**: `python netlib_ipm_solver.py`

#### `liu_nocedal/lbfgsb_solver.py`
- **Função**: Implementa o método L-BFGS-B (Limited-memory BFGS with Bounds)
- **Problemas**: Coleção de 16 problemas reportados em Liu e Nocedal (1989, Table 1) - Problemas de otimização não-linear irrestritos
- **Características**: Método quasi-Newton com memória limitada
- **Saída**: Resultados detalhados em formato LaTeX e PDF com valores das variáveis
- **Uso**: `python lbfgsb_solver.py`

#### `liu_nocedal/mirror_gradient_optimized.py`
- **Função**: Implementa o algoritmo de Gradiente Espelhado com divergências de Bregman
- **Problemas**: Coleção de Liu e Nocedal
- **Características**: 
  - Divergências de Bregman (euclidiana, entropia, norma-p)
  - Passo adaptativo
  - Restrições automáticas (bola, simplex, caixa)
- **Saída**: Resultados detalhados em formato LaTeX e PDF
- **Uso**: `python mirror_gradient_optimized.py`

#### `liu_nocedal/descent_coordinate_algorithm.py`
- **Função**: Implementa o algoritmo de Descida por Coordenadas otimizado
- **Problemas**: Coleção de Liu e Nocedal
- **Características**: 
  - Seleção aleatória de blocos de coordenadas
  - Minimização direta por blocos
  - Tamanho de bloco configurável (padrão: 10)
- **Saída**: Resultados detalhados em formato LaTeX e PDF
- **Uso**: `python descent_coordinate_algorithm.py`

### 🧪 **Arquivos e Utilitários**


#### `liu_nocedal/latex_to_pdf.py`
- **Função**: Converte arquivos LaTeX em PDF com tabelas detalhadas
- **Características**: Gera tabelas com valores das variáveis em múltiplas colunas
- **Uso**: Automático (chamado pelos solvers)

## 📂 **Diretórios**

### `netlib_ipm_solver/netlib_problems/`
- Contém os arquivos .mps da coleção NETLIB
- 82 problemas de programação linear

### `liu_nocedal/problems/`
- Contém a definição dos problemas de Liu e Nocedal
- `setup_problems.py`: Implementação das funções objetivo e configurações

### `liu_nocedal/latex_solution/`
- **`resultados_*.tex`**: Resultados detalhados de cada método
- **`resultados_*_detalhado.tex`**: Tabelas com valores das variáveis
- **PDFs gerados automaticamente**

### `netlib_ipm_solver/latex_solution/`
- **`resultados_netlib.tex`**: Arquivo principal com tabelas gerais
- **`individual_problems/`**: Arquivos LaTeX individuais para cada problema
- **`individual_problems_pdf/`**: PDFs gerados dos arquivos individuais


## 📋 **Dependências**

### Principais:
- `numpy`: Computação numérica
- `scipy`: Otimização e funções científicas
- `highspy`: Solver HiGHS para programação linear

### Para geração de relatórios:
- `pdflatex`: Conversão LaTeX para PDF (MiKTeX no Windows)
- `subprocess`: Execução de comandos do sistema


