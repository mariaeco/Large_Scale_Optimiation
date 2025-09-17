# Projeto de Otimização em Larga Escala

Este projeto implementa e compara diferentes métodos de otimização para problemas de larga escala, incluindo problemas da coleção NETLIB e problemas de Liu e Nocedal.

## 📁 Estrutura do Projeto

### 🔧 **Arquivos Principais**

#### `ipm_solver_netlib.py`
- **Função**: Resolve problemas da coleção NETLIB usando o solver HiGHS
- **Método**: Interior Point Method (IPM)
- **Entrada**: Arquivos .mps da coleção NETLIB
- **Saída**: 
  - Arquivo principal com tabelas gerais (`resultados_netlib.tex`)
  - Arquivos individuais para cada problema (`individual_problems/*.tex`)
  - Extração de variáveis primais e duais
- **Uso**: `python ipm_solver_netlib.py`

#### `comparison_solver.py`
- **Função**: Compara três métodos de otimização nos problemas de Liu e Nocedal
- **Métodos comparados**: L-BFGS-B, Mirror Gradient, Coordinate Descent
- **Saída**: Tabelas comparativas em LaTeX
- **Uso**: `python comparison_solver.py`

#### `lbfgsb_solver_liu_nocedal.py`
- **Função**: Implementa o método L-BFGS-B (Limited-memory BFGS with Bounds)
- **Problemas**: Coleção de Liu e Nocedal
- **Características**: Método quasi-Newton com memória limitada
- **Saída**: Resultados em formato LaTeX

#### `mirror_gradient_algorith.py`
- **Função**: Implementa o algoritmo Mirror Gradient
- **Problemas**: Coleção de Liu e Nocedal
- **Características**: Método de gradiente espelhado para otimização
- **Saída**: Resultados em formato LaTeX

#### `descent_coordinate_algorithm.py`
- **Função**: Implementa o algoritmo de Coordinate Descent
- **Problemas**: Coleção de Liu e Nocedal
- **Características**: Otimização coordenada por coordenada
- **Saída**: Resultados em formato LaTeX

#### `highs_solver_liu_nocedal.py`
- **Função**: Resolve problemas de Liu e Nocedal usando o solver HiGHS
- **Método**: Interior Point Method
- **Problemas**: Coleção de Liu e Nocedal
- **Saída**: Resultados em formato LaTeX

### 🧪 **Arquivos de Teste**

#### `test_comparison.py`
- **Função**: Testa o comparador de métodos com problemas selecionados
- **Uso**: `python test_comparison.py`

### 📊 **Arquivos de Conversão**

#### `netlib_latex_to_pdf.py`
- **Função**: Converte arquivos LaTeX individuais em PDF
- **Uso**: `python netlib_latex_to_pdf.py`

## 📂 **Diretórios**

### `netlib_problems/`
- Contém os arquivos .mps da coleção NETLIB
- 82 problemas de programação linear

### `liu_nocedal_problems/`
- Contém a definição dos problemas de Liu e Nocedal
- `problems.py`: Implementação das funções objetivo

### `latex_solution/`
- **`resultados_netlib.tex`**: Arquivo principal com tabelas gerais
- **`individual_problems/`**: Arquivos LaTeX individuais para cada problema
- **`individual_problems_pdf/`**: PDFs gerados dos arquivos individuais
- **`resultados_*.tex`**: Resultados específicos de cada método

### `RELATORIOS/`
- Relatórios em PDF com análises detalhadas de cada método

## 🚀 **Como Usar**

### 1. **Resolver problemas NETLIB:**
```bash
python ipm_solver_netlib.py
```

### 2. **Comparar métodos de otimização:**
```bash
python comparison_solver.py
```

### 3. **Testar comparador:**
```bash
python test_comparison.py
```

### 4. **Converter LaTeX para PDF:**
```bash
python netlib_latex_to_pdf.py
```

## 📋 **Dependências**

- `numpy`: Computação numérica
- `scipy`: Otimização e funções científicas
- `highspy`: Solver HiGHS para programação linear
- `pandas`: Manipulação de dados
- `pdflatex`: Conversão LaTeX para PDF (biblioteca Python)

## 🎯 **Objetivos**

1. **Implementar** diferentes métodos de otimização
2. **Comparar** performance em problemas de larga escala
3. **Gerar** relatórios automatizados em LaTeX
4. **Extrair** variáveis primais e duais das soluções
5. **Documentar** resultados de forma sistemática

## 📈 **Métodos Implementados**

- **L-BFGS-B**: Método quasi-Newton com memória limitada
- **Mirror Gradient**: Gradiente espelhado
- **Coordinate Descent**: Descente coordenada
- **Interior Point Method**: Método de pontos interiores (HiGHS)

## 🔍 **Coleções de Problemas**

- **NETLIB**: 82 problemas de programação linear
- **Liu e Nocedal**: Problemas de otimização não-linear
