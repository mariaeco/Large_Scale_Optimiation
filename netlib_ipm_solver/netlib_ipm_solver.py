# Solver em lote para problemas NETLIB usando HiGHS
import os
import highspy as hs
import pandas as pd
from time import perf_counter as pc
from highspy import Highs
import glob
from construct_latex import create_latex_document, create_individual_problem_document
from latex_to_pdf import salvar_pdf #to print general pfd
from netlib_latex_to_pdf import save_pdf #to print individual PROBLEMS pdfs




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
            'x_value': 'N/A',
            'z_value': 'N/A',
            'y_value': 'N/A'
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
                # Variáveis duais das variáveis (col_dual) - custos reduzidos
                col_dual_values = solution.col_dual
                row_dual_values = solution.row_dual
                result['z_value'] = col_dual_values
                result['y_value'] = row_dual_values

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
    
    def save_results_to_csv(self, filename='netlib_ipm_solver/latex_solution/resultados_netlib.csv'):
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
    
    def generate_latex_table(self, filename='netlib_ipm_solver/latex_solution/resultados_netlib.tex'):
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
        latex_content = create_latex_document(df)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"Tabela LaTeX salva em: {filename}")
    
    def generate_individual_problem_files(self, output_dir):
        """
        Gera um arquivo LaTeX individual para cada problema.
        
        Args:
            output_dir (str): Diretório onde salvar os arquivos individuais
        """
        if not self.results:
            print("Nenhum resultado disponível.")
            return
        
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        df = self.generate_results_table()
        
        for _, row in df.iterrows():
            problem_name = row['PROBLEMA']
            filename = os.path.join(output_dir, f"{problem_name}.tex")
            
            # Criar conteúdo LaTeX para o problema individual
            latex_content = create_individual_problem_document(row)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            print(f"Arquivo individual salvo: {filename}")



def main():
    """
    Função principal para executar a análise em lote.
    """
    # Criar solver
    solver = NetlibSolver(problems_dir='netlib_ipm_solver/netlib_problems', solver='ipm')
    
    # Resolver todos os problemas
    solver.solve_all_problems()
    
    # Imprimir resumo
    solver.print_summary()
    
    # Salvar resultados
    solver.save_results_to_csv('netlib_ipm_solver/latex_solution/relatorio_geral_netlib.csv')
    solver.generate_latex_table('netlib_ipm_solver/latex_solution/relatorio_geral_netlib.tex')
    salvar_pdf('netlib_ipm_solver/latex_solution/relatorio_geral_netlib.tex', 'netlib_ipm_solver/latex_solution/')
    
    # Gerar arquivos individuais para cada problema
    print("\nGerando arquivos individuais para cada problema...")
    solver.generate_individual_problem_files('netlib_ipm_solver/latex_solution/relatorio_individual_problems_tex/')
    save_pdf('netlib_ipm_solver/latex_solution/relatorio_individual_problems_tex/', 
        'netlib_ipm_solver/latex_solution/relatorio_individual_problems_pdf/')
    
    print("\nAnálise NETLIB concluída!")


if __name__ == "__main__":
    main()
