import numpy as np
from numpy import zeros
from problems.setup_problems import*
import time
from latex_to_pdf import salvar_pdf, generate_latex_table, generate_detailed_latex_table


def calcular_gradiente(f, x):
    """
    Calcula gradiente numericamente usando diferenças finitas centrais
    """
    n = len(x)
    grad = zeros(n)
    
    for i in range(n):
        h = 1e-6
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad


# =============================================================================
# DIVERGÊNCIAS DE BREGMAN
# =============================================================================

def euclidean_phi(x):
    """Função potencial para divergência euclidiana: φ(x) = (1/2)||x||²"""
    return 0.5 * np.sum(x**2)

def euclidean_grad_phi(x):
    """Gradiente da função potencial euclidiana: ∇φ(x) = x"""
    return x

def entropy_phi(x):
    """Função potencial para entropia: φ(x) = Σᵢ xᵢ log(xᵢ)"""
    x_safe = np.maximum(x, 1e-10)  # Evitar log(0)
    return np.sum(x_safe * np.log(x_safe))

def entropy_grad_phi(x):
    """Gradiente da função potencial de entropia: ∇φ(x) = 1 + log(x)"""
    x_safe = np.maximum(x, 1e-10)
    return 1 + np.log(x_safe)

def p_norm_phi(x, p=2):
    """Função potencial para norma-p: φ(x) = (1/p)||x||ᵖᵖ"""
    return (1/p) * np.sum(np.abs(x)**p)

def p_norm_grad_phi(x, p=2):
    """Gradiente da função potencial norma-p: ∇φ(x) = sign(x) * |x|^(p-1)"""
    return np.sign(x) * (np.abs(x) ** (p - 1))

# =============================================================================
# GRADIENTE ESPELHADO 
# =============================================================================



def gradiente_espelhado(f, x0, eta_inicial=0.01, max_iter=1000, tol=1e-6, bregman="euclidean", bounds=None, p=2):
    """
    Algoritmo de Gradiente Espelhado com passo adaptativo e divergências de Bregman corretas
    
    Parâmetros:
    - f: função objetivo
    - x0: ponto inicial
    - eta_inicial: passo inicial
    - max_iter: máximo de iterações
    - tol: tolerância para convergência
    - bregman: tipo de divergência de Bregman ("euclidean", "entropy", "p_norm")
    - bounds: tipo de restrição
    - p: parâmetro para norma-p
    """
    
    x = x0.copy()
    eta = eta_inicial
    fo = f(x0)
    last_improvement = 0
    
    # Determinar tipo de restrição
    if bounds is None:
        constraint_type = "ball"
        # Normalizar ponto inicial para dentro da bola unitária
        x_norm = np.linalg.norm(x)
        if x_norm > 1.0:
            x = x / x_norm * 0.9
    elif isinstance(bounds, list):
        constraint_type = "box"
    else:
        constraint_type = bounds
    
    for k in range(max_iter):
        # 1. Calcular gradiente da função objetivo
        grad_f = calcular_gradiente(f, x)
        
        # 2. Verificar convergência (tolerância mais realista)
        # if np.linalg.norm(grad_f) <= tol:
        #     break
        
        # 3. Aplicar Mirror Descent com divergências de Bregman corretas
        if bregman == "euclidean":
            x_new = x - eta * grad_f
            if constraint_type == "ball":
                x_norm = np.linalg.norm(x_new)
                if x_norm > 1.0:
                    x_new = x_new / x_norm * 0.99
                    
        elif bregman == "entropy":
            x_safe = np.maximum(x, 1e-10)
            x_new = np.exp(np.log(x_safe) - eta * grad_f)
            if constraint_type == "simplex":
                x_new = x_new / np.sum(x_new)
                
        elif bregman == "p_norm":
            if p == 2:
                x_new = x - eta * grad_f
            else:
                grad_phi_x = p_norm_grad_phi(x, p)
                grad_phi_new = grad_phi_x - eta * grad_f
                x_new = np.sign(grad_phi_new) * (np.abs(grad_phi_new) ** (1 / (p - 1)))
            
            if constraint_type == "ball":
                p_norm = np.sum(np.abs(x_new) ** p) ** (1/p)
                if p_norm > 1.0:
                    x_new = x_new / p_norm * 0.99
                    
        elif constraint_type == "box":
            x_new = x - eta * grad_f
            for i, (lower, upper) in enumerate(bounds):
                x_new[i] = np.clip(x_new[i], lower, upper)
        else:
            x_new = x - eta * grad_f
        
        # 4. Verificar se houve melhoria
        fo_new = f(x_new)
        improvement = fo - fo_new
        
        if improvement > 1e-8:
            # Boa melhoria: aceitar e aumentar passo
            x = x_new
            fo = fo_new
            last_improvement = k
            eta = min(eta * 1.2, 2.0)
        else:
            # Pouca melhoria: diminuir passo
            eta = eta * 0.15
            if k - last_improvement > 5:
                eta = max(eta, 1e-8)
        
        # # 5. Verificar convergência pela mudança no ponto (tolerância mais realista)
        if np.linalg.norm(x_new - x) < tol:
            break
    
    return x, fo, k


class MirrorGradientOptimizedSolver:
    """
    Classe para resolver problemas de otimização usando o método adaptativo do Gradiente Espelhado
    e gerar tabelas de resultados em formato LaTeX.
    """
    
    def __init__(self, eta=0.01, bregman="euclidean", p=2):
        """
        Inicializa o solver com gradiente espelhado adaptativo.
        
        Args:
            eta (float): Parâmetro de passo inicial
            bregman (str): Tipo de divergência de Bregman ('euclidean', 'entropy', 'p_norm')
            p (int): Parâmetro para norma-p
        """
        self.results = []
        self.problems_config = setup_problems()
        self.eta = eta
        self.bregman = bregman
        self.p = p
    
    def solve_problem(self, problem_name, max_iter=1000, tol=1e-6):
        """
        Resolve um problema específico usando Gradiente Espelhado Adaptativo com restrições.
        
        Args:
            problem_name (str): Nome do problema
            max_iter (int): Número máximo de iterações
            tol (float): Tolerância para convergência
            
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
            bounds = config['bounds']  # ULTS0 tem restrições de caixa
            
            # Função objetivo com argumentos
            def objective_with_args(x):
                return config['objective'](x, grid_shape, h)
        else:
            n, x0 = config['setup']()
            args = config['args']
            bounds = config['bounds']  # Pode ser None
            objective_with_args = config['objective']
        
        # Determinar tipo de restrição apropriado para cada problema
        constraint_type = self._determine_constraint_type(problem_name, bounds)
        
        # Executar otimização
        start_time = time.time()
        
        try:
            x_md, fo, iterations = gradiente_espelhado(
                objective_with_args, x0, self.eta, max_iter, tol, self.bregman, constraint_type, self.p
            )
            
            end_time = time.time()
            
            # Calcular norma do gradiente final
            grad_norm_final = calcular_gradiente(objective_with_args, x_md)
            grad_norm = np.linalg.norm(grad_norm_final)
            
            # Armazenar resultados
            result_dict = {
                'problem': problem_name,
                'success': True,
                'iterations': iterations + 1,
                'function_value': fo,
                'x_value': x_md,
                'gradient_norm': grad_norm,
                'message': f"Convergência atingida (restrição: {constraint_type})",
                'execution_time': end_time - start_time,
                'n_variables': n,
                'constraint_type': constraint_type
            }
            
        except Exception as e:
            end_time = time.time()
            result_dict = {
                'problem': problem_name,
                'success': False,
                'iterations': 0,
                'function_value': float('inf'),
                'x_value': None,
                'gradient_norm': float('inf'),
                'message': f"Erro: {str(e)}",
                'execution_time': end_time - start_time,
                'n_variables': n,
                'constraint_type': constraint_type
            }
        
        return result_dict
    
    def _determine_constraint_type(self, problem_name, bounds):
        """
        Determina o tipo de restrição apropriado para cada problema.
        """
        if bounds is not None and isinstance(bounds, list):
            return "box"  # Restrições de caixa
        
        # Mapear problemas para tipos de restrição apropriados
        constraint_mapping = {
            'ROSENBROCK': 'simplex',          # Problema de Rosenbrock na bola unitária
            'EXTENDED_ROSENBROCK': 'ball',  # Problema de Rosenbrock estendido na bola unitária
            'EXTENDED_POWELL': 'ball',      # Problema de Powell estendido na bola unitária
            'FREUDENTHAL_ROTH': 'ball',     # Problema de Freudenthal-Roth na bola unitária
            'ENGGVAL1': 'ball',             # Problema de Engvall na bola unitária
            'TRIGONOMETRIC': 'simplex',  # Problema trigonométrico funciona bem no simplex
            'PENALTY': 'ball',          # Problema de penalidade na bola unitária
            'QOR': 'ball',              # Quadrático com restrições na bola
            'GOR': 'ball',              # Quadrático com restrições na bola
            'PSP': 'ball',              # Problema de penalidade na bola
            'LINEAR_MINIMUM_SURFACE': 'ball',  # Superfície mínima na bola
            'SQUARE_ROOT_1': 'ball',    # Raiz quadrada na bola
            'SQUARE_ROOT_2': 'ball',    # Raiz quadrada na bola
            'SPARSE_MATRIX_SQRT': 'ball',  # Matriz esparsa na bola
            'ULTS0': 'box',             # ULTS0 já tem restrições de caixa
        }
        
        return constraint_mapping.get(problem_name, 'box')  # Padrão: bola unitária
    
    def solve_all_problems(self, max_iter=1000, tol=1e-6):
        """
        Resolve todos os problemas configurados.
        
        Args:
            max_iter (int): Número máximo de iterações
            tol (float): Tolerância para convergência
        """
        print(f"Iniciando resolução de todos os problemas com Gradiente Espelhado ({self.bregman})...")
        print(f"  - Modo: Adaptativo")
        print(f"  - Passo inicial: {self.eta}")
        print("=" * 70)
        
        self.results = []
        
        for i, problem_name in enumerate(self.problems_config.keys(), 1):
            print(f"\n[{i}/{len(self.problems_config)}] Resolvendo: {problem_name}")
            
            result = self.solve_problem(problem_name, max_iter, tol)
            self.results.append(result)
            
            # Imprimir resultado
            if result['success']:
                print(f"  ✓ Sucesso: {result['iterations']} iterações, f* = {result['function_value']:.6e}")
            else:
                print(f"  ✗ Falhou: {result['message']}")
        
        print("\n" + "=" * 70)
        print("Resolução concluída!")
    
    def print_summary(self):
        """
        Imprime um resumo dos resultados.
        """
        if not self.results:
            print("Nenhum resultado disponível.")
            return
        
        print("\n" + "=" * 80)
        print("RESUMO DOS RESULTADOS - GRADIENTE ESPELHADO ADAPTATIVO")
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
    Função principal para executar a análise com gradiente espelhado adaptativo.
    """
    print("=" * 80)
    print("GRADIENTE ESPELHADO")
    print("=" * 80)
    
    # Criar solver com tipo de Bregman específico
    solver = MirrorGradientOptimizedSolver(eta=0.01, bregman='p_norm', p=2) #bregman_type: euclidean, entropy, p_norm
    
    # Resolver todos os problemas
    solver.solve_all_problems(max_iter=1000, tol=1e-6)
    
    # Imprimir resumo
    solver.print_summary()
    
    # Armazenar resultados
    all_results = solver.results
    
     # Gerar tabela LaTeX específica
    method_name = f'Gradiente Espelhado'
    detailed_filename = f'liu_nocedal/latex_solution/resultados_mirror_gradient.tex'
    generate_detailed_latex_table(solver.results, detailed_filename, method_name)
    salvar_pdf(detailed_filename, 'liu_nocedal/latex_solution/')
    

if __name__ == "__main__":
    # Para executar apenas um exemplo simples, descomente a linha abaixo:
    # exemplo_uso()
    
    # Para executar a análise completa, use:
    main()
