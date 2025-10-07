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


def save_pdf(input_dir, output_dir):
    """
    Converte múltiplos arquivos LaTeX para PDF.
    
    Args:
        input_dir (str): Diretório com arquivos .tex
        output_dir (str): Diretório de saída para PDFs
    """
    print("=" * 60)
    print("CONVERSOR MÚLTIPLOS LaTeX PARA PDF")
    print("=" * 60)
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Encontrar todos os arquivos .tex
    tex_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.tex'):
                tex_files.append(os.path.join(root, file))
    
    print(f"Encontrados {len(tex_files)} arquivos LaTeX para converter...")
    
    success_count = 0
    for i, tex_file in enumerate(tex_files, 1):
        print(f"\n[{i}/{len(tex_files)}] Convertendo: {os.path.basename(tex_file)}")
        
        try:
            convert_with_subprocess(tex_file, output_dir)
            success_count += 1
        except Exception as e:
            print(f"✗ Erro ao converter {tex_file}: {e}")
    
    print(f"\n✓ Conversão concluída: {success_count}/{len(tex_files)} arquivos convertidos com sucesso!")
