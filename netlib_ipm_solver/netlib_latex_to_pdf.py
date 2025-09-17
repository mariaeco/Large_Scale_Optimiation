"""
Script para converter arquivos LaTeX individuais em PDF
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path


# Adicionar MiKTeX ao PATH temporariamente
miktex_path = r"C:\Program Files\MiKTeX\miktex\bin\x64"
os.environ["PATH"] = miktex_path + os.pathsep + os.environ["PATH"]


def save_pdf(problems_dir, output_dir):
    """
    Função principal
    """
    print("=" * 60)
    print("CONVERSOR LaTeX PARA PDF - SOLUÇÕES NETLIB")
    print("=" * 60)

    for file in os.listdir(problems_dir):
        if file.endswith('.tex'):
            print(f"Convertendo: {file}")
            convert_with_subprocess(os.path.join(problems_dir, file), output_dir)

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


