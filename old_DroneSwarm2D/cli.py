# droneswarm2d/cli.py
import argparse
import shutil
import sys
from pathlib import Path


def get_template_dir():
    """Retorna o diretório de templates da biblioteca."""



def init_project(project_name: str = "droneSwarm2D-project", template: str = "droneSwarm2D-project") -> bool:
    """
    Inicializa um novo projeto DroneSwarm2D.
    
    Args:
        project_name: Nome do projeto/diretório
        template: Tipo de template (basic, advanced)
    """
    # Criar diretório do projeto
    project_path = Path.cwd() / project_name
    
    if project_path.exists():
        print(f"\n❌ Erro: Diretório '{project_name}' já existe!\n\n")
        return False
    
    package_dir = Path(__file__).parent / "templates"
    template_dir = package_dir / template
        
    print(f"\n\n📁 Criando projeto '{project_name}'...")
    
    shutil.copytree(template_dir, project_path)
    print(f"✅ Projeto criado com sucesso em: {project_path}")
    print(f"\n📝 Próximos passos:")
    print(f"\tcd {project_name}\n\n")

    return True

def main():
    init_project()


if __name__ == "__main__":
    main()