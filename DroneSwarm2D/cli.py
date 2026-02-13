# droneswarm2d/cli.py
"""
Interface de linha de comando para DroneSwarm2D.
Fornece comandos para inicializar projetos exemplo e criar vídeos a partir de frames.
"""
import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional
from .tools.create_video import create_video_from_frames

def get_template_dir() -> Path:
    """
    Retorna o diretório de templates da biblioteca.
    
    Returns:
        Path: Caminho para o diretório de templates.
    """
    return Path(__file__).parent / "templates"


def init_project(
    project_name: str = "droneSwarm2D-project", 
    template: str = "droneSwarm2D-project"
) -> bool:
    """
    Inicializa um novo projeto DroneSwarm2D.
    
    Copia a estrutura de template para um novo diretório com o nome especificado.
    Se o diretório já existir, retorna False sem fazer alterações.
    
    Args:
        project_name: Nome do projeto/diretório a ser criado.
        template: Tipo de template a ser usado (ex: 'basic', 'advanced').
        
    Returns:
        bool: True se o projeto foi criado com sucesso, False caso contrário.
        
    Example:
        >>> init_project("meu_projeto", "basic")
        True
    """
    # Criar diretório do projeto
    project_path: Path = Path.cwd() / project_name
    
    if project_path.exists():
        print(f"\n❌ Erro: Diretório '{project_name}' já existe!\n")
        return False
    
    # Localizar diretório de templates
    package_dir: Path = Path(__file__).parent / "templates"
    template_dir: Path = package_dir / template
    
    if not template_dir.exists():
        print(f"\n❌ Erro: Template '{template}' não encontrado em {package_dir}!\n")
        return False
        
    print(f"\n📁 Criando projeto '{project_name}'...")
    
    try:
        shutil.copytree(template_dir, project_path)
        print(f"✅ Projeto criado com sucesso em: {project_path}")
        print(f"\n📝 Próximos passos:")
        print(f"\tcd {project_name}\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro ao criar projeto: {e}\n")
        return False


def create_video_command(args: argparse.Namespace) -> None:
    """
    Executa o comando create_video.
    
    Args:
        args: Argumentos parseados do argparse contendo path, output_dir, fps, codec e remove_frames.
    """    
    print(f"\n🎬 Criando vídeo a partir dos frames em: {args.path}")
    
    video_path = create_video_from_frames(
        frames_dir=args.path,
        output_dir=args.output_dir,
        fps=args.fps,
        codec=args.codec,
        remove_frames=args.remove_frames
    )
    
    if video_path:
        print(f"✅ Vídeo criado com sucesso: {video_path}\n")
        sys.exit(0)
    else:
        print("\n❌ Erro: Falha ao criar o vídeo.\n")
        sys.exit(1)


def main() -> None:
    """
    Função principal da CLI com suporte a múltiplos comandos.
    
    Comandos disponíveis:
        - init: Inicializa um novo projeto DroneSwarm2D
        - create_video: Cria um vídeo a partir de frames PNG
    """
    parser = argparse.ArgumentParser(
        prog='DroneSwarm2D',
        description='DroneSwarm2D - Simulação de Enxame de Drones Defensivos'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponíveis')
    
    # Subcomando: init
    init_parser = subparsers.add_parser(
        'init',
        help='Inicializa um novo projeto DroneSwarm2D'
    )
    init_parser.add_argument(
        '--name',
        type=str,
        default='droneSwarm2D-project',
        help='Nome do projeto (default: droneSwarm2D-project)'
    )
    init_parser.add_argument(
        '--template',
        type=str,
        default='droneSwarm2D-project',
        help='Tipo de template (default: droneSwarm2D-project)'
    )
    
    # Subcomando: create_video
    video_parser = subparsers.add_parser(
        'create_video',
        help='Cria um vídeo a partir de frames PNG'
    )
    video_parser.add_argument(
        'path',
        type=str,
        help='Caminho para o diretório contendo os frames PNG'
    )
    video_parser.add_argument(
        '--output-dir',
        type=str,
        default='./video',
        help='Diretório onde o vídeo será salvo (default: ./video)'
    )
    video_parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames por segundo do vídeo (default: 30)'
    )
    video_parser.add_argument(
        '--codec',
        type=str,
        default='libx264',
        help='Codec de vídeo a ser usado (default: libx264)'
    )
    video_parser.add_argument(
        '--remove-frames',
        action='store_true',
        help='Remove o diretório de frames após criar o vídeo'
    )
    
    # Parse dos argumentos
    args = parser.parse_args()
    
    # Se nenhum comando foi fornecido, executa init com valores padrão (comportamento original)
    if args.command is None:
        print("\n📦 Nenhum comando especificado. Executando 'init' com valores padrão...\n")
        init_project()
        return
    
    # Executa o comando apropriado
    if args.command == 'init':
        success = init_project(project_name=args.name, template=args.template)
        sys.exit(0 if success else 1)
        
    elif args.command == 'create_video':
        create_video_command(args)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()