import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

from matplotlib.patches import Patch
from scipy import stats

# Configurar tema
sns.set_theme(style="ticks", palette="pastel")

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.family'] = 'sans-serif'


def violinplot_with_bootstrap_test(data, filter_conditions=None, x_col=None, y_col=None, hue_col=None, 
                                  title="", subtitle="", xlabel="Categories", ylabel="Distribution", 
                                  legend_title=None, legend_labels=None, n_bootstrap=10000, alpha=0.05, figsize=(12, 7),
                                  split=True, gap=0.03, inner="quart", fill=True, linewidth=1.25, width=1,
                                  palette=["m", "g"], common_norm=False, density_norm="area", bw_adjust=1.5,
                                  x_order=None, ylim_bottom=None, ylim_top=None, save_fig=False, save_dir="./plots", filename=None):
    """
    Cria um violinplot com teste bootstrap automático e caixa de resultado colorida
    
    Parameters:
    -----------
    data : DataFrame
        Dados para plotar
    filter_conditions : dict
        Condições de filtro no formato {'coluna': valor}
    x_col, y_col, hue_col : str
        Nomes das colunas para x, y e hue
    title : str
        Título principal do gráfico
    subtitle : str
        Subtítulo (opcional, aparece abaixo do título principal)
    xlabel, ylabel : str
        Labels dos eixos
    legend_title : str
        Título da legenda (se None, usa o nome da coluna hue)
    legend_labels : dict ou None
        Dicionário para mapear valores da variável hue para labels customizados
        Ex: {True: 'Sim', False: 'Não'} ou {'A': 'Grupo A', 'B': 'Grupo B'}
    n_bootstrap : int
        Número de iterações bootstrap
    alpha : float
        Nível de significância
    figsize : tuple
        Tamanho da figura
    split : bool
        Se True, divide os violinos por hue
    gap : float
        Espaço entre violinos
    inner : str
        Tipo de marcação interna ("quart", "box", "point", "stick", None)
    fill : bool
        Se True, preenche os violinos
    linewidth : float
        Espessura das linhas
    palette : list ou str
        Cores dos violinos
    common_norm : bool
        Se True, normaliza todos os violinos juntos; se False, cada violino é normalizado independentemente
    density_norm : str
        Tipo de normalização da densidade ("area", "count", "width")
        - "area": cada violino tem área total igual (recomendado)
        - "count": normalização baseada no número de observações  
        - "width": todos os violinos têm largura máxima igual
    x_order : list ou None
        Ordem específica dos elementos no eixo X (ex: ['1:1', '0.7:1', '1:0.7'])
    ylim_bottom : float ou None
        Limite inferior do eixo Y
    ylim_top : float ou None
        Limite superior do eixo Y
    save_fig : bool
        Se True, salva o gráfico em PNG (300 DPI) e EPS
    save_dir : str
        Diretório onde salvar os arquivos
    filename : str
        Nome do arquivo (sem extensão). Se None, usa o título limpo
    
    Returns:
    --------
    dict : Resultado do teste bootstrap
    """
    
    def bootstrap_test(data1, data2):
        """Teste bootstrap interno"""
        data1, data2 = data1.dropna(), data2.dropna()
        
        if len(data1) == 0 or len(data2) == 0:
            return {'p_value': np.nan, 'significant': False, 'n1': 0, 'n2': 0, 'real_difference': np.nan}
        
        real_diff = np.mean(data1) - np.mean(data2)
        combined_data = np.concatenate([data1, data2])
        n1, n2 = len(data1), len(data2)
        
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            resampled = np.random.choice(combined_data, size=n1+n2, replace=True)
            boot_diff = np.mean(resampled[:n1]) - np.mean(resampled[n1:])
            bootstrap_diffs.append(boot_diff)
        
        # p_value = np.sum(np.abs(bootstrap_diffs) >= np.abs(real_diff)) / n_bootstrap
        p_value = (np.sum(np.abs(bootstrap_diffs) >= np.abs(real_diff)) + 1) / (n_bootstrap + 1)
        
        return {
            'p_value': p_value,
            'significant': p_value < alpha,
            'real_difference': real_diff,
            'n1': n1,
            'n2': n2
        }
    
    def add_multiple_result_boxes(ax, test_results):
        """Adiciona múltiplas caixas coloridas centralizadas"""
        if not test_results:
            return
            
        # Determinar posições das caixas (centralizadas)
        n_boxes = len(test_results)
        box_width = 0.18  # Largura de cada caixa
        box_height = 0.20  # Altura das caixas
        
        # Calcular posição inicial para centralizar as caixas
        total_width = n_boxes * box_width + (n_boxes - 1) * 0.01  # largura total + espaçamentos
        start_x = (1.0 - total_width) / 2  # centralizar
        y_position = 0.02  # Posição Y fixa
        
        for i, (category, test_result) in enumerate(test_results.items()):
            x_position = start_x + (i * (box_width + 0.01))  # Espaçamento entre caixas
            
            # Determinar cor da caixa
            if np.isnan(test_result['p_value']):
                color, text_color = 'gray', 'white'
                result_text = f"TESTE BOOTSTRAP\nH₀: μ₁ = μ₂\n{category}\nDados insuficientes"
            elif test_result['significant']:
                color, text_color = 'red', 'white'
                result_text = f"TESTE BOOTSTRAP\nH₀: μ₁ = μ₂\n{category}\nH₀ rejeitada\np = {test_result['p_value']:.4e}"
            else:
                color, text_color = 'green', 'white'
                result_text = f"TESTE BOOTSTRAP\nH₀: μ₁ = μ₂\n{category}\nH₀ não rejeitada\np = {test_result['p_value']:.4e}"
            
            # Adicionar caixa seguindo o padrão mostrado
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9)
            ax.text(x_position, y_position, result_text, transform=ax.transAxes, fontsize=9, 
                    verticalalignment='bottom', bbox=bbox_props, color=text_color, weight='bold')
    
    def add_mean_lines(ax, filtered_data, x_col, y_col, hue_col, palette, x_order, ylim_bottom, ylim_top, split=True):
        """Adiciona linhas horizontais da média próximas às curvas dos violinos"""
        if not hue_col:
            return
            
        # Obter categorias na ordem especificada ou padrão
        if x_order:
            categories = [cat for cat in x_order if cat in filtered_data[x_col].unique()]
        else:
            categories = sorted(filtered_data[x_col].unique())
            
        hue_groups = sorted(filtered_data[hue_col].unique())
        
        # Converter palette para cores mais claras
        if isinstance(palette, str):
            colors = ['plum', 'lightgreen'] if len(hue_groups) >= 2 else ['plum']
        else:
            # Mapear cores para versões mais claras
            color_map = {
                'm': 'plum',           # magenta → plum (mais claro)
                'magenta': 'plum',
                'purple': 'plum',
                'g': 'lightgreen',     # green → lightgreen
                'green': 'lightgreen',
                'b': 'lightblue',      # blue → lightblue
                'blue': 'lightblue',
                'r': 'lightcoral',     # red → lightcoral
                'red': 'lightcoral',
                'orange': 'moccasin',
                'yellow': 'lightyellow',
                'brown': 'tan',
                'pink': 'lightpink',
                'gray': 'lightgray',
                'black': 'dimgray'
            }
            
            colors = []
            for color in palette:
                if color in color_map:
                    colors.append(color_map[color])
                else:
                    colors.append(color)  # Manter cor original se não estiver no mapa
        
        # Mapear cores para cada grupo hue (ordem direta para corresponder aos violinos)
        hue_colors = {hue_groups[i]: colors[i] for i in range(min(len(hue_groups), len(colors)))}
        
        for i, category in enumerate(categories):
            category_data = filtered_data[filtered_data[x_col] == category]
            
            for j, hue_group in enumerate(hue_groups):
                group_data = category_data[category_data[hue_col] == hue_group][y_col]
                
                if len(group_data) > 0:
                    mean_value = group_data.mean()
                    color = hue_colors.get(hue_group, 'lightgray')
                    
                    if split:
                        # Para violinos divididos (split=True)
                        if j == 0:  # Primeiro grupo (lado esquerdo)
                            x_start = i - 0.35  # Começa mais à esquerda
                            x_end = i - 0.05    # Termina próximo ao centro
                        else:  # Segundo grupo (lado direito)
                            x_start = i + 0.05  # Começa próximo ao centro
                            x_end = i + 0.35    # Termina mais à direita
                    else:
                        # Para violinos não divididos
                        # Calcular posição baseada no gap e número de grupos
                        violin_width = (1 - gap) / len(hue_groups)
                        x_center = i + (j - (len(hue_groups) - 1) / 2) * (violin_width + gap)
                        x_start = x_center - violin_width * 0.4
                        x_end = x_center + violin_width * 0.4
                    
                    # Linha horizontal da média próxima à curva
                    ax.hlines(mean_value, x_start, x_end, colors=color, linestyles='dashed', 
                             linewidth=1.8, alpha=0.8, zorder=10)
                    
                    # Texto com o valor da média próximo à linha
                    text_x = x_end + 0.02 if split and j == 1 else x_start - 0.02
                    ha_alignment = 'left' if split and j == 1 else 'right'
                    
                    ax.text(text_x, mean_value, f'{mean_value:.1f}', 
                           color=color, fontweight='bold', fontsize=9,
                           ha=ha_alignment, va='center', 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                                   edgecolor=color, alpha=1))
    
    def print_statistics(filtered_data, x_col, y_col, hue_col, x_order, test_results):
        """Imprime estatísticas descritivas e resultados dos testes"""
        print("=" * 80)
        print(f"RELATÓRIO ESTATÍSTICO - {title}")
        print("=" * 80)
        
        # Usar ordem especificada ou padrão
        if x_order:
            categories = [cat for cat in x_order if cat in filtered_data[x_col].unique()]
        else:
            categories = sorted(filtered_data[x_col].unique())
            
        if hue_col and hue_col in filtered_data.columns:
            hue_groups = sorted(filtered_data[hue_col].unique())
            
            print("\nESTATÍSTICAS DESCRITIVAS POR GRUPO:")
            print("-" * 60)
            
            for category in categories:
                print(f"\n📊 CATEGORIA: {category}")
                print("-" * 40)
                
                category_data = filtered_data[filtered_data[x_col] == category]
                
                for hue_group in hue_groups:
                    group_data = category_data[category_data[hue_col] == hue_group][y_col].dropna()
                    
                    if len(group_data) > 0:
                        mean_val = group_data.mean()
                        std_val = group_data.std()
                        median_val = group_data.median()
                        n_val = len(group_data)
                        
                        print(f"  • {hue_group}:")
                        print(f"    - Média: {mean_val:.2f}")
                        print(f"    - Desvio Padrão: {std_val:.2f}")
                        print(f"    - Mediana: {median_val:.2f}")
                        print(f"    - N: {n_val}")
                    else:
                        print(f"  • {hue_group}: Sem dados")
        
        # Resultados dos testes bootstrap
        print("\n" + "=" * 60)
        print("RESULTADOS DOS TESTES BOOTSTRAP")
        print("=" * 60)
        print("H₀: μ₁ = μ₂ (médias são iguais)")
        print("H₁: μ₁ ≠ μ₂ (médias são diferentes)")
        print(f"Nível de significância: α = {alpha}")
        print(f"Número de iterações bootstrap: {n_bootstrap:,}")
        
        for category, test_result in test_results.items():
            print(f"\n🧪 TESTE PARA: {category}")
            print("-" * 40)
            
            if np.isnan(test_result['p_value']):
                print("❌ TESTE NÃO REALIZADO - Dados insuficientes")
            else:
                print(f"p-valor: {test_result['p_value']:.6f}")
                print(f"Diferença real (μ₁ - μ₂): {test_result['real_difference']:.3f}")
                print(f"Tamanho da amostra 1: {test_result['n1']}")
                print(f"Tamanho da amostra 2: {test_result['n2']}")
                
                if test_result['significant']:
                    print("✅ RESULTADO: H₀ REJEITADA - Diferença significativa detectada")
                    print(f"   (p < {alpha})")
                else:
                    print("⭕ RESULTADO: H₀ NÃO REJEITADA - Sem evidência de diferença")
                    print(f"   (p ≥ {alpha})")
        
        print("\n" + "=" * 80)
        
        # Resumo final
        significant_tests = sum(1 for result in test_results.values() if result['significant'])
        total_tests = len([result for result in test_results.values() if not np.isnan(result['p_value'])])
        
        print("RESUMO FINAL:")
        print(f"• Testes realizados: {total_tests}")
        print(f"• Diferenças significativas encontradas: {significant_tests}")
        print(f"• Taxa de diferenças significativas: {significant_tests/total_tests*100:.1f}%" if total_tests > 0 else "• Taxa de diferenças significativas: N/A")
        print("=" * 80)
    
    # Configurar tema
    sns.set_theme(style="ticks", palette="pastel")
    
    # Filtrar dados
    filtered_data = data.copy()
    if filter_conditions:
        for col, value in filter_conditions.items():
            filtered_data = filtered_data[filtered_data[col] == value]
    
    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Violinplot com configurações - aumentando espessura com cut=0 e bw_adjust
    violinplot_kwargs = {
        'ax': ax,
        'data': filtered_data,
        'x': x_col,
        'y': y_col,
        'hue': hue_col,
        'order': x_order,  # Ordem específica do eixo X
        'split': split,
        'gap': gap,
        'inner': inner,  # Garantir que sempre mostre quartis
        'fill': fill,
        'linewidth': linewidth,
        'palette': palette,
        'common_norm': common_norm,
        'density_norm': density_norm,
        'bw_adjust': bw_adjust,
        'width': width,  # Largura dos violinos
    }
    
    sns.violinplot(**violinplot_kwargs)
    
    # Adicionar linhas de média APÓS o violinplot - passando o parâmetro split
    add_mean_lines(ax, filtered_data, x_col, y_col, hue_col, palette, x_order, ylim_bottom, ylim_top, split=split)
    
    # Configurar limites do eixo Y se fornecido
    current_ylim = ax.get_ylim()
    
    # Encontrar o range real dos dados para não cortar distribuições
    data_min = filtered_data[y_col].min()
    data_max = filtered_data[y_col].max()
    
    # Usar limites que não cortam os dados
    new_ylim_bottom = min(ylim_bottom if ylim_bottom is not None else current_ylim[0], data_min - 2)
    new_ylim_top = max(ylim_top if ylim_top is not None else current_ylim[1], data_max + 2)
    
    ax.set_ylim(new_ylim_bottom, new_ylim_top)
    
    # Configurar ticks do eixo Y para respeitar limites definidos pelo usuário
    if ylim_bottom is not None or ylim_top is not None:
        # Criar ticks personalizados que respeitam os limites definidos
        tick_bottom = ylim_bottom if ylim_bottom is not None else current_ylim[0]
        tick_top = ylim_top if ylim_top is not None else current_ylim[1]
        
        # Gerar ticks no intervalo desejado
        n_ticks = 6  # Número de ticks aproximado
        tick_range = tick_top - tick_bottom
        tick_step = tick_range / (n_ticks - 1)
        
        custom_ticks = [tick_bottom + i * tick_step for i in range(n_ticks)]
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels([f'{tick:.0f}' for tick in custom_ticks])
    
    # Personalizar título da legenda e labels, posicionar no canto inferior direito
    if hue_col and ax.get_legend():
        legend = ax.get_legend()
        if legend_title is not None:
            legend.set_title(legend_title)
        else:
            legend.set_title(hue_col.replace('_', ' ').title())
        
        # Personalizar labels da legenda se fornecido
        if legend_labels is not None:
            handles, labels = ax.get_legend_handles_labels()
            new_labels = []
            for label in labels:
                # Converter string para tipo apropriado se necessário
                if label in legend_labels:
                    new_labels.append(legend_labels[label])
                elif label == 'True' and True in legend_labels:
                    new_labels.append(legend_labels[True])
                elif label == 'False' and False in legend_labels:
                    new_labels.append(legend_labels[False])
                else:
                    new_labels.append(label)
            legend = ax.legend(handles, new_labels, title=legend.get_title().get_text())
        
        # Estilizar e posicionar legenda no canto inferior direito
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_fontsize(11)
        legend.set_bbox_to_anchor((0.98, 0.02))  # Canto inferior direito
        legend.set_loc('lower right')
    
    # Configurar gráfico
    ax.set_title(title, fontsize=16, fontweight='bold', pad=30 if subtitle else 20)
    
    # Adicionar subtítulo se fornecido
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, 
                fontsize=12, ha='center', style='italic', color='dimgray')
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='medium')
    
    # Teste bootstrap para cada categoria do eixo x
    test_results = {}
    if hue_col and hue_col in filtered_data.columns and x_col and x_col in filtered_data.columns:
        # Usar ordem especificada ou padrão
        if x_order:
            categories = [cat for cat in x_order if cat in filtered_data[x_col].unique()]
        else:
            categories = filtered_data[x_col].unique()
            
        hue_groups = filtered_data[hue_col].unique()
        
        if len(hue_groups) >= 2:
            for category in categories:
                category_data = filtered_data[filtered_data[x_col] == category]
                
                if len(category_data) > 0:
                    group1_data = category_data[category_data[hue_col] == hue_groups[0]][y_col]
                    group2_data = category_data[category_data[hue_col] == hue_groups[1]][y_col]
                    
                    test_result = bootstrap_test(group1_data, group2_data)
                    test_results[category] = test_result
                else:
                    test_results[category] = {'p_value': np.nan, 'significant': False, 'n1': 0, 'n2': 0, 'real_difference': np.nan}
        else:
            # Se não há grupos suficientes, criar resultado vazio
            for category in categories:
                test_results[category] = {'p_value': np.nan, 'significant': False, 'n1': 0, 'n2': 0, 'real_difference': np.nan}
    else:
        test_results = {'overall': {'p_value': np.nan, 'significant': False, 'n1': 0, 'n2': 0, 'real_difference': np.nan}}
    
    # Adicionar múltiplas caixas de resultado - uma para cada categoria
    add_multiple_result_boxes(ax, test_results)
    
    # Finalizar
    sns.despine(offset=10, trim=True)
    
    # Ajustar layout
    if subtitle:
        plt.subplots_adjust(top=0.85, bottom=0.25, left=0.08, right=0.92)
    else:
        plt.subplots_adjust(bottom=0.25, left=0.08, right=0.92)
    
    plt.tight_layout()
    
    # Salvar figura se solicitado
    if save_fig:
        # Criar diretório se não existir
        os.makedirs(save_dir, exist_ok=True)
        
        # Gerar nome do arquivo se não fornecido
        if filename is None:
            # Limpar título para usar como nome do arquivo
            clean_title = re.sub(r'[^\w\s-]', '', title)  # Remove caracteres especiais
            clean_title = re.sub(r'\s+', '_', clean_title.strip())  # Substitui espaços por _
            filename = clean_title or "violinplot_bootstrap"
        
        # Caminhos dos arquivos
        png_path = os.path.join(save_dir, f"{filename}.png")
        eps_path = os.path.join(save_dir, f"{filename}.eps")
        
        # Salvar PNG (300 DPI)
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        
        # Salvar EPS
        # plt.savefig(eps_path, format='eps', bbox_inches='tight', 
        #             facecolor='white', edgecolor='none')
    
    # IMPRIMIR ESTATÍSTICAS ANTES DE MOSTRAR O GRÁFICO
    # print_statistics(filtered_data, x_col, y_col, hue_col, x_order, test_results)
            
    plt.show()
    
    return test_results

def violinplot_with_overlap(data, filter_conditions=None, x_col=None, y_col=None, hue_col=None, hue2_col=None,
                         title="", subtitle="", xlabel="Categories", ylabel="Distribution", 
                         legend_title=None, hue2_legend_title=None, legend_labels=None, hue2_legend_labels=None,
                         figsize=(12, 7), linewidth=2.5,
                         palette=["m", "g"], hue2_palette=None,
                         x_order=None, hue2_order=None, ylim_bottom=None, ylim_top=None, 
                         save_fig=False, save_dir="./plots", filename=None,
                         hemisphere_mode='split', loc='lower right', legend_spacing=0.15):  # ADICIONAR legend_spacing
    """
    Cria gráfico com linhas horizontais representando as médias dos grupos
    
    Parameters:
    -----------
    legend_labels : dict ou None
        Dicionário para mapear valores da variável hue para labels customizados
    hue2_legend_labels : dict ou None
        Dicionário para mapear valores da variável hue2 para labels customizados
    hemisphere_mode : str
        'split' - Linhas divididas por hue_col (padrão)
    loc : str
        Posição das legendas ('lower right', 'upper right', etc.)
    legend_spacing : float
        Distância horizontal entre as duas legendas (padrão: 0.15)
    """
    
    def get_hue2_palette(n_colors):
        """Gera paleta de cores para hue2"""
        if hue2_palette is not None:
            return hue2_palette
        
        base_palettes = [
            ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'],
            ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C'],
            ['#FF7675', '#74B9FF', '#00B894', '#FDCB6E', '#A29BFE', '#FD79A8'],
            ['#D63031', '#0984E3', '#00B894', '#E17055', '#6C5CE7', '#FD79A8']
        ]
        
        selected_palette = base_palettes[2]
        return selected_palette[:max(n_colors, 6)] 
      
    def draw_mean_lines(ax, filtered_data, x_col, y_col, hue_col, hue2_col, 
                    palette, hue2_colors, x_order, hue2_order, hemisphere_mode, 
                    vertical_gap=0.02):
        """Desenha linhas horizontais das médias com valores em caixas e range vertical"""
        
        # Obter grupos
        if x_order:
            categories = [cat for cat in x_order if cat in filtered_data[x_col].unique()]
        else:
            categories = sorted(filtered_data[x_col].unique())
        
        hue_groups = sorted(filtered_data[hue_col].unique()) if hue_col else [None]
        
        if hue2_col and hue2_order:
            hue2_groups = [cat for cat in hue2_order if cat in filtered_data[hue2_col].unique()]
        elif hue2_col:
            hue2_groups = sorted(filtered_data[hue2_col].unique())
        else:
            hue2_groups = [None]
        
        # Mapear cores do hue principal
        if palette is None or isinstance(palette, str):
            main_colors = ['purple', 'green'][:len(hue_groups)]
        else:
            main_colors = palette[:len(hue_groups)]
        
        hue_color_map = {group: color for group, color in zip(hue_groups, main_colors)}
        
        # Largura da linha horizontal colorida de média
        line_width = 0.15
        
        for i, category in enumerate(categories):
            category_data = filtered_data[filtered_data[x_col] == category]
            x_pos = i
            
            if hemisphere_mode == 'split':
                # Modo split: cada hue_col em um lado, hue2_col sobreposto
                for j, hue_group in enumerate(hue_groups):
                    if hue_group is None:
                        hue_data = category_data
                    else:
                        hue_data = category_data[category_data[hue_col] == hue_group]
                    
                    if len(hue_data) == 0:
                        continue
                    
                    # Para cada hue2 group, desenhar linha de média
                    for k, hue2_group in enumerate(hue2_groups):
                        if hue2_group is None:
                            group_data = hue_data[y_col].dropna()
                            color = hue_color_map.get(hue_group, 'gray')
                        else:
                            group_data = hue_data[hue_data[hue2_col] == hue2_group][y_col].dropna()
                            color = hue2_colors.get(hue2_group, 'gray')
                        
                        if len(group_data) == 0:
                            continue
                        
                        mean_value = np.mean(group_data)
                        min_value = np.min(group_data)
                        max_value = np.max(group_data)
                        
                        if j == 0 or hue_group is None:  # Lado esquerdo
                            # Posição da linha vertical com gap
                            x_vertical = x_pos - vertical_gap
                            linestyle = '-'
                            # Linha horizontal de média (SOMENTE a colorida curta)
                            x_mean_start = x_vertical - line_width
                            x_mean_end = x_vertical
                            text_x = x_mean_start - 0.02
                            ha_alignment = 'right'
                        else:  # Lado direito
                            # Posição da linha vertical com gap
                            x_vertical = x_pos + vertical_gap
                            linestyle = '--'
                            # Linha horizontal de média (SOMENTE a colorida curta)
                            x_mean_start = x_vertical
                            x_mean_end = x_vertical + line_width
                            text_x = x_mean_end + 0.02
                            ha_alignment = 'left'
                        
                        # DESENHAR LINHA VERTICAL (min-max)
                        ax.plot([x_vertical, x_vertical], [min_value, max_value],
                            color='black', linewidth=1.0, linestyle=linestyle,
                            alpha=0.6, zorder=5)
                        
                        # Desenhar SOMENTE linha horizontal colorida curta (SEM extensão pontilhada)
                        ax.plot([x_mean_start, x_mean_end], [mean_value, mean_value], 
                            color=color, linewidth=linewidth*15,  # Linha grossa e colorida
                            alpha=0.9, zorder=10, solid_capstyle='butt')
                        
                        # Adicionar caixa com valor da média
                        ax.text(text_x, mean_value, f'{mean_value:.1f}', 
                            color=color, fontweight='bold', fontsize=9,
                            ha=ha_alignment, va='center', 
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                                    edgecolor=color, alpha=1),
                            zorder=15)
            
            elif hemisphere_mode == 'unified':
                # Modo unificado: linhas completas
                all_category_data = category_data[y_col].dropna()
                
                if len(all_category_data) == 0:
                    continue
                
                for k, hue2_group in enumerate(hue2_groups):
                    if hue2_group is None:
                        group_data = all_category_data
                        color = main_colors[0] if main_colors else 'gray'
                    else:
                        group_data = category_data[category_data[hue2_col] == hue2_group][y_col].dropna()
                        color = hue2_colors.get(hue2_group, 'gray')
                    
                    if len(group_data) == 0:
                        continue
                    
                    mean_value = np.mean(group_data)
                    min_value = np.min(group_data)
                    max_value = np.max(group_data)
                    
                    # Centro para linha vertical
                    x_vertical = x_pos
                    
                    # DESENHAR LINHA VERTICAL (min-max)
                    ax.plot([x_vertical, x_vertical], [min_value, max_value],
                        color='black', linewidth=1.0, alpha=0.6, zorder=5)
                    
                    # Linha horizontal de média (SOMENTE colorida curta)
                    x_mean_start = x_vertical - line_width
                    x_mean_end = x_vertical + line_width
                    
                    ax.plot([x_mean_start, x_mean_end], [mean_value, mean_value], 
                        color=color, linewidth=linewidth*15, alpha=0.9, zorder=10,
                        solid_capstyle='butt')
                    
                    # Adicionar caixa com valor da média (à direita)
                    text_x = x_mean_end + 0.02
                    ax.text(text_x, mean_value, f'{mean_value:.1f}', 
                        color=color, fontweight='bold', fontsize=9,
                        ha='left', va='center', 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                                edgecolor=color, alpha=1),
                        zorder=15)

    def apply_legend_labels(original_labels, legend_labels_dict):
        """Aplica labels customizados aos labels originais"""
        if legend_labels_dict is None:
            return original_labels
        
        new_labels = []
        for label in original_labels:
            if label in legend_labels_dict:
                new_labels.append(legend_labels_dict[label])
            elif str(label) == 'True' and True in legend_labels_dict:
                new_labels.append(legend_labels_dict[True])
            elif str(label) == 'False' and False in legend_labels_dict:
                new_labels.append(legend_labels_dict[False])
            else:
                new_labels.append(str(label))
        return new_labels
    
    # Configurar tema
    sns.set_theme(style="ticks", palette="pastel")
    
    # Filtrar dados
    filtered_data = data.copy()
    if filter_conditions:
        for col, value in filter_conditions.items():
            filtered_data = filtered_data[filtered_data[col] == value]
    
    # Verificar se hue2_col existe
    if hue2_col and hue2_col not in filtered_data.columns:
        print(f"Aviso: Coluna '{hue2_col}' não encontrada.")
        hue2_col = None
    
    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Configurar paleta para hue2
    hue2_colors = {}
    if hue2_col:
        if hue2_order:
            hue2_groups = [cat for cat in hue2_order if cat in filtered_data[hue2_col].unique()]
        else:
            hue2_groups = sorted(filtered_data[hue2_col].unique())
        
        hue2_colors_list = get_hue2_palette(len(hue2_groups))
        hue2_colors = {group: color for group, color in zip(hue2_groups, hue2_colors_list)}
    
    # Desenhar linhas de média
    draw_mean_lines(ax, filtered_data, x_col, y_col, hue_col, hue2_col,
                   palette, hue2_colors, x_order, hue2_order, hemisphere_mode)
    
    # Configurar eixo X
    if x_order:
        categories = [cat for cat in x_order if cat in filtered_data[x_col].unique()]
    else:
        categories = sorted(filtered_data[x_col].unique())
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    
    # Criar legendas lado a lado
    if hue_col:
        hue_groups = sorted(filtered_data[hue_col].unique())
        from matplotlib.lines import Line2D
        
        custom_hue_labels = apply_legend_labels(hue_groups, legend_labels)
        
        legend_elements = []
        for i, (group, custom_label) in enumerate(zip(hue_groups, custom_hue_labels)):
            linestyle = '-' if i == 0 else '--'
            legend_elements.append(
                Line2D([0], [0], color='black', linewidth=2.5, linestyle=linestyle, 
                    label=custom_label)
            )
        
        # Mapear loc para bbox_to_anchor base
        loc_map = {
            'upper right': (1.0, 1.0),
            'upper left': (0.0, 1.0),
            'lower right': (1.0, 0.0),
            'lower left': (0.0, 0.0),
            'center right': (1.0, 0.5),
            'center left': (0.0, 0.5),
            'upper center': (0.5, 1.0),
            'lower center': (0.5, 0.0),
            'center': (0.5, 0.5)
        }
        
        # Obter posição base do loc
        base_x, base_y = loc_map.get(loc, (1.0, 0.0))
        
        # Determinar se as legendas devem ir para esquerda ou direita
        if 'right' in loc:
            # Legendas à direita - primeira mais à direita, segunda à esquerda
            bbox1 = (base_x, base_y)
            bbox2 = (base_x - legend_spacing, base_y)
        elif 'left' in loc:
            # Legendas à esquerda - primeira mais à esquerda, segunda à direita
            bbox1 = (base_x, base_y)
            bbox2 = (base_x + legend_spacing, base_y)
        else:
            # Centro - distribuir simetricamente
            bbox1 = (base_x - legend_spacing/2, base_y)
            bbox2 = (base_x + legend_spacing/2, base_y)
        
        # Criar primeira legenda (hue_col)
        legend1 = ax.legend(handles=legend_elements, 
                        title=legend_title or hue_col.replace('_', ' ').title(),
                        loc=loc,
                        bbox_to_anchor=bbox1,
                        frameon=True, facecolor='white', edgecolor='black',
                        fontsize=10)
        legend1.get_title().set_fontweight('bold')
        legend1.get_title().set_fontsize(11)
        ax.add_artist(legend1)  # IMPORTANTE: adicionar explicitamente

        # Legenda de simulação (hue2_col)
        if hue2_col and hue2_colors:
            hue2_groups_list = list(hue2_colors.keys())
            custom_hue2_labels = apply_legend_labels(hue2_groups_list, hue2_legend_labels)
            
            hue2_patches = [Patch(color=color, alpha=0.9, label=custom_label) 
                        for (group, color), custom_label in zip(hue2_colors.items(), custom_hue2_labels)]
            
            legend2 = ax.legend(handles=hue2_patches,
                            title=hue2_legend_title or hue2_col.replace('_', ' ').title(),
                            loc=loc,
                            bbox_to_anchor=bbox2,
                            frameon=True, facecolor='white', edgecolor='black',
                            fontsize=10)
            legend2.get_title().set_fontweight('bold')
            legend2.get_title().set_fontsize(11)
    
    # # Criar legendas
    # if hue_col:
    #     hue_groups = sorted(filtered_data[hue_col].unique())
    #     from matplotlib.lines import Line2D
        
    #     custom_hue_labels = apply_legend_labels(hue_groups, legend_labels)
        
    #     legend_elements = []
    #     for i, (group, custom_label) in enumerate(zip(hue_groups, custom_hue_labels)):
    #         linestyle = '-' if i == 0 else '--'
    #         legend_elements.append(
    #             Line2D([0], [0], color='black', linewidth=2.5, linestyle=linestyle, 
    #                    label=custom_label)
    #         )
        
    #     legend1 = ax.legend(handles=legend_elements, 
    #                       title=legend_title or hue_col.replace('_', ' ').title(),
    #                       loc=loc, bbox_to_anchor=(0.98, 0.02),
    #                       frameon=True, facecolor='white', edgecolor='black',
    #                       fontsize=10)
    #     legend1.get_title().set_fontweight('bold')
    #     legend1.get_title().set_fontsize(11)
    #     plt.gca().add_artist(legend1)

    # # Legenda de simulação
    # if hue2_col and hue2_colors:
    #     hue2_groups_list = list(hue2_colors.keys())
    #     custom_hue2_labels = apply_legend_labels(hue2_groups_list, hue2_legend_labels)
        
    #     hue2_patches = [Patch(color=color, alpha=0.9, label=custom_label) 
    #                    for (group, color), custom_label in zip(hue2_colors.items(), custom_hue2_labels)]
        
    #     legend2_y = 0.25
    #     legend2 = ax.legend(handles=hue2_patches,
    #                       title=hue2_legend_title or hue2_col.replace('_', ' ').title(),
    #                       loc=loc, bbox_to_anchor=(0.98, legend2_y),
    #                       frameon=True, facecolor='white', edgecolor='black')
    #     legend2.get_title().set_fontweight('bold')
    
    # Configurar limites do eixo Y
    if ylim_bottom is not None or ylim_top is not None:
        current_ylim = ax.get_ylim()
        data_min = filtered_data[y_col].min()
        data_max = filtered_data[y_col].max()
        
        new_ylim_bottom = ylim_bottom if ylim_bottom is not None else current_ylim[0]
        new_ylim_top = ylim_top if ylim_top is not None else current_ylim[1]
        
        ax.set_ylim(new_ylim_bottom, new_ylim_top)
    
    # Configurar gráfico
    ax.set_title(title, fontsize=16, fontweight='bold', pad=30 if subtitle else 20)
    
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, 
                fontsize=12, ha='center', style='italic', color='dimgray')
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='medium')
    
    # Grade horizontal leve
    ax.yaxis.grid(True, linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Finalizar
    sns.despine(offset=10, trim=True)
    
    plt.tight_layout()
    
    # Salvar se solicitado
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        
        if filename is None:
            clean_title = re.sub(r'[^\w\s-]', '', title)
            clean_title = re.sub(r'\s+', '_', clean_title.strip())
            filename = clean_title or "mean_lines_plot"
        
        png_path = os.path.join(save_dir, f"{filename}.png")
        
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
    
    plt.show()
    
    # Retornar informações
    groups_info = {
        'x_categories': categories,
        'hue_groups': sorted(filtered_data[hue_col].unique()) if hue_col else [],
        'hue2_groups': list(hue2_colors.keys()) if hue2_colors else [],
        'hue2_colors': hue2_colors,
        'hemisphere_mode': hemisphere_mode
    }
    
    return groups_info