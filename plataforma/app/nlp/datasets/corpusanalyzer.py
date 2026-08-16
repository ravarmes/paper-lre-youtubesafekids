import pandas as pd
import numpy as np
from collections import Counter

# Importações opcionais para visualização
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("[AVISO] Matplotlib e/ou Seaborn não estão instalados. Visualizações serão desabilitadas.")

class CorpusAnalyzer:
    def __init__(self, csv_path):
        """
        Inicializa o analisador com o arquivo CSV
        """
        try:
            # Tenta ler o CSV com diferentes separadores
            try:
                self.df = pd.read_csv(csv_path, encoding='utf-8', sep=';')
            except:
                self.df = pd.read_csv(csv_path, encoding='utf-8', sep=',')
            
            print(f"[INFO] Arquivo carregado com sucesso: {len(self.df)} registros")
            print(f"[INFO] Colunas encontradas: {list(self.df.columns)}")
            
            self.clean_data()
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
        except Exception as e:
            raise Exception(f"Erro ao carregar o arquivo CSV: {str(e)}")
        
    def clean_data(self):
        """
        Limpa e padroniza os dados
        """
        # Remove espaços extras
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].astype(str).str.strip()
        
        # Padroniza valores da coluna CATEGORIA
        if 'CATEGORIA' in self.df.columns:
            self.df['CATEGORIA'] = self.df['CATEGORIA'].str.title()
            
    def relatorio_geral(self):
        """
        Gera relatório geral do corpus
        """
        print("="*60)
        print("RELATÓRIO GERAL DO CORPUS")
        print("="*60)
        
        # Informações básicas
        print(f"[INFO] Total de registros: {len(self.df)}")
        print(f"[INFO] Quantidade de filmes diferentes: {self.df['ID'].nunique()}")
        print(f"[INFO] Quantidade total de frases: {len(self.df)}")
        
        # Lista dos filmes
        print(f"\n[FILMES] Filmes no corpus:")
        filmes = self.df['ID'].unique()
        for filme in filmes:
            count = len(self.df[self.df['ID'] == filme])
            print(f"   -> {filme}: {count} frases")
            
        print("\n" + "="*60)
        
    def relatorio_por_parte(self):
        """
        Relatório de distribuição por parte do vídeo
        """
        print("[PARTE] DISTRIBUIÇÃO POR PARTE DO VÍDEO")
        print("-"*40)
        
        if 'PARTE' in self.df.columns:
            parte_counts = self.df['PARTE'].value_counts()
            for parte, count in parte_counts.items():
                porcentagem = (count / len(self.df)) * 100
                print(f"   {parte}: {count} frases ({porcentagem:.1f}%)")
        else:
            print("   [AVISO] Coluna 'PARTE' não encontrada")
            
        print()
        
    def relatorio_por_categoria(self):
        """
        Relatório de distribuição por categoria
        """
        print("[CATEGORIA] DISTRIBUIÇÃO POR CATEGORIA")
        print("-"*40)
        
        # Tenta diferentes nomes de coluna para categoria
        categoria_col = None
        for col in ['CATEGORIAS', 'CATEGORI', 'CATEGORIA']:
            if col in self.df.columns:
                categoria_col = col
                break
                
        if categoria_col:
            cat_counts = self.df[categoria_col].value_counts()
            for categoria, count in cat_counts.items():
                porcentagem = (count / len(self.df)) * 100
                print(f"   {categoria}: {count} frases ({porcentagem:.1f}%)")
        else:
            print("   [AVISO] Coluna de categoria não encontrada")
            
        print()
        
    def relatorio_toxicidade(self):
        """
        Relatório de distribuição por toxicidade
        """
        print("[TOXICIDADE] DISTRIBUIÇÃO POR TOXICIDADE")
        print("-"*40)
        
        if 'TOX' in self.df.columns:
            tox_counts = self.df['TOX'].value_counts()
            for tox, count in tox_counts.items():
                porcentagem = (count / len(self.df)) * 100
                print(f"   {tox}: {count} frases ({porcentagem:.1f}%)")
        else:
            print("   [AVISO] Coluna 'TOX' não encontrada")
            
        print()
        
    def relatorio_cruzado(self, coluna1, coluna2):
        """
        Relatório cruzado entre duas variáveis
        """
        if coluna1 in self.df.columns and coluna2 in self.df.columns:
            print(f"🔄 RELATÓRIO CRUZADO: {coluna1} x {coluna2}")
            print("-"*50)
            
            crosstab = pd.crosstab(self.df[coluna1], self.df[coluna2], margins=True)
            print(crosstab)
            print()
        else:
            print(f"❌ Uma ou ambas as colunas ({coluna1}, {coluna2}) não foram encontradas")
            
    def filtrar_e_contar(self, filtros):
        """
        Aplica filtros e conta os resultados
        Exemplo: {'CATEGORI': 'Negativo', 'PARTE': 'Início'}
        """
        df_filtrado = self.df.copy()
        
        print("🔍 RELATÓRIO COM FILTROS")
        print("-"*30)
        print("Filtros aplicados:")
        
        for coluna, valor in filtros.items():
            if coluna in df_filtrado.columns:
                df_filtrado = df_filtrado[df_filtrado[coluna] == valor]
                print(f"   • {coluna} = {valor}")
            else:
                print(f"   ❌ Coluna '{coluna}' não encontrada")
                
        resultado = len(df_filtrado)
        print(f"\n📊 Resultado: {resultado} frases encontradas")
        
        if resultado > 0:
            print("\nPrimeiras 5 frases encontradas:")
            for i, row in df_filtrado.head().iterrows():
                if 'FRASE' in row:
                    print(f"   {i+1}. {row['FRASE'][:50]}...")
                    
        print()
        return df_filtrado
        
    def gerar_visualizacoes(self):
        """
        Gera gráficos para visualizar os dados
        """
        if not VISUALIZATION_AVAILABLE:
            print("[AVISO] Visualizações não disponíveis. Instale matplotlib e seaborn para habilitar.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análise do Corpus - Classificação de Frases', fontsize=16)
        
        # Gráfico 1: Distribuição por filme
        if len(self.df['ID'].unique()) <= 10:  # Só mostra se não tiver muitos filmes
            self.df['ID'].value_counts().plot(kind='bar', ax=axes[0,0])
            axes[0,0].set_title('Frases por Filme')
            axes[0,0].set_xlabel('ID do Filme')
            axes[0,0].set_ylabel('Quantidade de Frases')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # Gráfico 2: Distribuição por parte
        if 'PARTE' in self.df.columns:
            self.df['PARTE'].value_counts().plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
            axes[0,1].set_title('Distribuição por Parte do Vídeo')
            axes[0,1].set_ylabel('')
            
        # Gráfico 3: Distribuição por categoria
        categoria_col = None
        for col in ['CATEGORIAS', 'CATEGORI', 'CATEGORIA']:
            if col in self.df.columns:
                categoria_col = col
                break
                
        if categoria_col:
            self.df[categoria_col].value_counts().plot(kind='bar', ax=axes[1,0])
            axes[1,0].set_title('Distribuição por Categoria')
            axes[1,0].set_xlabel('Categoria')
            axes[1,0].set_ylabel('Quantidade de Frases')
            axes[1,0].tick_params(axis='x', rotation=45)
            
        # Gráfico 4: Distribuição por toxicidade
        if 'TOX' in self.df.columns:
            self.df['TOX'].value_counts().plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title('Distribuição por Toxicidade')
            axes[1,1].set_xlabel('Nível de Toxicidade')
            axes[1,1].set_ylabel('Quantidade de Frases')
            axes[1,1].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.show()
        
    def exportar_relatorio(self, nome_arquivo='relatorio_corpus.txt'):
        """
        Exporta todos os relatórios para um arquivo de texto
        """
        import sys
        from io import StringIO
        
        # Captura a saída
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Gera todos os relatórios
        self.relatorio_geral()
        self.relatorio_por_parte()
        self.relatorio_por_categoria()
        self.relatorio_toxicidade()
        
        # Restaura a saída normal
        sys.stdout = old_stdout
        
        # Salva no arquivo
        with open(nome_arquivo, 'w', encoding='utf-8') as f:
            f.write(captured_output.getvalue())
            
        print(f"📄 Relatório salvo em: {nome_arquivo}")

def main():
    """
    Exemplo de uso da classe
    """
    import os
    import sys

    # O corpus e unico no repositorio (src/dados); config.PATHS e quem sabe onde ele esta
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    sys.path.insert(0, project_root)
    from app.nlp.config import PATHS

    csv_path = PATHS['corpus_file']

    # Verifica se o arquivo existe
    if not os.path.exists(csv_path):
        print(f"[ERRO] Corpus não encontrado em: {csv_path}")
        print(f"[INFO] Raiz do projeto: {project_root}")

        # Tenta encontrar o arquivo em outras localizações possíveis
        alternative_paths = [
            os.path.join(project_root, 'data', 'corpus.csv'),
            os.path.join(project_root, 'corpus.csv'),
            'corpus.csv'
        ]

        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                csv_path = alt_path
                print(f"[INFO] Arquivo encontrado em: {csv_path}")
                break
        else:
            csv_path = input("Digite o caminho completo do arquivo CSV: ")
    
    try:
        # Inicializa o analisador
        analyzer = CorpusAnalyzer(csv_path)
        
        # Gera relatórios
        analyzer.relatorio_geral()
        analyzer.relatorio_por_parte()
        analyzer.relatorio_por_categoria()
        analyzer.relatorio_toxicidade()
        
        # Relatório cruzado - exemplo
        analyzer.relatorio_cruzado('CATEGORIA', 'PARTE')
        
        # Filtros personalizados - exemplos
        print("EXEMPLOS DE FILTROS PERSONALIZADOS:")
        print("="*50)
        
        # Frases negativas do início
        analyzer.filtrar_e_contar({'CATEGORIA': 'Negativo', 'PARTE': 'Início'})
        
        # Frases não tóxicas do meio
        analyzer.filtrar_e_contar({'TOX': 'Não tóxico', 'PARTE': 'Meio'})
        
        # Gera visualizações
        try:
            analyzer.gerar_visualizacoes()
        except Exception as viz_error:
            print(f"Aviso: Não foi possível gerar visualizações: {viz_error}")
        
        # Exporta relatório
        analyzer.exportar_relatorio()
        
    except FileNotFoundError:
        print(f"[ERRO] Arquivo '{csv_path}' não encontrado!")
        print("[INFO] Certifique-se de que o arquivo existe e o caminho está correto.")
        print(f"[INFO] Pasta atual: {os.getcwd()}")
    except Exception as e:
        print(f"[ERRO] Erro ao processar o arquivo: {str(e)}")

if __name__ == "__main__":
    main()