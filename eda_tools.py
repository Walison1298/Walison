import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import os

def _ensure_dataset():
    if not os.path.exists('/tmp/dataset.csv'):
        data = {
            'idade':[25,30,40,35,45,28,38,50],
            'salario':[3500,5000,8000,6000,9000,4200,7200,9500],
            'tempo_empresa':[1,3,10,5,12,2,8,15]
        }
        pd.DataFrame(data).to_csv('/tmp/dataset.csv', index=False)

def carregar_csv(path: str = None) -> str:
    """Carrega dataset interno ou de path opcional.""
    try:
        if path:
            df = pd.read_csv(path)
            df.to_csv('/tmp/dataset.csv', index=False)
            return '✅ Dataset carregado de: ' + path
        else:
            _ensure_dataset()
            return '✅ Dataset interno carregado.'
    except Exception as e:
        return f'❌ Erro ao carregar CSV: {e}'

def estatisticas_descritivas() -> str:
    try:
        df = pd.read_csv('/tmp/dataset.csv')
        stats = df.select_dtypes(include='number').describe().to_string()
        return '📊 Estatísticas descritivas:\n' + stats
    except Exception as e:
        return f'❌ Erro ao calcular estatísticas: {e}'

def valores_nulos() -> str:
    try:
        df = pd.read_csv('/tmp/dataset.csv')
        nulos = df.isnull().sum().to_string()
        return '🧩 Valores nulos por coluna:\n' + nulos
    except Exception as e:
        return f'❌ Erro ao verificar nulos: {e}'

def gerar_histograma(coluna: str) -> str:
    try:
        df = pd.read_csv('/tmp/dataset.csv')
        plt.figure()
        df[coluna].dropna().hist(bins=8)
        caminho = f'/tmp/hist_{coluna}.png'
        plt.savefig(caminho)
        plt.close()
        return f'✅ Histograma gerado: {caminho}'
    except Exception as e:
        return f'❌ Erro ao gerar histograma: {e}'

def gerar_mapa_correlacao() -> str:
    try:
        df = pd.read_csv('/tmp/dataset.csv')
        corr = df.corr()
        plt.figure()
        sns.heatmap(corr, annot=True)
        caminho = '/tmp/mapa_correlacao.png'
        plt.savefig(caminho)
        plt.close()
        return '✅ Mapa de correlação gerado: ' + caminho
    except Exception as e:
        return f'❌ Erro ao gerar mapa de correlação: {e}'

def executar_kmeans(k: int = 3) -> str:
    try:
        df = pd.read_csv('/tmp/dataset.csv')
        X = df.select_dtypes(include='number').dropna()
        kmeans = KMeans(n_clusters=int(k), random_state=42)
        labels = kmeans.fit_predict(X)
        plt.figure()
        plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels)
        caminho = '/tmp/kmeans_clusters.png'
        plt.savefig(caminho)
        plt.close()
        return '✅ K-Means executado. Gráfico salvo em ' + caminho
    except Exception as e:
        return f'❌ Erro ao executar K-Means: {e}'

def metodo_elbow(k_min: int =1, k_max:int =10) -> str:
    try:
        df = pd.read_csv('/tmp/dataset.csv')
        X = df.select_dtypes(include='number').dropna()
        wcss=[]
        for k in range(int(k_min), int(k_max)+1):
            kmeans=KMeans(n_clusters=k, random_state=42).fit(X)
            wcss.append(kmeans.inertia_)
        plt.figure()
        plt.plot(range(int(k_min), int(k_max)+1), wcss, marker='o')
        caminho='/tmp/elbow_plot.png'
        plt.savefig(caminho)
        plt.close()
        return '✅ Método Elbow gerado: ' + caminho
    except Exception as e:
        return f'❌ Erro no método Elbow: {e}'

def gerar_boxplot(coluna: str) -> str:
    try:
        df = pd.read_csv('/tmp/dataset.csv')
        plt.figure()
        sns.boxplot(y=df[coluna].dropna())
        caminho=f'/tmp/boxplot_{coluna}.png'
        plt.savefig(caminho)
        plt.close()
        return '✅ Boxplot gerado: ' + caminho
    except Exception as e:
        return f'❌ Erro ao gerar boxplot: {e}'

def gerar_dispersao(x_col: str, y_col: str) -> str:
    try:
        df = pd.read_csv('/tmp/dataset.csv')
        plt.figure()
        plt.scatter(df[x_col], df[y_col])
        caminho=f'/tmp/dispersao_{x_col}_{y_col}.png'
        plt.savefig(caminho)
        plt.close()
        return '✅ Dispersão gerada: ' + caminho
    except Exception as e:
        return f'❌ Erro ao gerar dispersão: {e}'

def detectar_outliers() -> str:
    try:
        df = pd.read_csv('/tmp/dataset.csv')
        info = ''
        for col in df.select_dtypes(include='number').columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            low = Q1 - 1.5 * IQR
            high = Q3 + 1.5 * IQR
            out = df[(df[col] < low) | (df[col] > high)]
            info += f"Col '{col}': {len(out)} outliers\n"
        return '📊 Outliers detectados:\n' + info
    except Exception as e:
        return f'❌ Erro ao detectar outliers: {e}'
