from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os
from insights import resumo_executivo

def gerar_relatorio_completo():
    try:
        caminho_pdf = '/tmp/Agentes_Autonomos_Relatorio_Atividade_Extra.pdf'
        doc = SimpleDocTemplate(caminho_pdf, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph('<b>Relatório Completo - Análise EDA</b>', styles['Title']))
        story.append(Spacer(1, 12))

        imagens = [
            '/tmp/hist_idade.png',
            '/tmp/mapa_correlacao.png',
            '/tmp/kmeans_clusters.png',
            '/tmp/elbow_plot.png',
            '/tmp/boxplot_salario.png',
            '/tmp/dispersao_idade_salario.png'
        ]

        for img in imagens:
            if os.path.exists(img):
                story.append(Image(img, width=400, height=300))
                story.append(Spacer(1, 12))

        story.append(Spacer(1, 24))
        story.append(Paragraph(resumo_executivo(), styles['Normal']))

        doc.build(story)
        return f'✅ Relatório completo gerado em {caminho_pdf}'
    except Exception as e:
        return f'❌ Erro ao gerar relatório: {e}'
