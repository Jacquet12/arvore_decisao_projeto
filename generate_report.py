from fpdf import FPDF
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
import os

# Dados iniciais
data = {
    "Clima": ["Chuvoso", "Ensolarado", "Nublado", "Nublado", "Ensolarado", "Nublado", "Chuvoso", "Chuvoso", "Ensolarado", "Nublado"],
    "Temperatura": ["Frio", "Agradável", "Calor", "Agradável", "Calor", "Frio", "Calor", "Agradável", "Agradável", "Frio"],
    "Humidade": ["Alta", "Baixa", "Normal", "Normal", "Normal", "Normal", "Alta", "Alta", "Baixa", "Baixa"],
    "Vento": ["Intenso", "Fraco", "Fraco", "Fraco", "Fraco", "Intenso", "Intenso", "Intenso", "Intenso", "Fraco"],
    "Futebol": ["Não", "Sim", "Sim", "Sim", "Sim", "Não", "Sim", "Não", "Sim", "Não"]
}

# Preparação dos dados
df = pd.DataFrame(data)
df_encoded = df.apply(lambda x: pd.factorize(x)[0])
X = df_encoded.drop("Futebol", axis=1)
y = df_encoded["Futebol"]

# Treinando a árvore de decisão
clf = DecisionTreeClassifier(criterion="gini")
clf = clf.fit(X, y)

# Caminho para salvar a imagem da árvore
os.makedirs("output", exist_ok=True)
tree_image_path = "output/arvore_decisao.png"

# Plotando a árvore de decisão e salvando
fig, ax = plt.subplots(figsize=(10, 6))
plot_tree(clf, feature_names=list(X.columns), class_names=["Não", "Sim"], filled=True, rounded=True, fontsize=10)
plt.title("Estrutura da Árvore de Decisão - Futebol")
plt.savefig(tree_image_path, format="png")
plt.close(fig)

# Função para gerar o PDF
def generate_pdf():
    pdf = FPDF()
    pdf.add_page()

    # Cabeçalho
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 8, "Universidade Federal da Fronteira Sul", ln=True, align="C")
    pdf.cell(200, 8, "Estudante: Jacquet Leme", ln=True, align="C")
    pdf.cell(200, 8, "Disciplina: Inteligência Artificial", ln=True, align="C")
    pdf.cell(200, 8, "Trabalho: Árvores de Decisão", ln=True, align="C")

    # Título do relatório
    pdf.set_font("Arial", "B", 16)
    pdf.ln(8)
    pdf.cell(200, 10, "Construção e Cálculo da Árvore de Decisão", ln=True, align="C")

    # Dados Iniciais
    pdf.set_font("Arial", "B", 12)
    pdf.ln(10)
    pdf.cell(200, 8, "Passo 1: Tabela de Dados Iniciais", ln=True)

    data_initial = [
        ["Clima", "Temperatura", "Humidade", "Vento", "Futebol"],
        ["Chuvoso", "Frio", "Alta", "Intenso", "Não"],
        ["Ensolarado", "Agradável", "Baixa", "Fraco", "Sim"],
        ["Nublado", "Calor", "Normal", "Fraco", "Sim"],
        ["Nublado", "Agradável", "Normal", "Fraco", "Sim"],
        ["Ensolarado", "Calor", "Normal", "Fraco", "Sim"],
        ["Nublado", "Frio", "Normal", "Intenso", "Não"],
        ["Chuvoso", "Calor", "Alta", "Intenso", "Sim"],
        ["Chuvoso", "Agradável", "Alta", "Intenso", "Não"],
        ["Ensolarado", "Agradável", "Baixa", "Intenso", "Sim"],
        ["Nublado", "Frio", "Baixa", "Fraco", "Não"]
    ]

    pdf.set_font("Arial", size=10)
    col_width = pdf.w / 5.5
    row_height = pdf.font_size * 1.25

    for row in data_initial:
        for item in row:
            pdf.cell(col_width, row_height, txt=item, border=1)
        pdf.ln(row_height)

    # Cálculo do índice de Gini
    pdf.set_font("Arial", "B", 12)
    pdf.ln(8)
    pdf.cell(200, 8, "Passo 2: Cálculo do Índice de Gini Inicial", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.ln(5)
    pdf.multi_cell(0, 8, "O índice de Gini é uma medida de pureza. Para o conjunto completo:\n"
                         "Gini(S) = 1 - (p(Sim)^2 + p(Não)^2) = 0.48\n")

    # Representação da Árvore
    pdf.set_font("Arial", "B", 12)
    pdf.ln(8)
    pdf.cell(200, 8, "Passo 4: Estrutura da Árvore de Decisão", ln=True)
    pdf.image(tree_image_path, x=20, w=170)

    # Salvando o PDF final
    pdf_output_path = "output/Trabalho_Arvore_Decisao.pdf"
    pdf.output(pdf_output_path)
    print(f"PDF gerado com sucesso em: {pdf_output_path}")

# Executa a função de geração do PDF
if __name__ == "__main__":
    generate_pdf()
