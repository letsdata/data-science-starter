# Lembra do arquivo que criamos? Vamos importá-lo aqui
import pickle
import pandas as pd

# Flask é um microframework para criar aplicações web
from flask import Flask, render_template, request

# Instanciando a aplicação com a pasta de templates
app = Flask(__name__, template_folder='template', static_folder='template/assets')

# Treina lá, usa cá
# Aqui vamos carregar o modelo que treinamos
modelo_precos = pickle.load(open('./models/modelo_previsao_carros.pickle', 'rb'))
carros_df = pickle.load(open('./models/carros_df.pickle', 'rb'))
scaler = pickle.load(open('./models/scaler.pickle', 'rb'))

# Endpoint principal (home)
@app.route('/')
def home():
    return render_template("homepage.html")


# Endpoint para o formulário
@app.route('/carros')
def carros():
    return render_template("form.html")


# Endpoint que executa a previsão do preço do carro e retorna o resultado
@app.route('/preco_carro', methods=['POST'])
def previsao_preco():
    taxa = request.form.get('Taxa')
    marca = request.form.get('Marca')
    modelo = request.form.get('Modelo')
    ano_fabricacao = request.form.get('AnoFabricacao')
    categoria = request.form.get('Categoria')
    bancos_couro = request.form.get('BancosDeCouro')
    combustivel = request.form.get('Combustivel')
    volume_motor = request.form.get('VolumeMotor')
    quilometragem = request.form.get('Quilometragem')
    cilindradas = request.form.get('Cilindradas')
    cambio = request.form.get('Cambio')
    tracao = request.form.get('Tracao')
    portas = request.form.get('Portas')
    direcao = request.form.get('Direcao')
    cor = request.form.get('Cor')
    airbags = request.form.get('Airbags')

    # Criação do DataFrame com os valores recebidos
    data = {'Taxa': [taxa],
            'Marca': [marca],
            'Modelo': [modelo],
            'Ano Fabricacao': [ano_fabricacao],
            'Categoria': [categoria],
            'Bancos de Couro': [bancos_couro],
            'Combustivel': [combustivel],
            'Volume Motor': [volume_motor],
            'Quilometragem': [quilometragem],
            'Cilindradas': [cilindradas],
            'Cambio': [cambio],
            'Tracao': [tracao],
            'Portas': [portas],
            'Direcao': [direcao],
            'Cor': [cor],
            'Airbags': [airbags]}

    df = pd.DataFrame(data)

    df = scaler.transform(df)


    # Executa a classificação
    preco = modelo_precos.predict(df)[0]

    return render_template('result.html', preco=preco)


# Roda a aplicação
if __name__ == "__main__":
    app.run(debug=True)
