from flask import Flask, render_template, g
import sqlite3

app = Flask(__name__)

# Configuração do Banco
DATABASE = 'vigia.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row # Para acessar colunas pelo nome
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# Rota Principal (Dashboard)
@app.route('/')
def index():
    cur = get_db().cursor()
    
    # Busca quem está presente AGORA
    cur.execute("SELECT * FROM acessos WHERE status = 'Presente' ORDER BY id DESC")
    presentes = cur.fetchall()
    
    # Busca histórico dos últimos 10 que saíram
    cur.execute("SELECT * FROM acessos WHERE status = 'Finalizado' ORDER BY id DESC LIMIT 10")
    historico = cur.fetchall()
    
    return render_template('index.html', presentes=presentes, historico=historico)

if __name__ == '__main__':
    # Roda o site na rede local (acessível pelo celular)
    # Porta 5000 é padrão
    app.run(host='0.0.0.0', port=5000, debug=True)