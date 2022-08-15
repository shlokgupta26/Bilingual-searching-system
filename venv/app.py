from flask import Flask
from routes.search import search

app = Flask(__name__)

app.register_blueprint(search)
app.secret_key = 'secret1234'

if __name__ == '__main__':    
    app.run()




