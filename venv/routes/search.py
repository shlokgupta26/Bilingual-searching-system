from flask import Blueprint, render_template, request
from retrieval import query_retrieval
import translation 
search = Blueprint('search', __name__, template_folder='templates')

@search.route('/')
def home():
    return render_template('home.html')

@search.route('/fetch')
def fetch():
    if 'query1' and 'num_res' in request.args:
        query1 = request.args.get('query1')
        num = request.args.get('num_res')
        num = num.strip('"')
        num = int(num)
        
        trans_results1 = translation.de_eng_direct(query1)
        trans_results2 = translation.de_eng_noisy_translate(query1)
        trans_results3 = translation.de_eng_noisy_translate2(query1)

        results1 = query_retrieval(trans_results1,num)
        results2 = query_retrieval(trans_results2,num)
        results3 = query_retrieval(trans_results3,num)

    return render_template('home.html', query1 = query1, trans_results1 = trans_results1, trans_results2 = trans_results2, trans_results3 = trans_results3, results1 = results1, results2 = results2, results3 = results3)

@search.route('/home_translate')
def home_translate():
    return render_template('translate.html')

@search.route('/translate_results')
def translate_resuults():
    if 'query2' in request.args:
        query2 = request.args.get('query2')
        trans_results1 = translation.de_eng_direct(query2)
        trans_results2 = translation.de_eng_noisy_translate(query2)
        trans_results3 = translation.de_eng_noisy_translate2(query2)
    return render_template('translate.html', query2 = query2 , trans_results1 = trans_results1, trans_results2 = trans_results2, trans_results3 = trans_results3)

@search.route('/home_search')
def home_search():
    return render_template('search.html')

@search.route('/search_results')
def search_results():
    if 'search1' and 'search2' in request.args:
        query = request.args.get('search1')
        num = request.args.get('search2')
        num = num.strip('"')
        num = int(num)
        results = query_retrieval(query,num)
        
    return render_template('search.html', results = results)
    