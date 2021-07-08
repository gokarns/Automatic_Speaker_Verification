from flask import Flask, request, jsonify


app = Flask(__name__)

books_list = [
    {
        "id": 0,
        "author": "Chinua Achebe",
        "language": "English", 
        "title": "Things Fall Apart",
        },
    {
        "id": 1,
        "author": "Hans Christian Andersen",
        "language": "Danish",
        "title": "Fairy Tales",
        }]

@app.route('/books', methods=['GET', 'POST'])
def books():
    if request.method == 'GET':
        if len(books_list) > 0:
            return jsonify(books_list)
        else:
            'Nothing Found', 404

    if request.method == 'POST':
        new_author = request.form['author']
        new_lang = request.form['language']
        new_title = request.form['title']
        iD = books_list[-1]['id']+1

        new_obj = {
            'id': iD,
            'author': new_author,
            'language': new_lang,
            'title': new_title
        }

        books_list.append(new_obj)
        return jsonify(books_list), 201


if __name__ == '__main__':
    app.run(debug=True)
