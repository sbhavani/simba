import os
from bottle import get, run, template, post, request, route, static_file, redirect

from classify import classify_lion

upload_dir = 'uploads'
result = ''

# Static Routes
@get('/<filename:re:.*\.js>')
def javascripts(filename):
    return static_file(filename, root='static/js')


@get('/<filename:re:.*\.css>')
def stylesheets(filename):
    return static_file(filename, root='static/css')


@get('/<filename:re:.*\.(jpg|png|gif|ico)>')
def images(filename):
    return static_file(filename, root='static/img')


@get('/<filename:re:.*\.(eot|ttf|woff|svg)>')
def fonts(filename):
    return static_file(filename, root='static/fonts')


@route('/')
@route('/<lion>')
def index(lion='Unknown'):
    return template('index.html', lion=lion)


@post('/upload')
def do_upload():
    lion = request.files

    # add this line
    data = request.files.data

    print(lion, type(lion))

    if lion is not None:
        file = lion.file
        print(file.filename, type(file))
        target_path = os.path.join(upload_dir, file.filename)

        file.save(target_path)

        lion_pred = classify_lion([target_path])
    else:
        return "<p>Invalid lion.</p>"


if __name__ == '__main__':
    port = os.environ.get('PORT', 8080)

    # Run the app.
    run(host='0.0.0.0', port=port)