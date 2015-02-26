from bottle import get, run, static_file, route, template, request

from classify import classify_lion

import os

import config

# Global Variables
# original dataset
DATASETPATH = 'testset'
# features extracted from each superpixel
FEATUREPATH = 'feature_files'
# labels for patient data
LABELSPATH = 'label_files'
# segmentation files
SEGPATH = 'segmentation_files'

ALL_PATHS = [DATASETPATH, FEATUREPATH, LABELSPATH, SEGPATH]

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
def index():
    tpl = '{base}/index.tpl'.format(base=config.PAGES)
    return template(tpl)


@route('/upload<:re:/?>', method=['GET', 'POST'])
def upload():
    tpl = '{base}/upload.tpl'.format(base=config.PAGES)
    tpl_json = dict(
        file=request.files.get('file'),
        upload=request.forms.get('upload'),
        file_name='',
        error=''
    )

    if tpl_json['upload']:
        if not tpl_json['file']:
            tpl_json['error'] = 'There was no file to upload'
        else:
            # Note before starting the upload:
            # We need to assign the file-object to another variable upon doing
            # read() because it will become empty / unreusable afterwards: I
            # don't know the reason behind it though :(
            file_contents = tpl_json['file'].file.read()
            file_extension = tpl_json['file'].filename.split('.')[-1]

            if not len(file_contents):
                tpl_json['error'] = 'The uploaded file is empty'
            elif len(file_contents) > config.UPLOAD_MAX_SIZE:
                tpl_json['error'] = \
                    'Exceeded allowed file size: ' \
                    'Only {file_size}MB is allowed'.format(
                        file_size=int(config.UPLOAD_MAX_SIZE / (1024 * 1024))
                    )
            elif '.' in tpl_json['file'].filename \
            and file_extension not in config.UPLOAD_ALLOWED_EXTENSIONS:
                tpl_json['error'] = \
                    'Only {exts} and {ext} ' \
                    'extensions are allowed'.format(
                        exts=', '.join(config.UPLOAD_ALLOWED_EXTENSIONS[:-1]),
                        ext=config.UPLOAD_ALLOWED_EXTENSIONS[-1]
                    )
            else:
                file_path = '{base}/{file_name}'.format(base=config.IMAGES,
                                                        **tpl_json)

                # Let's create the destination directories if it doesn't exist:
                if not os.path.exists(config.IMAGES):
                    os.makedirs(config.IMAGES)

                tpl_json['file'].save(file_path)

                print(file_path)

                lion_pred = classify_lion([os.path.join(file_path, tpl_json['file'].filename)])

                tpl_json['lion'] = lion_pred

    return template(tpl, tpl_json)


if __name__ == '__main__':
    port = os.environ.get('PORT', 8080)

    print("---------------------")
    print("## Creating directories")
    for curr_path in ALL_PATHS:
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)

    # Run the app.
    run(host='0.0.0.0', port=port)