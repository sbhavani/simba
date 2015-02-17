import os

# Paths:
ROOT = os.path.abspath(os.path.dirname(__file__))
TEMPLATES = '{base}/templates'.format(base=ROOT)
PAGES = '{base}/pages'.format(base=TEMPLATES)
ASSETS = '{base}/assets'.format(base=ROOT)
IMAGES = '{base}/img'.format(base=ASSETS)

# Server:
SERVER = 'wsgiref'
HOST = 'localhost'
PORT = 8080

# Upload:
UPLOAD_MAX_SIZE = (1024 * 1024) * 2 # (1024 bytes -> 1kb, 1024kb -> 1mb)
UPLOAD_ALLOWED_EXTENSIONS = ['ico', 'gif', 'png', 'jpg', 'jpeg', 'JPG', 'JPEG']
UPLOAD_ALLOWED_MIME_TYPES = [
    'image/x-icon',
    'image/gif',
    'image/png',
    'image/jpeg'
]