import urllib.request
from flask import Flask
from flask import request
from service import *
app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World!"

@app.route('/hieu')
def hello1():
    return "Hello World!1"

@app.route('/import')
def import_model_route():
    profileId = request.args.get('profileId')
    fileUrl = request.args.get('fileUrl')
    fileName = request.args.get('fileName')

    urllib.request.urlretrieve(fileUrl, "temp/" + fileName)

    filePath = "temp/" + fileName

    model = load_model(filePath)
    options = {
        "startLayerIdx": 9
    }
    import_model(profileId, model, fileName, filePath, options)
    return "ok"


@app.route('/test')
def run_test():
    profileId = request.args.get('profileId')
    testName = request.args.get('name')
    modelId = request.args.get('modelId')
    data_path = request.args.get('dataPath')
    configurationId = request.args.get('cId')

    db_model = get_model(modelId)
    print(db_model)

    model = load_model(db_model.path)
    # PATH = '/Users/hieutran/Desktop/school/CMPE295/modelLib/imageData'
    PATH = data_path
    data_dir = os.path.join(PATH, 'test')
    image_generator = load_images(data_dir, 100, 100, 1)


    label = os.listdir(data_dir)[0]
    
    image_dir_path = os.path.join(data_dir, label)
    # image_path = os.path.join(image_dir_path, os.listdir(image_dir_path)[0])
    image_path = "/imageData/test/" + label + "/" + os.listdir(image_dir_path)[0]

    test_images(model, profileId, testName, image_generator,{"id": modelId}, {"id": configurationId}, {"startLayerIdx": 9, "image_path": image_path})
    return "ok"


if __name__ == '__main__':
    app.run(port=2000)
