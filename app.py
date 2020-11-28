import urllib.request
from flask import Flask
from flask import request
from flask import send_from_directory
import zipfile
import shutil
from service import *
app = Flask(__name__)

SERVER_URL = "http://localhost:2000"
if os.environ.get('SERVER_URL'):
    SERVER_URL = os.environ.get('SERVER_URL')

@app.route('/')
def hello():
    return "LandingPage"

@app.route('/dataset/<path:path>')
def send_file(path):
    print("path: " + path)
    return send_from_directory('dataset', path)

@app.route('/import')
def import_model_route():
    profileId = request.args.get('profileId')
    fileUrl = request.args.get('fileUrl')
    fileName = request.args.get('fileName')

    urllib.request.urlretrieve(fileUrl, "temp/" + fileName)

    filePath = "temp/" + fileName
    print("./" + filePath)
    model = load_model("./" + filePath)

    startLayerIdx = 0

    for idx in range(0, len(model.layers)):
        if model.layers[idx].__class__.__name__ == 'Flatten':
            startLayerIdx = idx + 1


    options = {
        "startLayerIdx": startLayerIdx
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
    datasetId = request.args.get('datasetId')

    db_model = get_model(modelId)
    print(db_model)

    model = load_model(db_model.path)
    
    # Prepare dataset
    dataset = get_dataset(datasetId)

    dir_path = "dataset/" + str(datasetId)
    path = dir_path + "/temp.zip"

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    
    urllib.request.urlretrieve(dataset.path, path)

    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(dir_path)

    os.remove(path)
    if os.path.exists(dir_path + "/__MACOSX"):
        shutil.rmtree(dir_path + "/__MACOSX") # Remove this folder if any at all

    # # PATH = '/Users/hieutran/Desktop/school/CMPE295/modelLib/imageData'
    # PATH = data_path
    # data_dir = os.path.join(PATH, 'test')
    startLayerIdx = 0

    for idx in range(0, len(model.layers)):
        if model.layers[idx].__class__.__name__ == 'Flatten':
            startLayerIdx = idx + 1

    width = model.layers[0].input.shape[1]
    height = model.layers[0].input.shape[2]

    image_generator = load_images(dir_path, width, height, 1)

    labels = os.listdir(dir_path)
    
    # image_dir_path = os.path.join(dir_path, label)
    # # image_path = os.path.join(image_dir_path, os.listdir(image_dir_path)[0])
    # image_path = "http://localhost:2000/" + dir_path + "/" + label + "/" + os.listdir(image_dir_path)[0

    total_count = image_generator.__len__()
    i = 0
    fname = []

    for root,d_names,f_names in os.walk(dir_path):
        for f in f_names:
            fname.append(os.path.join(root, f))

    while(i < total_count):
        data = image_generator.next()

        layer_outputs = get_layer_outputs(model, data)

        expected_label = labels[get_highest_idx(data[1][0])]

        image_dir_path = os.path.join(dir_path, expected_label)
        image_path = SERVER_URL + "/" + fname[i]

        i = i + 1
        test_images(model, profileId, testName + str(i), {"id": modelId}, {"id": configurationId}, image_generator.next(), expected_label, {"startLayerIdx": startLayerIdx, "image_path": image_path})
        
    return "ok"


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=2000)
