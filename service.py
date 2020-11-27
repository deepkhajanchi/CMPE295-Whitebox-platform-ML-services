from __future__ import absolute_import, division, print_function

import os
import sys

import tensorflow as tf
keras = tf.keras

from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import pgdb
import datetime
from string import Template

BATCH_SIZE = 1000

def init_db_connection():
    if os.environ.get('DB_HOST') != None and os.environ.get('DB_DATABASE') != None and os.environ.get('DB_USERNAME') != None and os.environ.get('DB_PASSWORD') != None:
        connection = pgdb.connect(
            host=os.environ['DB_HOST'],
            database=os.environ['DB_DATABASE'],
            user=os.environ['DB_USERNAME'],
            password=os.environ['DB_PASSWORD'],
        )
    else:
        connection = pgdb.connect(
            host="localhost",
            database="cmpe295b_v5",
            user="postgres",
            password="root",
        )
    connection.autocommit = True
    return connection

def load_model(path):
    return keras.models.load_model(path)


def get_layer_outputs(loaded_model, test_data):
    inputs = loaded_model.input                                           # input placeholder
    outputs = [layer.output for layer in loaded_model.layers]          # all layer outputs
    functions = [K.function([inputs], [out]) for out in outputs]   # evaluation function

    return [func([test_data]) for func in functions]

def batch_insert(cur, query, values):
    statement = query + ",".join(values) + ";"
    cur.execute(statement)

###### Importing ######

def save_model_to_db(connection, model, name, filePath, profileId):
    try:
        cur = connection.cursor()
        cur.execute("""
            INSERT INTO models(name, "createdAt", "updatedAt", "profileId", "path") VALUES (
                %(name)s,
                %(createdAt)s,
                %(updatedAt)s,
                %(profileId)s,
                %(path)s
            );
        """, {
            "name": name,
            "createdAt": datetime.datetime.now().isoformat(),
            "updatedAt": datetime.datetime.now().isoformat(),
            "profileId": profileId,
            "path": filePath
        })
        
        cur.execute('select * from models order by "createdAt" desc;')
        createdModel = cur.fetchone()
        
        shape = []

        for s in model.layers:
            shape.append(s.output_shape[1])
        
        cur.execute("""
            INSERT INTO configurations(name, "isOriginal", "layerNum", "layerShape", "activationFunction", regulation, "learningRate", "createdAt", "updatedAt", "modelId", "status") VALUES (
                %(name)s,
                %(isOriginal)s,
                %(layerNum)d,
                %(layerShape)s,
                %(activationFunction)s,
                %(regulation)s,
                %(learningRate)s,
                %(createdAt)s,
                %(updatedAt)s,
                %(modelId)d,
                %(status)s
            );
        """, {
            "name": "configuration model " + str(createdModel.id),
            "isOriginal": "true",
            "layerNum": len(model.layers),
            "layerShape": str(shape),
            "activationFunction": "SIGMOID",
            "regulation": "",
            "learningRate": 0.2,
            "createdAt": datetime.datetime.now().isoformat(),
            "updatedAt": datetime.datetime.now().isoformat(),
            "modelId": createdModel.id,
            "status": "INITIALIZING"
        })
        
        cur.execute('select * from configurations order by "createdAt" desc;')
        createdConfiguration = cur.fetchone()
        
        return createdModel, createdConfiguration
    except Exception as error:
        print ("Oops! An exception has occured 1:", error)
        print ("Exception TYPE:", type(error))


def save_layers_to_db(connection, model, savedModel, savedConfiguration, startLayerIdx = 0):
    try:
        cur = connection.cursor()
        
        cur.execute("""
            INSERT INTO layers(id, name, type, data, "createdAt", "updatedAt", "configurationId") VALUES (
                %(id)s,
                %(name)s,
                %(type)s,
                %(data)s,
                %(createdAt)s,
                %(updatedAt)s,
                %(configurationId)s
            );
        """, {
            "id": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "lI",
            "name": model.layers[startLayerIdx].name + "_INPUT",
            "type": "INPUT",
            "data": """{ "nodeCount": """ + str(model.layers[startLayerIdx].input_shape[1]) + """}""",
            "createdAt": datetime.datetime.now().isoformat(),
            "updatedAt": datetime.datetime.now().isoformat(),
            "configurationId": savedConfiguration.id
        })
        
        for idx in range(startLayerIdx, len(model.layers)):
            layer = model.layers[idx]
            
            cur.execute("""
                INSERT INTO layers(id, name, type, data, "createdAt", "updatedAt", "configurationId") VALUES (
                    %(id)s,
                    %(name)s,
                    %(type)s,
                    %(data)s,
                    %(createdAt)s,
                    %(updatedAt)s,
                    %(configurationId)s
                );
            """, {
                "id": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "l" + str(idx),
                "name": layer.name,
                "type": "OUTPUT" if idx == len(model.layers) - 1 else "HIDDEN",
                "data": """{ "nodeCount": """ + str(layer.output_shape[1]) + """}""",
                "createdAt": datetime.datetime.now().isoformat(),
                "updatedAt": datetime.datetime.now().isoformat(),
                "configurationId": savedConfiguration.id
            })
        
    except Exception as error:
        print ("Oops! An exception has occured 2:", error)
        print ("Exception TYPE:", type(error))


def save_neurons_to_db(connection, model, savedModel, savedConfiguration, startLayerIdx = 0):
    try:
        cur = connection.cursor()
        base_statement = """INSERT INTO neurons(id, bias, type, "activationFunction", "createdAt", "updatedAt", "layerId") VALUES """
        template = Template("""('$id', $bias, '$type', '$activationFunction', '$createdAt', '$updatedAt', '$layerId')""")
        buffer = []
        for neuronIdx in range(model.layers[startLayerIdx].input_shape[1]):
            payload = {
                "id": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "lI" + "n" + str(neuronIdx),
                "bias": 0,
                "type": "",
                "activationFunction": "SIGMOID",
                "createdAt": datetime.datetime.now().isoformat(),
                "updatedAt": datetime.datetime.now().isoformat(),
                "layerId": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "lI"
            }
            buffer.append(template.substitute(**payload))
            if (neuronIdx + 1) % BATCH_SIZE == 0:
                batch_insert(cur, base_statement, buffer)
                buffer = []
        if len(buffer) > 0:
            batch_insert(cur, base_statement, buffer)
            buffer = []    
        
        for layerIdx in range(startLayerIdx, len(model.layers)):
            layer = model.layers[layerIdx]
            
            weight_n_bias = layer.get_weights()
            
            
            biases = None 
            if len(weight_n_bias) == 2:
                bias = weight_n_bias[1]
            
            for neuronIdx in range(model.layers[layerIdx].output_shape[1]):
                payload = {
                    "id": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "l" + str(layerIdx) + "n" + str(neuronIdx),
                    "bias": 0 if biases == None else bias[neuronIdx].item(),
                    "type": "",
                    "activationFunction": "SIGMOID",
                    "createdAt": datetime.datetime.now().isoformat(),
                    "updatedAt": datetime.datetime.now().isoformat(),
                    "layerId": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "l" + str(layerIdx)
                }
                buffer.append(template.substitute(**payload))
                if len(buffer) % BATCH_SIZE == 0:
                    batch_insert(cur, base_statement, buffer)
                    buffer = []
            if len(buffer) > 0:
                batch_insert(cur, base_statement, buffer)
                buffer = [] 
           
    except Exception as error:
        print ("Oops! An exception has occured 3:", error)
        print ("Exception TYPE:", type(error))


def save_links_to_db(connection, model, savedModel, savedConfiguration, startLayerIdx = 0):
    try:
        cur = connection.cursor()
        
        layer = model.layers[startLayerIdx]
            
        weight_n_bias = layer.get_weights()

        if len(weight_n_bias) == 2:
            weights = weight_n_bias[0]
        else:
            weights = model.layers[startLayerIdx].get_weights()[0]

        base_statement = """INSERT INTO links(id, weight, "createdAt", "updatedAt", "sourceId", "destId", "neuronId") VALUES """    
        template = Template("""('$id', $weight, '$createdAt', '$updatedAt', '$sourceId', '$destId', '$neuronId')""")
        buffer = []
        print("processing Input layer ")
        
        for srcNeuronIdx in range(model.layers[startLayerIdx].input_shape[1]):
            if srcNeuronIdx % 100 == 0 or srcNeuronIdx == model.layers[startLayerIdx].input_shape[1] - 1:
                print("processing neuron " + str(srcNeuronIdx))
            for destNeuronIdx in range(model.layers[startLayerIdx].output_shape[1]):
                payload = {
                    "id": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "lI" + "n" + str(srcNeuronIdx) + "-l" + str(startLayerIdx) + "n" + str(destNeuronIdx),
                    "weight":  weights[srcNeuronIdx][destNeuronIdx].item(),
                    "createdAt": datetime.datetime.now().isoformat(),
                    "updatedAt": datetime.datetime.now().isoformat(),
                    "sourceId": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "lI" + "n" + str(srcNeuronIdx),
                    "destId": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "l" + str(startLayerIdx) + "n" + str(destNeuronIdx),
                    "neuronId": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "lI" + "n" + str(srcNeuronIdx)
                }
                buffer.append(template.substitute(**payload))
                if len(buffer) % BATCH_SIZE == 0:
                    batch_insert(cur, base_statement, buffer)
                    buffer = []
                    
            if len(buffer) > 0:
                batch_insert(cur, base_statement, buffer)
                buffer = [] 

        
        for layerIdx in range(startLayerIdx + 1, len(model.layers)):
            print("processing layer " + str(layerIdx))
            layer = model.layers[layerIdx]
            
            weight_n_bias = layer.get_weights()
            
            if len(weight_n_bias) == 2:
                weights = weight_n_bias[0]
            else:
                weights = model.layers[layerIdx - 1].get_weights()[0]
            
            for srcNeuronIdx in range(model.layers[layerIdx].input_shape[1]):
                if srcNeuronIdx % 100 == 0 or srcNeuronIdx == model.layers[layerIdx].input_shape[1] - 1:
                    print("processing neuron " + str(srcNeuronIdx))
                for destNeuronIdx in range(model.layers[layerIdx].output_shape[1]):  
                    payload = {
                        "id": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "l" + str(layerIdx - 1) + "n" + str(srcNeuronIdx) + "-l" + str(layerIdx) + "n" + str(destNeuronIdx),
                        "weight":  weights[srcNeuronIdx][destNeuronIdx].item(),
                        "createdAt": datetime.datetime.now().isoformat(),
                        "updatedAt": datetime.datetime.now().isoformat(),
                        "sourceId": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "l" + str(layerIdx - 1) + "n" + str(srcNeuronIdx),
                        "destId": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "l" + str(layerIdx) + "n" + str(destNeuronIdx),
                        "neuronId": "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "l" + str(layerIdx - 1) + "n" + str(srcNeuronIdx)
                    }
                    buffer.append(template.substitute(**payload))
                    if len(buffer) % BATCH_SIZE == 0:
                        batch_insert(cur, base_statement, buffer)
                        buffer = []
                        
                if len(buffer) > 0:
                    batch_insert(cur, base_statement, buffer)
                    buffer = []
        
    except Exception as error:
        print ("Oops! An exception has occured 4:", error)
        print ("Exception TYPE:", type(error))

def import_model(profileId, loaded_model, filename, filePath, options = {}):
    connection = init_db_connection()
    
    # Capture various elements of model
    savedModel, savedConfiguration = save_model_to_db(connection, loaded_model, filename, filePath, profileId)
    save_layers_to_db(connection, loaded_model, savedModel, savedConfiguration, options['startLayerIdx'])
    save_neurons_to_db(connection, loaded_model, savedModel, savedConfiguration, options['startLayerIdx'])
    save_links_to_db(connection, loaded_model, savedModel, savedConfiguration, options['startLayerIdx'])
    
    # Flag model as ready
    cur = connection.cursor()

    cur.execute("""
            update configurations set status = 'READY', "updatedAt" = current_timestamp where id = %(id)s;
        """, {
            "id": savedConfiguration.id
        })

    # Clean up
    connection.close()

###### Testing ######
def create_test_to_db(connection, model, savedModel, savedConfiguration, profileId, testName, image_path):
    try:
        cur = connection.cursor()
       
        cur.execute("""
            INSERT INTO tests(name, status, input, "timestamp", "createdAt", "updatedAt", "profileId", "configurationId") VALUES (
                %(name)s,
                %(status)s,
                %(input)s,
                %(timestamp)s,
                %(createdAt)s,
                %(updatedAt)s,
                %(profileId)s,
                %(configurationId)s
            );
        """, {
            "name": testName,
            "status": "RUNNING",
            "input": image_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "createdAt": datetime.datetime.now().isoformat(),
            "updatedAt": datetime.datetime.now().isoformat(),
            "profileId": profileId,
            "configurationId": savedConfiguration['id']
        })
           
        cur.execute('select * from tests order by "createdAt" desc;')
        createdTest = cur.fetchone()
        
        return createdTest
        
    except Exception as error:
        print ("Oops! An exception has occured:", error)
        print ("Exception TYPE:", type(error))

def capture_test_result_db(connection, model, savedModel, savedConfiguration, test, outputs, inputData, startLayerIdx = 0):
    try:
        cur = connection.cursor()
       
        #Create a test
        cur.execute("""
            INSERT INTO results(name, timestamp, "confusionMatrix", loss, accuracy, "createdAt", "updatedAt", "testId") VALUES (
                %(name)s,
                %(timestamp)s,
                %(confusionMatrix)s,
                %(loss)s,
                %(accuracy)s,
                %(createdAt)s,
                %(updatedAt)s,
                %(testId)s
            );
        """, {
            "name": "First Test",
            "timestamp": datetime.datetime.now().isoformat(),
            "confusionMatrix": "[[368,24],[2,164]]",
            "loss": 0.2,
            "accuracy": 0.9,
            "createdAt": datetime.datetime.now().isoformat(),
            "updatedAt": datetime.datetime.now().isoformat(),
            "testId": test.id
        })
        
        cur.execute('select * from results order by "createdAt" desc;')
        createdResult = cur.fetchone()
        
        base_statement = """INSERT INTO "nodeResults"(output, input, "createdAt", "updatedAt", "resultId", "neuronId") VALUES """    
        template = Template("""($output, '$input', '$createdAt', '$updatedAt', $resultId, '$neuronId')""")
        buffer = []
        
        
        for neuronIdx in range(model.layers[startLayerIdx].input_shape[1]):
            payload = {
                "output": inputData[neuronIdx].item(),
                "input": "{}",
                "createdAt": datetime.datetime.now().isoformat(),
                "updatedAt": datetime.datetime.now().isoformat(),
                "resultId": createdResult.id,
                "neuronId":  "m" + str(savedModel['id']) + "c" + str(savedConfiguration['id']) + "lI" + "n" + str(neuronIdx),
            }
            buffer.append(template.substitute(**payload))
            if len(buffer) % BATCH_SIZE == 0:
                batch_insert(cur, base_statement, buffer)
                buffer = []
        if len(buffer) > 0:
                batch_insert(cur, base_statement, buffer)
                buffer = []
        
        
        
        for layerIdx in range(startLayerIdx, len(model.layers)):
            layer = model.layers[layerIdx]
            
            for neuronIdx in range(layer.output_shape[1]):
                
#                 cur.execute("""
#                     INSERT INTO "nodeResults"(output, input, "createdAt", "updatedAt", "resultId", "neuronId") VALUES (
#                         %(output)s,
#                         %(input)s,
#                         %(createdAt)s,
#                         %(updatedAt)s,
#                         %(resultId)s,
#                         %(neuronId)s
#                     );
#                 """, {
#                     "output": outputs[layerIdx][0][0][neuronIdx].item(),
#                     "input": "{}",
#                     "createdAt": datetime.datetime.now().isoformat(),
#                     "updatedAt": datetime.datetime.now().isoformat(),
#                     "resultId": createdResult.id,
#                     "neuronId":  "m" + str(savedModel.id) + "c" + str(savedConfiguration.id) + "l" + str(layerIdx) + "n" + str(neuronIdx),
#                 })
                
                payload = {
                    "output": outputs[layerIdx][0][0][neuronIdx].item(),
                    "input": "{}",
                    "createdAt": datetime.datetime.now().isoformat(),
                    "updatedAt": datetime.datetime.now().isoformat(),
                    "resultId": createdResult.id,
                    "neuronId":  "m" + str(savedModel['id']) + "c" + str(savedConfiguration['id']) + "l" + str(layerIdx) + "n" + str(neuronIdx),
                }
                buffer.append(template.substitute(**payload))
                if len(buffer) % BATCH_SIZE == 0:
                    batch_insert(cur, base_statement, buffer)
                    buffer = []
            if len(buffer) > 0:
                    batch_insert(cur, base_statement, buffer)
                    buffer = []
                    
                    
        cur.execute("""
            update tests set status = 'DONE' where "id" = %(testId)s;
        """, {
            "testId": test.id
        })
    except Exception as error:
        print ("Oops! An exception has occured:", error)
        print ("Exception TYPE:", type(error))

def load_images(path, height, width, batch_size):
    image_data_generator = ImageDataGenerator(rescale=1./255)

    # PATH = '/Users/hieutran/Desktop/school/CMPE295/modelLib/imageData'
    # data_dir = os.path.join(PATH, 'test')

    image_generator = image_data_generator.flow_from_directory(
        batch_size=batch_size,
        directory=path,
        shuffle=False,
        target_size=(height, width),
        class_mode='categorical')

    return image_generator

def get_highest_idx(layer_output):
    # last_layer_output = layer_outputs[-1][0][0]
    highest = 0
    highest_idx = 0
    for idx in range(0, len(layer_output)):
        if highest < layer_output[idx]:
            highest = layer_output[idx]
            highest_idx = idx
    return highest_idx

def test_images(model, profileId, test_name, savedModel, savedConfiguration, data, expected_label, options = {}):
    connection = init_db_connection()

    # Create a test entry
    createdTest = create_test_to_db(connection, model, savedModel, savedConfiguration, profileId, test_name, options['image_path'])

    # Get the immediate output at the previous layer of the current layer
    inter_output_model = keras.Model(model.input, model.get_layer(index = options['startLayerIdx'] - 1).output )
    inter_output = inter_output_model.predict(data)

    # Compute layer outputs 
    layer_outputs = get_layer_outputs(model, data)

    # Save output to db
    capture_test_result_db(connection, model, savedModel, savedConfiguration, createdTest, layer_outputs, inter_output[0], options['startLayerIdx'])

def get_model(modelId):
    connection = init_db_connection()
    cur = connection.cursor()
    cur.execute('select * from models where id = ' + str(modelId) + ';')
    savedModel = cur.fetchone()
    connection.close()
    return savedModel

def get_dataset(datasetId):
    connection = init_db_connection()
    cur = connection.cursor()
    cur.execute('select * from datasets where id = ' + str(datasetId) + ';')
    savedModel = cur.fetchone()
    connection.close()
    return savedModel