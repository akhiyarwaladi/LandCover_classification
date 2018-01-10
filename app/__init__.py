from datetime import datetime
from celery import Celery
from flask import Flask, render_template, request, flash
from redis import StrictRedis
from socketio import socketio_manage
from socketio.namespace import BaseNamespace
from celery.task.control import revoke
from arcpy.sa import *
from assets import assets
from sklearn.externals import joblib

import time
import urllib2
import config
import os
import ftpLahan as ftl
import maskCloud as mc
import shutil
import time
import data_process as dp
import arcpy
import pandas as pd
import numpy as np
import os.path

redis = StrictRedis(host=config.REDIS_HOST)
redis.delete(config.MESSAGES_KEY)
redis.delete(config.MESSAGES_KEY_2)
# celery = Celery(__name__)
# celery.config_from_object(celeryconfig)

app = Flask(__name__)
app.config.from_object(config)
assets.init_app(app)

app.config['SECRET_KEY'] = 'top-secret!'
app.config['SOCKETIO_CHANNEL'] = 'tail-message'
app.config['MESSAGES_KEY'] = 'tail'
app.config['CHANNEL_NAME'] = 'tail-channel'

app.config['SOCKETIO_CHANNEL_2'] = 'val-message'
app.config['MESSAGES_KEY_2'] = 'val'
app.config['CHANNEL_NAME_2'] = 'val-channel'

app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


def internet_on():
    for timeout in [1,5,10,15]:
        try:
            response=urllib2.urlopen('http://google.com',timeout=timeout)
            return True
        except urllib2.URLError as err: pass
    return False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if(internet_on()):
            if redis.llen(config.MESSAGES_KEY):
                flash('Task is already running', 'error')
            # elif(redis.llen(config.MESSAGES_KEY) == 0):
            #     flash('Task is finished', 'success')
            else:
                tail.delay()
                flash('Task started. Please wait until complete', 'info')
        else:
             flash('Internet connection is bad. Please pay your internet bill :)','error')

    return render_template('index.html')

@app.route('/socket.io/<path:remaining>')
def socketio(remaining):
    socketio_manage(request.environ, {
        '/tail': TailNamespace
    })
    return app.response_class()

@app.route('/stop', methods=['GET', 'POST'])
def stop():
    if request.method == 'POST':
        tail.delay()
    return render_template('index.html')

@celery.task
def tail():

    msg = str(datetime.now()) + '\t' + "Importing Library ... \n"
    redis.rpush(config.MESSAGES_KEY, msg)
    redis.publish(config.CHANNEL_NAME, msg)

    scene_id = ftl.downloadFile()

    dataPath =  config.dataPath + scene_id
    out_reflectance = dataPath + "/reflectance"
    modelPath = config.modelPath
    outputPath = config.outputPath
    shpPath = config.shpPath

    ######################### check output directory #########################
    if(os.path.exists(outputPath + scene_id)):
        print "dir exists"
        shutil.rmtree(outputPath + scene_id)
        time.sleep(3)
        os.makedirs(outputPath + scene_id)

    else:
        print "make dir"
        os.makedirs(outputPath + scene_id)

    ##############################################################################

    ######################### Make mask cloud ############################
    masktype = 'Cloud'
    confidence = 'High'
    cummulative = 'false'   
    mc.mask_cloud(dataPath, masktype, confidence, cummulative, outputPath + scene_id)
    #######################################################################


    ######################### Change to reflectance ##############################
    text_files = [f for f in os.listdir(dataPath) if f.endswith('.TIF') or f.endswith('.tif')]
    inFC = os.path.join(dataPath, text_files[0])

    SR = arcpy.Describe(inFC).spatialReference
    data_type = "LANDSAT_8"
    os.makedirs(out_reflectance)
    dp.process_landsat(dataPath, SR, out_reflectance, "_Pre", data_type, "")
    ###############################################################################

    dataPath = "C:/data/lahan/input/LC81190652017291LGN00/reflectance/processed_Pre/toa"
    
  
    ###################### Begin preprocessing and load data ##########################
    msg = str(datetime.now()) + '\t' + "Processing file "+dataPath+"\n"
    redis.rpush(config.MESSAGES_KEY, msg)
    redis.publish(config.CHANNEL_NAME, msg)

    rasterarrayband6 = arcpy.RasterToNumPyArray(dataPath + "/" + scene_id + "_B6.TIF")
    rasterarrayband5 = arcpy.RasterToNumPyArray(dataPath + "/" + scene_id + "_B5.TIF")
    rasterarrayband3 = arcpy.RasterToNumPyArray(dataPath + "/" + scene_id + "_B3.TIF")

    print("Change raster format to numpy array")
    data = np.array([rasterarrayband6.ravel(), rasterarrayband5.ravel(), rasterarrayband3.ravel()], dtype=np.float16)
    data = data.transpose()

    print("Change to dataframe format")

    msg = str(datetime.now()) + '\t' + "Change to dataframe format \n"
    redis.rpush(config.MESSAGES_KEY, msg)
    redis.publish(config.CHANNEL_NAME, msg)
    #time.sleep(1)

    columns = ['band6', 'band5', 'band3']
    df = pd.DataFrame(data, columns=columns)
    del data

    print("Split data to 20 chunks ")
    msg = str(datetime.now()) + '\t' + "Split data to 20 chunks \n"
    redis.rpush(config.MESSAGES_KEY, msg)
    redis.publish(config.CHANNEL_NAME, msg)
    #time.sleep(1)

    df_arr = np.array_split(df, 25)
    clf = joblib.load(modelPath) 
    kelasAll = []

    for i in range(len(df_arr)):
        
        print ("predicting data chunk-%s\n" % i)
        msg = str(datetime.now()) + '\t' + "predicting data chunk-%s\n" % i
        redis.rpush(config.MESSAGES_KEY, msg)
        redis.publish(config.CHANNEL_NAME, msg)

        msg2 = i
        redis.rpush(config.MESSAGES_KEY_2, msg2)
        redis.publish(config.CHANNEL_NAME_2, msg2)
        #time.sleep(1)
        kelas = clf.predict(df_arr[i])
        dat = pd.DataFrame()
        dat['kel'] = kelas
        print ("mapping to integer class")
        msg = str(datetime.now()) + '\t' + "mapping to integer class \n"
        redis.rpush(config.MESSAGES_KEY, msg)
        redis.publish(config.CHANNEL_NAME, msg)
        #time.sleep(1)
        mymap = {'cloudshodow':1, 'air':2, 'tanah':3, 'vegetasi':4}
        dat['kel'] = dat['kel'].map(mymap)

        band1Array = dat['kel'].values
        print ("extend to list")
        msg = str(datetime.now()) + '\t' + "extend to list \n"
        redis.rpush(config.MESSAGES_KEY, msg)
        redis.publish(config.CHANNEL_NAME, msg)
        #time.sleep(1)

        kelasAll.extend(band1Array.tolist())

    del df_arr
    del clf
    del kelas
    del dat
    del band1Array
    del data

    print ("change list to np array")
    msg = str(datetime.now()) + '\t' + "change list to np array \n"
    redis.rpush(config.MESSAGES_KEY, msg)
    redis.publish(config.CHANNEL_NAME, msg)

    kelasAllArray = np.array(kelasAll, dtype=np.uint8)

    print ("reshaping np array")
    msg = str(datetime.now()) + '\t' + "reshaping np array \n"
    redis.rpush(config.MESSAGES_KEY, msg)
    redis.publish(config.CHANNEL_NAME, msg)

    band1 = np.reshape(kelasAllArray, (-1, rasterarrayband6[0].size))
    band1 = band1.astype(np.uint8)

    # raster = arcpy.Raster(dataPath)
    # inputRaster = dataPath
    raster = arcpy.Raster(dataPath + "/" + scene_id + "_B6.TIF")
    inputRaster = dataPath + "/" + scene_id + "_B6.TIF"
    print inputRaster

    spatialref = arcpy.Describe(inputRaster).spatialReference
    cellsize1  = raster.meanCellHeight
    cellsize2  = raster.meanCellWidth
    extent     = arcpy.Describe(inputRaster).Extent
    pnt        = arcpy.Point(extent.XMin,extent.YMin)

    del raster
    del kelasAllArray

    # save the raster
    print ("numpy array to raster ..")
    msg = str(datetime.now()) + '\t' + "numpy array to raster .. \n"
    redis.rpush(config.MESSAGES_KEY, msg)
    redis.publish(config.CHANNEL_NAME, msg)

    out_ras = arcpy.NumPyArrayToRaster(band1, pnt, cellsize1, cellsize2)

    #arcpy.CheckOutExtension("Spatial")
    print ("define projection ..")
    msg = str(datetime.now()) + '\t' + "define projection ..\n"
    redis.rpush(config.MESSAGES_KEY, msg)
    redis.publish(config.CHANNEL_NAME, msg)

    print (outputPath + scene_id)

    arcpy.CopyRaster_management(out_ras, outputPath  + scene_id +"/"+scene_id+"_classified.TIF")
    arcpy.DefineProjection_management(outputPath  + scene_id  +"/"+scene_id+"_classified.TIF", spatialref)

    print("Masking Cloud")
    arcpy.CheckOutExtension("Spatial")
    mask = Raster(outputPath + scene_id + "/" + 'mask_cloud_' + scene_id + '.TIF')
    inRas = Raster(outputPath  + scene_id + "/" + scene_id + "_classified.TIF")
    outRas = Con((mask == 0), inRas, 1)

    outRas2 = SetNull(inRas == 1, inRas)
    outRas2.save(outputPath + scene_id + "/" + scene_id + "_maskCloud.TIF")

    print("Masking with shp indonesia")
    arcpy.CheckOutExtension("Spatial")
    inMaskData = os.path.join(shpPath, "INDONESIA_PROP.shp")
    outExtractByMask = ExtractByMask(outRas2, inMaskData)
    outExtractByMask.save(outputPath + scene_id + "/" + scene_id + "_maskShp.TIF")

    del out_ras
    del band1
    del spatialref
    del extent
    arcpy.Delete_management("in_memory")

    print("Finished")
    msg = str(datetime.now()) + '\t' + "Finished ... \n"
    redis.rpush(config.MESSAGES_KEY, msg)
    redis.publish(config.CHANNEL_NAME, msg)

    redis.delete(config.MESSAGES_KEY)
    redis.delete(config.MESSAGES_KEY_2)


def splitDataFrameIntoSmaller(df, chunkSize = 10000): 
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf

class TailNamespace(BaseNamespace):
    def listener(self):
        # Emit the backlog of messages
        messages = redis.lrange(config.MESSAGES_KEY, 0, -1)        
        messages2 = redis.lrange(config.MESSAGES_KEY_2, 0, -1)

        print(messages2)
        self.emit(config.SOCKETIO_CHANNEL, ''.join(messages))
        self.emit(config.SOCKETIO_CHANNEL_2, ''.join(messages2))

        self.pubsub.subscribe(config.CHANNEL_NAME)
        self.pubsub.subscribe(config.CHANNEL_NAME_2)
        i=8
        for m in self.pubsub.listen():
            if m['type'] == 'message':
                self.emit(config.SOCKETIO_CHANNEL, m['data'])
                self.emit(config.SOCKETIO_CHANNEL_2, i)
                i=i+1

    def on_subscribe(self):
        self.pubsub = redis.pubsub()
        self.spawn(self.listener)
