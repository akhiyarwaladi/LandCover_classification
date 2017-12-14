
import time
from datetime import datetime
from celery import Celery
from flask import Flask, render_template, request, flash
from redis import StrictRedis
from socketio import socketio_manage
from socketio.namespace import BaseNamespace
from celery.task.control import revoke
import urllib2
from assets import assets
import config
import celeryconfig
import os

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
                flash('Task started', 'info')
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
    
    # for i in range(0, 20):
    #     msg = 'Task message %s\n' % i
    #     redis.rpush(config.MESSAGES_KEY, msg)
    #     redis.publish(config.CHANNEL_NAME, msg)
    #     time.sleep(1)
    # redis.delete(config.MESSAGES_KEY)

    msg = str(datetime.now()) + '\t' + "Importing Library ... \n"
    redis.rpush(config.MESSAGES_KEY, msg)
    redis.publish(config.CHANNEL_NAME, msg)

    import arcpy
    import pandas as pd
    import numpy as np
    import os.path
    import ftpClient as ft

    stat = ft.downloadFile()
    arrfile = os.listdir(config.dataPath)
    print(arrfile)
    for filename in arrfile:
        dataPath =  config.dataPath + filename
        modelPath = config.modelPath
        outputPath = config.outputPath + filename
        if(os.path.exists(outputPath)):
            os.remove(outputPath)
        rasterarray = arcpy.RasterToNumPyArray(dataPath)

        msg = str(datetime.now()) + '\t' + "Processing file "+filename+"\n"
        redis.rpush(config.MESSAGES_KEY, msg)
        redis.publish(config.CHANNEL_NAME, msg)

        data = np.array([rasterarray[0].ravel(), rasterarray[1].ravel(), rasterarray[2].ravel()])
        data = data.transpose()

        import pandas as pd
        print("Change to dataframe format")

        msg = str(datetime.now()) + '\t' + "Change to dataframe format \n"
        redis.rpush(config.MESSAGES_KEY, msg)
        redis.publish(config.CHANNEL_NAME, msg)
        #time.sleep(1)

        columns = ['band1','band2', 'band3']
        df = pd.DataFrame(data, columns=columns)

        print("Split data to 20 chunks ")
        msg = str(datetime.now()) + '\t' + "Split data to 20 chunks \n"
        redis.rpush(config.MESSAGES_KEY, msg)
        redis.publish(config.CHANNEL_NAME, msg)
        #time.sleep(1)

        df_arr = np.array_split(df, 20)
        from sklearn.externals import joblib
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
            mymap = {'awan':1, 'air':2, 'tanah':3, 'vegetasi':4}
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

        band1 = np.reshape(kelasAllArray, (-1, rasterarray[0][0].size))
        band1 = band1.astype(np.uint8)

        raster = arcpy.Raster(dataPath)
        inputRaster = dataPath

        spatialref = arcpy.Describe(inputRaster).spatialReference
        cellsize1  = raster.meanCellHeight
        cellsize2  = raster.meanCellWidth
        extent     = arcpy.Describe(inputRaster).Extent
        pnt        = arcpy.Point(extent.XMin,extent.YMin)

        del raster

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

        arcpy.CopyRaster_management(out_ras, outputPath)
        arcpy.DefineProjection_management(outputPath, spatialref)

        msg = str(datetime.now()) + '\t' + "Finished ... \n"
        redis.rpush(config.MESSAGES_KEY, msg)
        redis.publish(config.CHANNEL_NAME, msg)

        redis.delete(config.MESSAGES_KEY)
        redis.delete(config.MESSAGES_KEY_2)


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
        i=11
        for m in self.pubsub.listen():
            if m['type'] == 'message':
                self.emit(config.SOCKETIO_CHANNEL, m['data'])
                self.emit(config.SOCKETIO_CHANNEL_2, i)
                i=i+1

    def on_subscribe(self):
        self.pubsub = redis.pubsub()
        self.spawn(self.listener)
