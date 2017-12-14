
import time

from celery import Celery
from flask import Flask, render_template, request, flash
from redis import StrictRedis
from socketio import socketio_manage
from socketio.namespace import BaseNamespace

from assets import assets
import config
import celeryconfig

redis = StrictRedis(host=config.REDIS_HOST)
redis.delete(config.MESSAGES_KEY)
# celery = Celery(__name__)
# celery.config_from_object(celeryconfig)

app = Flask(__name__)
app.config.from_object(config)
assets.init_app(app)

app.config['SECRET_KEY'] = 'top-secret!'
app.config['SOCKETIO_CHANNEL'] = 'tail-message'
app.config['MESSAGES_KEY'] = 'tail'
app.config['CHANNEL_NAME'] = 'tail-channel'
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if redis.llen(config.MESSAGES_KEY):
            flash('Task is already running', 'error')
        else:
            tail.delay()
            flash('Task started', 'info')
    return render_template('index.html')


@app.route('/socket.io/<path:remaining>')
def socketio(remaining):
    socketio_manage(request.environ, {
        '/tail': TailNamespace
    })
    return app.response_class()


@celery.task
def tail():
    
    # for i in range(0, 20):
    #     msg = 'Task message %s\n' % i
    #     redis.rpush(config.MESSAGES_KEY, msg)
    #     redis.publish(config.CHANNEL_NAME, msg)
    #     time.sleep(1)
    # redis.delete(config.MESSAGES_KEY)
    import arcpy
    import pandas as pd
    import numpy as np
    dataPath = "C:/Apps/data_banghendrik/Combinasi_654_Jabo_Lapan_modified.tif"
    modelPath = "C:/Apps/data_banghendrik/DataTest_decisionTree.pkl"
    outputPath = "C:/Prog/banghendrik/Combinasi_654_Jabo_Lapan_modified_clf.tif"
    rasterarray = arcpy.RasterToNumPyArray(dataPath)

    data = np.array([rasterarray[0].ravel(), rasterarray[1].ravel(), rasterarray[2].ravel()])
    data = data.transpose()

    import pandas as pd
    print("Change to dataframe format")

    msg = "Change to dataframe format \n"
    redis.rpush(config.MESSAGES_KEY, msg)
    redis.publish(config.CHANNEL_NAME, msg)
    #time.sleep(1)

    columns = ['band1','band2', 'band3']
    df = pd.DataFrame(data, columns=columns)

    print("Split data to 20 chunks ")
    msg = "Split data to 20 chunks \n"
    redis.rpush(config.MESSAGES_KEY, msg)
    redis.publish(config.CHANNEL_NAME, msg)
    #time.sleep(1)

    df_arr = np.array_split(df, 20)
    from sklearn.externals import joblib
    clf = joblib.load(modelPath) 
    kelasAll = []
    for i in range(len(df_arr)):
        
        print ("predicting data chunk-%s\n" % i)
        msg = "predicting data chunk-%s\n" % i
        redis.rpush(config.MESSAGES_KEY, msg)
        redis.publish(config.CHANNEL_NAME, msg)
        #time.sleep(1)
        kelas = clf.predict(df_arr[i])
        dat = pd.DataFrame()
        dat['kel'] = kelas
        print ("mapping to integer class")
        msg = "mapping to integer class \n"
        redis.rpush(config.MESSAGES_KEY, msg)
        redis.publish(config.CHANNEL_NAME, msg)
        #time.sleep(1)
        mymap = {'awan':1, 'air':2, 'tanah':3, 'vegetasi':4}
        dat['kel'] = dat['kel'].map(mymap)

        band1Array = dat['kel'].values
        print ("extend to list")
        msg = "extend to list \n"
        redis.rpush(config.MESSAGES_KEY, msg)
        redis.publish(config.CHANNEL_NAME, msg)
        #time.sleep(1)

        kelasAll.extend(band1Array.tolist())
    redis.delete(config.MESSAGES_KEY)
    # del df_arr
    # del clf
    # del kelas
    # del dat
    # del band1Array
    # del data

    # print ("change list to np array")
    # logging.info("change list to np array")
    # kelasAllArray = np.array(kelasAll, dtype=np.uint8)

    # print ("reshaping np array")
    # logging.info("reshaping np array")
    # band1 = np.reshape(kelasAllArray, (-1, rasterarray[0][0].size))
    # band1 = band1.astype(np.uint8)

    # raster = arcpy.Raster(dataPath)
    # inputRaster = dataPath

    # spatialref = arcpy.Describe(inputRaster).spatialReference
    # cellsize1  = raster.meanCellHeight
    # cellsize2  = raster.meanCellWidth
    # extent     = arcpy.Describe(inputRaster).Extent
    # pnt        = arcpy.Point(extent.XMin,extent.YMin)

    # del raster

    # # save the raster
    # print ("numpy array to raster ..")
    # logging.info("numpy array to raster ..")
    # out_ras = arcpy.NumPyArrayToRaster(band1, pnt, cellsize1, cellsize2)
    # #out_ras.save(outputPath)
    # #arcpy.CheckOutExtension("Spatial")
    # print ("define projection ..")
    # logging.info ("define projection ..")
    # arcpy.CopyRaster_management(out_ras, outputPath)
    # arcpy.DefineProjection_management(outputPath, spatialref)


    # redis.delete(config.MESSAGES_KEY)
class TailNamespace(BaseNamespace):
    def listener(self):
        # Emit the backlog of messages
        messages = redis.lrange(config.MESSAGES_KEY, 0, -1)
        self.emit(config.SOCKETIO_CHANNEL, ''.join(messages))

        self.pubsub.subscribe(config.CHANNEL_NAME)

        for m in self.pubsub.listen():
            if m['type'] == 'message':
                self.emit(config.SOCKETIO_CHANNEL, m['data'])

    def on_subscribe(self):
        self.pubsub = redis.pubsub()
        self.spawn(self.listener)
