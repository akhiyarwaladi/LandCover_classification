from ftplib import FTP
import os
from redis import StrictRedis
import config
from datetime import datetime

def uploadFile():
	filename = 'Combinasi_654_Jabo_Lapan_modified.tif' #replace with your file in your home folder
	ftp.storbinary('STOR '+filename, open(filename, 'rb'))
	ftp.quit()

def downloadFile():
	redis = StrictRedis(host=config.REDIS_HOST)
	msg = str(datetime.now()) + '\t' + "Connecting to ftp server \n"
	redis.rpush(config.MESSAGES_KEY, msg)
	redis.publish(config.CHANNEL_NAME, msg)
	ftp = FTP( )
	ftp.connect(host='localhost', port=21, timeout=1246)
	ftp.login(user='lapan', passwd='lapan2017')
	ftp.retrlines('LIST')
	msg = str(datetime.now()) + '\t' + "Downloading data ... \n"
	redis.rpush(config.MESSAGES_KEY, msg)
	redis.publish(config.CHANNEL_NAME, msg)
	filenames = ftp.nlst()
	for filename in filenames:    
		localfile = open(filename, 'wb')
		if(os.path.exists('C:/Apps/data/tes/'+filename)):
			os.remove('C:/Apps/data/tes/'+filename)
		ftp.retrbinary('RETR ' + filename, localfile.write, 1024)
		localfile.close()
		msg = str(datetime.now()) + '\t' + "Moving data to directory " + str('C:/Apps/data/tes/'+filename) +'\n'
		redis.rpush(config.MESSAGES_KEY, msg)
		redis.publish(config.CHANNEL_NAME, msg)
		os.rename(filename,'C:/Apps/data/tes/'+filename)
	ftp.quit()
	

	
	return "sukses"

#uploadFile()
#downloadFile()