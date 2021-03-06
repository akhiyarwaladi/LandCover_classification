from ftplib import FTP
import os
from datetime import datetime
import shutil
def downloadFile():

	tupDate = datetime.now()
	print tupDate.year
	print tupDate.strftime('%j')
	#print now.year, now.month, now.day, now.hour, now.minute, now.second # check every datetime detail
	ftp = FTP( )
	ftp.connect(host='localhost', port=21, timeout=1246)
	ftp.login(user='akhiyarwaladi', passwd='rickss12')
	#ftp.retrlines('LIST') # use to check file after connected
	ftp.cwd('Landsat_8')
	#ftp.cwd(str(tupDate.year))
	ftp.cwd('2017')
	#ftp.cwd(str(tupDate.strftime('%j')))
	ftp.cwd('291')
	for level in ftp.nlst():
		print level
		ftp.cwd(str(level))
		for scene in ftp.nlst():
			print scene
			if(os.path.exists('C:/data/lahan/input/'+ scene)):
				shutil.rmtree('C:/data/lahan/input/'+ scene)
				#break;
			os.makedirs('C:/data/lahan/input/'+ scene)
			ftp.cwd(scene)
			filesPreFlood = ftp.nlst()

			for file in filesPreFlood:
				print file
				msg = str(datetime.now()) + '\t' + "Downloading "+ str(file) + "\n"
				filename = file #replace with your file in the directory ('directory_name')
				localfile = open(filename, 'wb')
				if(os.path.exists('C:/data/lahan/input/'+ scene +'/'+filename)):
					os.remove('C:/data/lahan/input/'+ scene +'/'+filename)

				ftp.retrbinary('RETR ' + filename, localfile.write, 1024)
				localfile.close()


				os.rename(filename,'C:/data/lahan/input/'+ scene +'/'+filename)

			ftp.cwd("../")
		ftp.cwd("../")

	os.chdir('C:/data/lahan/input/'+ scene)

	for filename in os.listdir('C:/data/lahan/input/'+ scene):
		productID = filename.split(".")[0]
		extension = filename.split(".")[1]
		unique = filename.split(".")[0].split("_")[7]

		os.rename(filename, scene + "_" + unique + "." + extension)
		# print productID
		# print extension
		# print unique

	return scene

# if __name__ == '__main__':
	
# 	downloadFile()