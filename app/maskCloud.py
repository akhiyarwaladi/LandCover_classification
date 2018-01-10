import arcpy as ap
import os

class masker:
    '''Provides access to functions that produces masks from remote sensing image, according to its bit structure.'''

    def __init__(self, band, *var):
        self.bandarray = band

    def getmask(self, bitpos, bitlen, value, cummulative):
        '''Generates mask with given bit information.
        Parameters
            bitpos      -   Position of the specific QA bits in the value string.
            bitlen      -   Length of the specific QA bits.
            value       -   A value indicating the desired condition.
        '''
        lenstr = ''
        for i in range(bitlen):
            lenstr += '1'
        bitlen = int(lenstr, 2)

        if type(value) == unicode:
            value = int(value, 2)

        posValue = bitlen << bitpos
        conValue = value << bitpos

        if cummulative:
            mask = (self.bandarray & posValue) >= conValue
        else:
            mask = (self.bandarray & posValue) == conValue

        return mask.astype(int)
        
def mask_cloud(path, masktype, confidence, cummulative, out):
    convalue = {'High' : 3, 'Medium' : 2, 'Low' : 1, 'None' : 0}
    maskvalue = {'Cloud' : 14, 'Cirrus' : 12, 'Snow' : 10, 'Vegetation' : 8, 'Water' : 4}

    ap.env.workspace = out
    output = out+'/mask_cloud_'+str(os.path.basename(path))+'.TIF'
    ap.AddMessage('Making mask cloud using BQA')
    raster = ap.Raster(path + '/' + str(os.path.basename(path))+'_BQA.TIF')
    rasterarray = ap.RasterToNumPyArray(raster)
    bitmasker = masker(rasterarray)
    outarray = bitmasker.getmask(maskvalue[masktype], 2, convalue[confidence], cummulative)

    outraster = ap.NumPyArrayToRaster(outarray, 
                                         ap.Point(raster.extent.XMin, raster.extent.YMin),
                                         raster,
                                         raster,
                                         raster.noDataValue)

    outraster.save(output)
    ap.AddMessage('Finished Making mask cloud using BQA saved in: ' + str(output))