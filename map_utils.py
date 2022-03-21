
from osgeo import gdal
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import sys
import argparse
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from statsmodels.api import MixedLM, OLS, GLM
from subprocess import Popen, PIPE
import statsmodels.formula.api as smf
import pandas as pd

def gdal_getpixel(f, x, y):
    cmd = ['gdallocationinfo', '-valonly', f, str(x), str(y)]
    p = Popen(cmd, stdout=PIPE)
    val = p.stdout.read()
    return np.array([float(i) for i in val.decode("utf-8").split("\n") if len(i) > 0])

def random_sample_from_map(data, n=1000, ndv=None, in_memory=True, verbose=True):
    if not ndv:
        ndv = data.ndv
    n_sampled = 0
    if not in_memory:
        X = np.ones(shape=(n,data.n_features))*-1 
        xoffs = np.random.randint(0, data.cols, size = n)
        yoffs = np.random.randint(0, data.rows, size = n)
        while n_sampled < n:
            for i, (xoff, yoff) in enumerate(zip(xoffs, yoffs)):
                pixarr = gdal_getpixel(f, xoff, yoff)            
                while np.in1d(ndv, pixarr):
                    try:
                        xoffs[i] = np.random.randint(1, data.cols)
                        yoffs[i] = np.random.randint(1, data.rows)
                        pixarr = gdal_getpixel(f, xoffs[i], yoffs[i])  
                    except ValueError:
                        xoffs[i] = np.random.randint(1, data.cols)
                        yoffs[i] = np.random.randint(1, data.rows)
                        pixarr = gdal_getpixel(f, xoffs[i], yoffs[i])  
                    if verbose:
                        print("Sample {} contained ndv".format(i))
                else:
                    if verbose:
                        print("Got {} of {}".format(n_sampled+1, n))
                    X[i,] = pixarr
                    n_sampled += 1
    else:
        X = np.ones(shape=(n,data.n_features))*-1 
        xoffs = np.random.randint(0, data.cols, size = n)
        yoffs = np.random.randint(0, data.rows, size = n)
        dat = data.data.ReadAsArray()
        while n_sampled < n:
            for i, (xoff, yoff) in enumerate(zip(xoffs, yoffs)):
                if len(dat.shape)<3:
                    pixarr = np.squeeze(dat[yoff,xoff])
                else:
                    pixarr = np.squeeze(dat[:,yoff,xoff])
                while np.in1d(ndv, pixarr):
                    try:
                        xoffs[i] = np.random.randint(1, data.cols)
                        yoffs[i] = np.random.randint(1, data.rows)
                        if len(dat.shape)<3:
                            pixarr = np.squeeze(dat[yoffs[i],xoffs[i]])
                        else:
                            pixarr = np.squeeze(dat[:,yoffs[i],xoffs[i]])
                    except ValueError:
                        xoffs[i] = np.random.randint(1, data.cols)
                        yoffs[i] = np.random.randint(1, data.rows)
                        if len(dat.shape)<3:
                            pixarr = np.squeeze(dat[yoffs[i],xoffs[i]])
                        else:
                            pixarr = np.squeeze(dat[:,yoffs[i],xoffs[i]])
                    if verbose:
                        print("Sample {} contained ndv".format(i))
                else:
                    if verbose:
                        print("Got {} of {}".format(n_sampled+1, n))
                    X[i,] = pixarr
                    n_sampled += 1
    return X.astype(float)

def band2arrary(dataset, band):
    return dataset.GetRasterBand(band).ReadAsArray()

class GeoMap:    
    def __init__(self, filename, band_names=None):
        self.name = filename
        self.data = gdal.Open(self.name)
        self.rows = self.data.RasterYSize
        self.cols = self.data.RasterXSize
        self.proj = self.data.GetProjection()
        self.n_features = self.data.RasterCount
        self._gt = self.data.GetGeoTransform()
        self.resx = self._gt[1]
        self.resy = self._gt[5]
        self.minx = self._gt[0]
        self.maxy = self._gt[3]
        self.maxx = self.minx + self.resx * self.cols
        self.miny = self.maxy + self.resy * self.rows
        self.ndv = self.data.GetRasterBand(1).GetNoDataValue()
        self.dtype = self.data.GetRasterBand(1).DataType
        if len(self.data.GetRasterBand(1).GetDescription()) > 0:
            self.band_names = {self.data.GetRasterBand(b).GetDescription():b for b in range(1, self.n_features+1)}
        else:
            self.band_names = {b:b for b in range(1,self.n_features+1)}

        self.driver = self.data.GetDriver().ShortName     
    
    def duplicate(self, outname, band_nums, of=None):
        """
        Duplicate a raster by its geotransform and projection info

        args:
            outname (str) - output file name
            band_nums (int) - output number of bands
            of (str) - output format
        """
        
        if not of:
            of = self.driver
        driver = gdal.GetDriverByName(of) 
        driver.Register()
        outDataset = driver.Create(outname,
                                   self.cols,
                                   self.rows,
                                   band_nums,
                                   self.dtype)
        outDataset.SetProjection(self.proj)
        outDataset.SetGeoTransform(self._gt)
        return outDataset
    
    def to_table(self, xoff=0, yoff=0, xsize=None, ysize=None, bands=None):
        """
        Turn a raster (or part of a raster) into a table.
        Helper method for going from array of shape (bands, cols, rows)
        to (rows*cols, bands).
        """
        
        if not bands:
            bands = self.n_features
        if not xsize:
            xsize = self.cols
        if not ysize:
            ysize = self.rows
        dat = np.squeeze(self.data.ReadAsArray(xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize, band_list=bands))
        table = dat.reshape((ysize*xsize, len(bands)))
        return table

    def fit_model(self, method, Y_band=None, X_bands=[], formula=None, groups=None, sample_size=None, **kwargs):
        if sample_size:
            data = random_sample_from_map(self, n=sample_size, verbose=False)
        else:
            data = self.data.ReadAsArray().reshape(self.rows*self.cols, self.n_features)
        df = pd.DataFrame(data, columns = [i for i in self.band_names])
        methods = {"GBM_c":GradientBoostingClassifier,
                   "GBM_r":GradientBoostingRegressor,
                   "NN_c":MLPClassifier,
                   "NN_r":MLPRegressor,
                   "RF_c":RandomForestClassifier,
                   "RF_r":RandomForestRegressor,
                   "LME": smf.mixedlm,
                   "OLS": smf.ols,
                   "GLM": smf.glm}
        if formula:
            if groups:
                model = methods[method](formula, data = df, groups=df["{}".format(groups)], **kwargs).fit()
            else:
                model = methods[method](formula, data = df, **kwargs).fit()
        else:
            X = np.array(df.loc[:, X_bands])
            Y = np.array(df.loc[:,[Y_band]]).ravel()
            model = methods[method](**kwargs).fit(X, Y)
        return model
    
    def make_row_offsets(self, rowsize = 100):
        offsets = []
        natural_breaks = self.rows // rowsize
        leftovers = self.rows % rowsize
        max_natural_row = natural_breaks*rowsize
        for i, offset in enumerate(range(0, max_natural_row+rowsize, rowsize)):
            print(i, offset)
            if i+1 <= natural_breaks:
                offsets.append((offset, rowsize))
            elif offset + leftovers == self.rows:
                offsets.append((max_natural_row, leftovers))
        return offsets 
    def make_data_table_chunks(self, **kwargs):
        offsets = self.make_row_offsets(**kwargs)
        for yoff, ysize in offsets:
            yield Chunk(self, xoff=0, yoff=yoff, xsize=None, ysize=ysize)
class Chunk:
    def __init__(self, geomap, xoff, yoff, xsize, ysize):
        self.xoff = xoff
        self.yoff = yoff
        self.xsize = xsize
        self.ysize = ysize
        self.data = geomap.data
        self.minx = geomap.minx + (self.xoff*geomap.resx)
        self.miny = geomap.miny + (self.yoff*geomap.resy*-1)
        self.maxx = self.minx + (self.xsize*geomap.resx) 
        self.maxy = self.miny + (self.ysize*geomap.resy*-1) 
    def read(self, band):
        rasterband = self.data.GetRasterBand(band)
        return rasterband.ReadAsArray(xoff=self.xoff, yoff=self.yoff, win_xsize=self.xsize, win_ysize=self.ysize)
    
    def write(self, dataset, data, band):
        band = dataset.GetRasterBand(band)
        band.WriteArray(data, self.xoff, self.yoff)
        


if __name__ == "__main__":
    f = '/home/es182091e/libtest.tif'
    m = GeoMap(f)
    cs = m.make_data_table_chunks() 
    for c in cs:
        print(c.yoff)
