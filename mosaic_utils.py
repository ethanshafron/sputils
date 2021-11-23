from glob import glob
from sputils.map_utils import GeoMap
from osgeo import gdal
import subprocess

def bbox_finder(a):
    return [a[:,0].max(), a[:,1].min(), a[:,2].min(), a[:,3].max()]
#itertools.starreduce(bbox_finder, intersections)

def overlap(img1, img2):
    r1 = [img1.minx, img1.maxy, img1.maxx, img1.miny]
    r2 = [img2.minx, img2.maxy, img2.maxx, img2.miny]
    
    ### returns bounding box of intersection in minx, maxy, maxx, miny format
    ### in georeferenced coordinates
    intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]), min(r1[2], r2[2]), max(r1[3], r2[3])] 
    
    ### bounding box of mosaiced footprints between img1 and img2
    outersection = [min(r1[0], r2[0]), max(r1[1], r2[1]), max(r1[2], r2[2]), min(r1[3], r2[3])]
    return outersection
    
    left1 = int(round((intersection[0]-r1[0])/img1.resx)) # difference divided by pixel dimension
    top1 = int(round((intersection[1]-r1[1])/img1.resy))
    col1 = int(round((intersection[2]-r1[0])/img1.resx)) - left1 # difference minus offset left
    row1 = int(round((intersection[3]-r1[1])/img1.resy)) - top1
    
    left2 = int(round((intersection[0]-r2[0])/img2.resx)) # difference divided by pixel dimension
    top2 = int(round((intersection[1]-r2[1])/img2.resy))
    col2 = int(round((intersection[2]-r2[0])/img2.resx)) - left2 # difference minus new left offset
    row2 = int(round((intersection[3]-r2[1])/img2.resy)) - top2


    
    #print '\tcol1:',col1,'row1:',row1,'col2:',col2,'row2:',row2
    if col1 != col2 or row1 != row2:
        raise IndexError
    # these arrays should now have the same spatial geometry though NaNs may differ
    
    chunk1 = Chunk(img1, left1, top1, col1, row1)
    chunk2 = Chunk(img2, left2, top2, col2, row2)
    
    
#ml = list(itertools.combinations(maps, 2))
#outersections = list(itertools.starmap(overlap, ml))
#arr = np.array(outersections)
#bbox = bbox_finder(arr)
#ncols = bbox[2] - bbox[0]
#nrows = bbox[1] - bbox[3]
#vrt = gdal.GetDriverByName('VRT').Create('', ncols, nrows, 1)





def add_pixel_fn(filename: str, method : str) -> None:
    """inserts pixel-function into vrt file named 'filename'
    Args:
        filename (:obj:`string`): name of file, into which the function will be inserted
        method (str) - average,mode, min, or max. pixel function to use for overlaps.
        resample_name (:obj:`string`): name of resampling method
    """

    header = """  <VRTRasterBand dataType="UInt16" band="1" subClass="VRTDerivedRasterBand">"""
    contents = """
    <PixelFunctionType>{}</PixelFunctionType>
    <PixelFunctionLanguage>Python</PixelFunctionLanguage>
    <PixelFunctionCode><![CDATA[
import numpy as np
def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    div = np.zeros(in_ar[0].shape)
    for i in range(len(in_ar)):
        div += (in_ar[i] != 0)
    div[div == 0] = 1

    y = np.sum(in_ar, axis = 0, dtype = 'uint16')
    y = y / div
    np.clip(y,0,70000, out = out_ar)

def mode(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    mode_arr = stats.mode(in_ar, axis=0, nan_policy = 'omit')[0][0]
    np.clip(mode_arr, 0,70000, out = out_ar)

def max(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    y = np.array(in_ar).min(axis=0).astype('uint16')    
    np.clip(y,0,70000, out = out_ar)

def max(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    y = np.array(in_ar).max(axis=0).astype('uint16')    
    np.clip(y,0,70000, out = out_ar)

]]>
    </PixelFunctionCode>""".format(method)

    lines = open(filename, 'r').readlines()
    lines[3] = header
    lines.insert(4, contents)
    open(filename, 'w').write("".join(lines))

def mosaic_rasters(output, files, rm_vrt = True, method = "average"):
    out_name = output.replace(".tif", "")
    gdal.SetConfigOption('GDAL_VRT_ENABLE_PYTHON', 'YES')

    gdal.BuildVRT(f'{out_name}.vrt', files)

    add_pixel_fn(f'{out_name}.vrt', method)

    ds = gdal.Open(f'{out_name}.vrt')
    translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine("-co BIGTIFF=YES -co COMPRESS=LZW"))
    ds = gdal.Translate(output, ds, options=translateoptions)
    if rm_vrt:
        subprocess.run(["rm", f'{out_name}.vrt'])

