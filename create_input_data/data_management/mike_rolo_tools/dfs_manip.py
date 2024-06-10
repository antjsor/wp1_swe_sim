from mikecore.DfsFactory import DfsBuilder, DfsFactory  # type: ignore
from mikecore.DfsFile import DfsFile, DfsSimpleType  # type: ignore
from mikecore.DfsFileFactory import DfsFileFactory  # type: ignore
from mikecore.eum import eumQuantity, eumUnit, eumItem  # type: ignore
import numpy as np


def get_spatial_reference(filename):
    #from mikecore.DfsFileFactory import DfsFileFactory
    status=1
    dfs2File = DfsFileFactory.Dfs2FileOpen(filename)
    XCount = dfs2File.SpatialAxis.XCount
    YCount = dfs2File.SpatialAxis.YCount
    xysize = XCount * YCount
    Dx = dfs2File.SpatialAxis.Dx #5
    Dy = dfs2File.SpatialAxis.Dy #5
    X0 = dfs2File.SpatialAxis.X0 #0
    Y0 = dfs2File.SpatialAxis.Y0 #0
    Lat = dfs2File.FileInfo.Projection.Latitude #-37.89519109801612
    Lon = dfs2File.FileInfo.Projection.Longitude #145.1455131881479
    Or = dfs2File.FileInfo.Projection.Orientation #1.13931185349487
    WKT = dfs2File.FileInfo.Projection.WKTString
    dfs2File.Close()
    return [status,XCount,YCount,xysize,Dx,Dy,X0,Y0,Lat,Lon,Or,WKT]

def read_dfs2_timestep(filename, index, timestep):
    status=1
    dfs2File = DfsFileFactory.Dfs2FileOpen(filename)
    dfsdata = dfs2File.ReadItemTimeStep(index, timestep) #item no, time step
    dfs2File.Close()
    return [status, dfsdata]
    
def dfsdata_to_numpy1D(dfsdata,XCount,YCount):
    #take a data item from dfs2 file and convert it to a 1D-numpy array which in this form can be written back to dfs
    status=1
    #dfsdata_np=numpy.array(list(dfsdata.Data) )#convert the dfs data to numpy array for easier processing
    dfsdata_np=dfsdata.Data
    #data are now column wise in the 1D-array (where bottom is the first and top the last value))
    dfsdata_np2=np.reshape(dfsdata_np,(XCount,YCount),order='C')
    dfsdata_np3=np.reshape(dfsdata_np2,XCount*YCount, order='F')
    return [status, dfsdata_np3]
    
def dfsdata_to_numpy2D(dfsdata,XCount,YCount):
    #take a data item from dfs2 file and convert it to a 2D-numpy array which can be converted to ArcGIS raster
    status=1
    #tmp=numpy.array(list(dfsdata.Data))
    tmp=dfsdata.Data
    tmp=np.reshape(tmp,(XCount,YCount),order='F')
    tmp=np.rot90(tmp, k=1)
    return [status, tmp]


