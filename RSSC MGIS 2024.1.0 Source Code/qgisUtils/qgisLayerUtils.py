from qgis._core import QgsRectangle, QgsCoordinateReferenceSystem, QgsVectorDataProvider, QgsWkbTypes, \
    QgsRasterDataProvider
from qgis.core import QgsMapLayer,QgsRasterLayer,QgsVectorLayer,QgsProject
from qgis.gui import QgsMapCanvas
import os
import os.path as osp

from qgisUtils.file import getFileSize

PROJECT = QgsProject.instance()

def addMapLayer(layer:QgsMapLayer,mapCanvas:QgsMapCanvas,firstAddLayer=False):
    if layer.isValid():
        if firstAddLayer:
            mapCanvas.setDestinationCrs(layer.crs())
            mapCanvas.setExtent(layer.extent())

        while(PROJECT.mapLayersByName(layer.name())):
            layer.setName(layer.name()+"_1")

        PROJECT.addMapLayer(layer)
        layers = [layer] + [PROJECT.mapLayer(i) for i in PROJECT.mapLayers()]
        mapCanvas.setLayers(layers)
        mapCanvas.refresh()

def readRasterFile(rasterFilePath):
    rasterLayer = QgsRasterLayer(rasterFilePath,osp.basename(rasterFilePath))
    return rasterLayer

def readVectorFile(vectorFilePath):
    vectorLayer = QgsVectorLayer(vectorFilePath,osp.basename(vectorFilePath),"ogr")
    return vectorLayer

def getRasterLayerAttrs(rasterLayer:QgsRasterLayer):
    print("name: ", rasterLayer.name()) # 图层名
    print("type: ", rasterLayer.type()) # 栅格还是矢量图层
    print("height - width: ", rasterLayer.height(),rasterLayer.width()) #尺寸
    print("bands: ", rasterLayer.bandCount()) #波段数
    print("extent", rasterLayer.extent()) #外接矩形范围
    print("source", rasterLayer.source()) #图层的源文件地址
    print("crs", rasterLayer.crs())  # 图层的坐标系统

def getVectorLayerAttrs(vectorLayer:QgsVectorLayer):
    print("name: ", vectorLayer.name())  # 图层名
    print("type: ", vectorLayer.type())  # 栅格还是矢量图层
    print("extent", vectorLayer.extent())  # 外接矩形范围
    print("source", vectorLayer.source())  # 图层的源文件地址
    print("crs", vectorLayer.crs())  # 图层的坐标系统

def getVectorLayerAttrs(vectorLayer:QgsVectorLayer):
    vdp : QgsVectorDataProvider = vectorLayer.dataProvider()
    crs: QgsCoordinateReferenceSystem = vectorLayer.crs()
    extent: QgsRectangle = vectorLayer.extent()
    resDict = {
        "name" : vectorLayer.name(),
        "source" : vectorLayer.source(),
        "memory": getFileSize(vectorLayer.source()),
        "extent" : f"min:[{extent.xMinimum():.6f},{extent.yMinimum():.6f}]; max:[{extent.xMaximum():.6f},{extent.yMaximum():.6f}]",
        "geoType" : QgsWkbTypes.geometryDisplayString(vectorLayer.geometryType()),
        "featureNum" : f"{vectorLayer.featureCount()}",
        "encoding" : vdp.encoding(),
        "crs" : crs.description(),
        "dpSource" : vdp.description()
    }
    return resDict

qgisDataTypeDict = {
    0 : "UnknownDataType",
    1 : "Uint8",
    2 : "UInt16",
    3 : "Int16",
    4 : "UInt32",
    5 : "Int32",
    6 : "Float32",
    7 : "Float64",
    8 : "CInt16",
    9 : "CInt32",
    10 : "CFloat32",
    11 : "CFloat64",
    12 : "ARGB32",
    13 : "ARGB32_Premultiplied"
}

def getRasterLayerAttrs(rasterLayer:QgsRasterLayer):
    rdp : QgsRasterDataProvider = rasterLayer.dataProvider()
    crs : QgsCoordinateReferenceSystem = rasterLayer.crs()
    extent: QgsRectangle = rasterLayer.extent()
    resDict = {
        "name" : rasterLayer.name(),
        "source" : rasterLayer.source(),
        "memory" : getFileSize(rasterLayer.source()),
        "extent" : f"min:[{extent.xMinimum():.6f},{extent.yMinimum():.6f}]; max:[{extent.xMaximum():.6f},{extent.yMaximum():.6f}]",
        "width" : f"{rasterLayer.width()}",
        "height" : f"{rasterLayer.height()}",
        "dataType" : qgisDataTypeDict[rdp.dataType(1)],
        "bands" : f"{rasterLayer.bandCount()}",
        "crs" : crs.description()
    }
    return resDict