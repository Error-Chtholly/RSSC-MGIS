import os
import sys
import traceback
import webbrowser
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from PyQt5.QtGui import QPainter, QPixmap, QColor
from osgeo import gdal
from qgis._core import QgsCoordinateReferenceSystem, QgsMapSettings, QgsVectorLayer, QgsMapLayer, QgsRasterLayer, \
    QgsColorRampShader, QgsRasterShader, QgsSingleBandPseudoColorRenderer
from qgis._gui import QgsMapToolIdentifyFeature, QgsMapToolPan, QgsMapToolZoom
from qgis.core import QgsProject, QgsLayerTreeModel, QgsMapLayerType
from qgis.gui import QgsLayerTreeView,QgsMapCanvas,QgsLayerTreeMapCanvasBridge
from PyQt5.QtCore import QUrl, QSize, QMimeData, QUrl, Qt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from qgisUtils.mapTool import PointMapTool
from ui.myWindow import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QStatusBar, QLabel, \
    QComboBox
from qgisUtils import addMapLayer, readVectorFile, readRasterFile, menuProvider, PolygonMapTool
from widgetAndDialog.ReClass import ReClass_NDVI
from widgetAndDialog.cutImage import CutDialog
from widgetAndDialog.ndviAnalyst import NdviDialog
from widgetAndDialog.randomForest import RandomForestDialog
from widgetAndDialog.unsupervise import Unsupervised

PROJECT = QgsProject.instance()

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.NDVI = NdviDialog(self)
        self.NDVI_RECLASS = ReClass_NDVI(self)
        self.UNSUPERVISE = Unsupervised(self)
        self.randomForestDialog=RandomForestDialog(self)
        self.cutDialog=CutDialog(self)
        self.setupUi(self)
        # 1 修改标题
        self.setWindowTitle("RSSC MGIS 2024.1.0")
        # 2 初始化图层树
        vl = QVBoxLayout(self.dockWidgetContents)
        self.layerTreeView = QgsLayerTreeView(self)
        vl.addWidget(self.layerTreeView)
        # 3 初始化地图画布
        self.mapCanvas = QgsMapCanvas(self)
        hl = QHBoxLayout(self.frame)
        hl.setContentsMargins(0,0,0,0) #设置周围间距
        hl.addWidget(self.mapCanvas)
        # 4 设置图层树风格
        self.model = QgsLayerTreeModel(PROJECT.layerTreeRoot(),self)
        self.model.setFlag(QgsLayerTreeModel.AllowNodeRename) #允许图层节点重命名
        self.model.setFlag(QgsLayerTreeModel.AllowNodeReorder) #允许图层拖拽排序
        self.model.setFlag(QgsLayerTreeModel.AllowNodeChangeVisibility) #允许改变图层节点可视性
        self.model.setFlag(QgsLayerTreeModel.ShowLegendAsTree) #展示图例
        self.model.setAutoCollapseLegendNodes(10) #当节点数大于等于10时自动折叠
        self.layerTreeView.setModel(self.model)
        # 4 建立图层树与地图画布的桥接
        self.layerTreeBridge = QgsLayerTreeMapCanvasBridge(PROJECT.layerTreeRoot(),self.mapCanvas,self)
        # 5 初始加载影像
        self.firstAdd = True
        # 6 允许拖拽文件
        self.setAcceptDrops(True)
        # 7 图层树右键菜单创建
        self.rightMenuProv = menuProvider(self)
        self.layerTreeView.setMenuProvider(self.rightMenuProv)
        # 8.0 提前给予基本CRS
        self.mapCanvas.setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:4490"))
        # 8 状态栏控件
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet('color: rgb(255, 255, 255); border: rgb(255, 255, 255); background-color:rgb(170, 170, 255); font: 9pt "幼圆" ;')
        self.statusXY = QLabel('{:<40}'.format(''))  # x y 坐标状态
        self.statusBar.addWidget(self.statusXY, 1)
        self.statusScaleLabel = QLabel('比例尺')
        self.statusScaleComboBox = QComboBox(self)
        self.statusScaleComboBox.setFixedWidth(120)
        self.statusScaleComboBox.addItems(
            ["1:500", "1:1000", "1:2500", "1:5000", "1:10000", "1:25000", "1:100000", "1:500000", "1:1000000"])
        self.statusScaleComboBox.setEditable(True)
        self.statusBar.addWidget(self.statusScaleLabel)
        self.statusBar.addWidget(self.statusScaleComboBox)
        self.statusCrsLabel = QLabel(
            f"坐标系: {self.mapCanvas.mapSettings().destinationCrs().description()}-{self.mapCanvas.mapSettings().destinationCrs().authid()}")
        self.statusBar.addWidget(self.statusCrsLabel)
        self.setStatusBar(self.statusBar)
        # 9 error catch
        self.old_hook = sys.excepthook
        sys.excepthook = self.catch_exceptions
        # 打开工程
        self.actionOpen.triggered.connect(self.actionOpenTriggered)
        #保存工程
        self.actionSave.triggered.connect(self.actionSaveTriggered)
        # 退出程序
        self.actionQuit.triggered.connect(self.close)
        # 地图工具
        self.actionPanTriggered()
        self.actionPan.triggered.connect(self.actionPanTriggered)
        self.actionZoomIn.triggered.connect(self.actionZoomInTriggered)
        self.actionZoomOut.triggered.connect(self.actionZoomOutTriggered)
        self.actionIdentify.triggered.connect(self.actionIdentifyTriggered)
        self.actionPoint.triggered.connect(self.actionPointTriggered)
        # 图像裁剪
        self.cutDialog.accepted.connect(self.updateCutResult)
        self.randomForestDialog.accepted.connect(self.imageRF)
        # 导出地图
        self.actionexportMap.triggered.connect(self.export_map_view_to_image)
        # A 按钮、菜单栏功能
        self.connectFunc()
        # B 初始设置控件
        self.actionEditShp.setEnabled(False)
        self.editTempLayer: QgsVectorLayer = None  # 初始编辑图层为None
        # 关于Qt
        self.actionAboutQt.triggered.connect(lambda: QMessageBox.aboutQt(self, '关于Qt配置'))
        self.actionAbout.triggered.connect(lambda: QMessageBox.about(self, '关于系统', 'RSSC MGIS\nVersion : 2024.1.0.\nInformation : A Geographic Information System for conducting relevant remote sensing classification algorithms.\n\nLearn more : This system is open source, you can get the code of it from the website "https://github.com/Error-Chtholly".\n\nDesigned by : Error Chtholly(周泽同), Normal Lee(李普通) and Akizuki Kanna39(罗祥澄).\nFriendly performance : Bee(?马蜂).\n\nLast released at 2024.07.05.\n\nAll rights reserved by the development team of RSSC MGIS.\n\nHope you can enjoy every beautiful day and cherish the life in this world.\n                                                     ------by developer Error Chtholly'))
        self.actionDeveloper.triggered.connect(lambda: QMessageBox.about(self, '联系开发者','Hello! I am Error Chtholly, the developer of the RSSC MGIS!\n\nMy Email : 1696304992@qq.com(QQ Email)\nzzt15563168887@outlook.com(Outlook Email)\n\nQQ Number : 1696304992\n\nWeixin Number : zzt15563168887\n\nMy Github : https://github.com/Error-Chtholly\n\nWelcome if you have any questions or suggestions!'))
        self.actionHelp.triggered.connect(lambda: QMessageBox.about(self, '系统帮助','欢迎使用RSSC MGIS系统，本系统基于QGIS开发，QGIS是一个开源的跨平台地理信息系统，它可以用来处理各种地理数据，包括矢量数据、栅格数据、栅格影像、地形数据、地理编码数据等。\n\n工程文件 : 打开保存文档或退出系统。\n地图操作 : 地图的放大缩小拖动等操作。\n添加数据 : 添加矢量和栅格数据。\n编辑要素 : 编辑和创建矢量要素。\n遥感分类 : 进行NDVI等算法对遥感图像的分类。\n关于系统 : 系统开发信息和系统帮助。\n状态栏 : 更改比例尺，查看坐标系。\n图层右键菜单 : 对选中图层或组的相关操作。\n地图右键菜单 : 获取当前点位的坐标数据。\nCtrl+O : 打开工程。\nCtrl+S : 保存工程。\nAlt+E : 导出地图。\nCtrl+E : 退出系统。\nCtrl+Shift+P : 地图拖动。\nCtrl+Shift+I : 地图放大。\nCtrl+Shift+O : 地图缩小。\nCtrl+Shift+D : 地图识别。\nCtrl+V : 添加矢量。\nCtrl+R : 添加栅格。\nCtrl+Shift+E : 启用编辑。\nCtrl+D : 删除选中要素。\nCtrl+N : 计算NDVI.\nCtrl+M : 非监督分类.\nCtrl+Shift+R : 随机森林监督分类。\nCtrl+C : 影像裁剪。\nAlt+R : NDVI重分类。\nCtrl+Q : 查看Qt配置。\nCtrl+Shift+S : 查看系统信息。\nCtrl+H : 查看系统帮助。\nCtrl+Alt+E : 查看开发者联系方式。\nCtrl+Shift+G : 打开Github获取源代码。\nCtrl+Shift+L : 查看开发日志。\n\n感谢您对本系统的支持！'))
        self.actionLogs.triggered.connect(lambda: QMessageBox.about(self, '开发日志','Version Beta : \n体验版本，实现了一些基础GIS功能。\n\nVersion 2024.1.0 : \n正式发布版本，实现了遥感图像识别分类等高级分析功能，优化了系统界面设计和子系统界面设计，增加了基本功能。'))
        # 打开Github
        self.actionGithub.triggered.connect(self.openGithub)

    # 打开Github
    def openGithub(self):
        webbrowser.open('https://github.com/Error-Chtholly')

    # 计算NDVI值并显示
    def updateNdviResult(self):
        redBand = rio.open(self.NDVI.redBandPath)
        nirBand = rio.open(self.NDVI.nirBandpath)
        red = redBand.read(1).astype("float64")
        nir = nirBand.read(1).astype("float64")
        ndvi = np.where(
            (nir + red) == 0., 0,
            (nir - red) / (nir + red))
        kwargs = redBand.meta
        kwargs.update(
            dtype=rio.float32,
            count=1,
            compress='lzw'
        )
        output = self.NDVI.outputPath + "/ndvi_result.tif"
        with rio.open(output, 'w', **kwargs) as dst:
            dst.write_band(1, ndvi.astype(rio.float32))
        if output:
            rasterLayer = readRasterFile(output)
            if rasterLayer.isValid():
                if self.firstAdd:
                    addMapLayer(rasterLayer, self.mapCanvas, True)
                    self.firstAdd = False
                else:
                    addMapLayer(rasterLayer, self.mapCanvas)

    # 重分类
    def reclass(self):
        def classify_ndvi(ndvi):
            classes = np.empty_like(ndvi, dtype=int)
            classes[ndvi < 0.1] = 1
            classes[(ndvi >= 0.1) & (ndvi < 0.4)] = 2
            classes[(ndvi >= 0.4) & (ndvi < 0.7)] = 3
            classes[ndvi >= 0.7] = 4
            return classes

        with rio.open(self.NDVI_RECLASS.NdviPath) as src:
            ndvi_array = src.read(1)
            profile = src.profile
            classified_array = classify_ndvi(ndvi_array)
            profile.update(
                dtype=rio.uint8,
                count=1
            )
            metadata = {
                'TIFFTAG_DOCUMENTNAME': 'Classified NDVI',
                'CLASSIFICATION': '1=LOW Vegetation,2=Sparse Vegetation,3=Moderate Vegetation,4=Hige Vegetation'
            }
            with rio.open(self.NDVI_RECLASS.OutputPath + "/classified_ndvi.tif", mode='w', **profile,
                        metadata=metadata) as dst:
                dst.write(classified_array.astype(rio.uint8), 1)
                layer_path = self.NDVI_RECLASS.OutputPath + "/classified_ndvi.tif"
                layer = QgsRasterLayer(layer_path, 'Classified Raster')
                if not layer.isValid():
                    print("Layer failed to load!")
                else:
                    colorRampShader = QgsColorRampShader()
                    colorRampShader.setColorRampItemList([
                        QgsColorRampShader.ColorRampItem(0, QColor(128, 128, 0)),  # 红色
                        QgsColorRampShader.ColorRampItem(1, QColor(135, 255, 0)),  # 绿色
                        QgsColorRampShader.ColorRampItem(2, QColor(0, 184, 255)),  # 蓝色
                    ])
                    rasterShader = QgsRasterShader()
                    rasterShader.setRasterShaderFunction(colorRampShader)
                    renderer = QgsSingleBandPseudoColorRenderer(
                        layer.dataProvider(),
                        1,
                        rasterShader
                    )
                    layer.setRenderer(renderer)
                    QgsProject.instance().addMapLayer(layer)
                output = self.NDVI_RECLASS.OutputPath + "/classified_ndvi.tif"
            if output:
                rasterLayer = readRasterFile(output)
                if rasterLayer.isValid():
                    if self.firstAdd:
                        addMapLayer(rasterLayer, self.mapCanvas, True)
                        self.firstAdd = False
                    else:
                        addMapLayer(rasterLayer, self.mapCanvas)

        # 非监督分类
    def imageunsupervise(self):
        rasterFilePath = self.UNSUPERVISE.reclassPath
        with rio.open(rasterFilePath) as src:
            band = src.read(1)
        band_normalized = (band - band.min()) / (band.max() - band.min())
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=0).fit(band_normalized.reshape(-1, 1))
        labels = kmeans.labels_
        labels_2d = labels.reshape(band.shape)
        with rio.open(self.UNSUPERVISE.outputPath + "/UnsupervisedResult.tif", mode='w', driver='GTiff',
                        height=src.height, width=src.width, count=1, dtype=np.uint8,
                        crs=src.crs, transform=src.transform) as dst:
            dst.write(labels_2d.astype(np.uint8), 1)
            layer_path = self.UNSUPERVISE.outputPath + "/UnsupervisedResult.tif"
            layer = QgsRasterLayer(layer_path, 'Classified Raster')
            if not layer.isValid():
                print("Layer failed to load!")
            else:
                colorRampShader = QgsColorRampShader()
                colorRampShader.setColorRampItemList([
                    QgsColorRampShader.ColorRampItem(0, QColor(128, 128, 0)),  # 红色
                    QgsColorRampShader.ColorRampItem(1, QColor(198, 60, 0)),  # 黄色
                    QgsColorRampShader.ColorRampItem(2, QColor(135, 255, 0)),  # 绿色
                    QgsColorRampShader.ColorRampItem(3, QColor(0, 184, 255)),  # 蓝色
                    QgsColorRampShader.ColorRampItem(4, QColor(255, 99, 123))  # 品红色
                ])
                rasterShader = QgsRasterShader()
                rasterShader.setRasterShaderFunction(colorRampShader)
                renderer = QgsSingleBandPseudoColorRenderer(
                    layer.dataProvider(),
                    1,
                    rasterShader
                )
                layer.setRenderer(renderer)
                QgsProject.instance().addMapLayer(layer)

    def showNdviWindows(self):
        self.NDVI.exec_()

    def showNdvi_reclassWindows(self):
        self.NDVI_RECLASS.exec_()

    def showUnsuperviseWindows(self):
        self.UNSUPERVISE.exec_()

    def imageRF(self):
        #dataset_list = [rio.open(f'./data/band{i}.tif') for i in range(2, 8)]
        dataset_list = [rio.open(self.randomForestDialog.rasterPath+f'/band{i}.tif') for i in range(2, 8)]
        calgary_trainingpointer_gpd = gpd.read_file(self.randomForestDialog.vectorPath)
        all_read_vector = np.concatenate([dataset_list[i].read() for i in range(len(dataset_list))], axis=0)

        def location2value(x, y):
            row, col = dataset_list[0].index(x, y)
            res = all_read_vector[:, row, col]
            return pd.Series(res)

        trainX = calgary_trainingpointer_gpd.to_crs(dataset_list[0].crs.to_string()).pipe(
            lambda x: x.assign(**{'lon': x.geometry.x, 'lat': x.geometry.y})).pipe(
            lambda x: x.apply(lambda x: location2value(x['lon'], x['lat']), axis=1))
        trainY = calgary_trainingpointer_gpd['class']
        X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, train_size=0.8, random_state=42)
        rf_fit = RandomForestClassifier()  # SVC()
        rf_fit.fit(X_train, y_train)
        predict_all_x = np.hstack([dataset_list[i].read().reshape(-1, 1) for i in range(len(dataset_list))])
        predict_all_result = rf_fit.predict(predict_all_x)
        class_list = np.unique(predict_all_result).tolist()
        class_dict = {value: index + 1 for index, value in enumerate(class_list)}
        result = pd.DataFrame({'class': predict_all_result})['class'].map(class_dict).values.reshape(
            dataset_list[0].read().shape[1:])
        result_mask = result.copy().astype(np.float64)
        prof = dataset_list[0].profile

        # 更新 profile 中的数据类型以匹配分类结果
        prof.update(
            dtype=rio.uint8,  # 假设我们使⽤ uint8 存储分类结果
            count=1  # 分类结果是⼀个波段
        )
        metadata = {
            'TIFFTAG_DOCUMENTNAME': 'Classified NDVI',
            'CLASSIFICATION': '1=water, 2=forest, 3=land, 4=building'
        }
        # 写⼊新的栅格⽂件
        with rio.open(self.randomForestDialog.outputPath+'/rf_result.tif', 'w', **prof, metadata=metadata) as dst:
            dst.write(result_mask.astype(rio.uint8), 1)
            # 栅格⽂件路径
        layer_path = self.randomForestDialog.outputPath+'/rf_result.tif'
        # 加载栅格图层
        layer = QgsRasterLayer(layer_path, 'Classified Raster')
        # 如果需要，将图层添加到QGIS项⽬中
        QgsProject.instance().addMapLayer(layer)

    def export_map_view_to_image(self):
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出地图",
            ".",
            "PNG图片(*.png)"
        )
        if output_path:
            # 获取当前项目的画布
            canvas=self.mapCanvas
            # 获取画布的范围
            extent=canvas.extent()
            # 设置输出的分辨率
            dpi=300

            map_settings=QgsMapSettings()
            map_settings.setExtent(extent)
            map_settings.setOutputSize(canvas.size())
            map_settings.setOutputDpi(dpi)

            image=QPixmap(canvas.size())
            painter=QPainter(image)
            canvas.render(painter)
            image.save(output_path,"png",dpi)

    def updateCutResult(self):
        layer=QgsRasterLayer(self.cutDialog.rasterPath,os.path.basename(self.cutDialog.rasterPath))
        layervector=QgsVectorLayer(self.cutDialog.vectorPath,os.path.basename(self.cutDialog.vectorPath))
        addMapLayer(layer,self.mapCanvas)
        addMapLayer(layervector,self.mapCanvas)
        path=Path(self.cutDialog.rasterPath)
        fname=path.stem
        outpath=self.cutDialog.outputPath+"/"+fname+"_cut.tif"
        OutTile=gdal.Warp(outpath,self.cutDialog.rasterPath,cutlineDSName=self.cutDialog.vectorPath,cropToCutline=True,dstNodata=0)
        OutTile=None
        if outpath:
            rasterLayer=readRasterFile(outpath)
            if rasterLayer.isValid():
                if self.firstAdd:
                    addMapLayer(rasterLayer,self.mapCanvas,True)
                    self.firstAdd=False
                else:
                    addMapLayer(rasterLayer,self.mapCanvas)

    def connectFunc(self):
        # 按钮、菜单栏功能
        self.actionOpenRaster.triggered.connect(self.actionOpenRasterTriggered)
        self.actionOpenShp.triggered.connect(self.actionOpenShpTriggered)

        #每次移动鼠标，坐标和比例尺变化
        self.mapCanvas.destinationCrsChanged.connect(self.showCrs)
        self.mapCanvas.xyCoordinates.connect(self.showXY)
        self.mapCanvas.scaleChanged.connect(self.showScale)
        self.statusScaleComboBox.editTextChanged.connect(self.changeScaleForString)

        # 单击、双击图层 触发事件
        self.layerTreeView.clicked.connect(self.layerClicked)
        self.actionEditShp.triggered.connect(self.actionEditShpTriggered)
        self.actionSelectFeature.triggered.connect(self.actionSelectFeatureTriggered)
        self.actionDeleteFeature.triggered.connect(self.actionDeleteFeatureTriggered)
        self.actionPolygon.triggered.connect(self.actionPolygonTriggered)

        # 遥感影像分类
        self.actionClip.triggered.connect(self.showCutImage)
        self.actionRandomForest.triggered.connect(self.showRandomForest)
        self.actionNDVI.triggered.connect(self.showNdviWindows)
        self.NDVI.accepted.connect(self.updateNdviResult)
        self.actionReclass.triggered.connect(self.showNdvi_reclassWindows)
        self.NDVI_RECLASS.accepted.connect(self.reclass)
        self.actionunspuervise.triggered.connect(self.showUnsuperviseWindows)
        self.UNSUPERVISE.accepted.connect(self.imageunsupervise)

    def showRandomForest(self):
        self.randomForestDialog.exec_()

    def showCutImage(self):
        self.cutDialog.exec_()

    def changeScaleForString(self, str):
        try:
            left, right = str.split(":")[0], str.split(":")[-1]
            if int(left) == 1 and int(right) > 0 and int(right) != int(self.mapCanvas.scale()):
                self.mapCanvas.zoomScale(int(right))
                self.mapCanvas.zoomWithCenter()
        except:
            print(traceback.format_stack())

    def showScale(self, scale):
        self.statusScaleComboBox.setEditText(f"1:{int(scale)}")

    def showXY(self, point):
        x = point.x()
        y = point.y()
        self.statusXY.setText(f'{x:.6f}, {y:.6f}')

    def showCrs(self):
        mapSetting: QgsMapSettings = self.mapCanvas.mapSettings()
        self.statusCrsLabel.setText(
            f"坐标系: {mapSetting.destinationCrs().description()}-{mapSetting.destinationCrs().authid()}")

    def dragEnterEvent(self, fileData):
        if fileData.mimeData().hasUrls():
            fileData.accept()
        else:
            fileData.ignore()

    # 拖拽文件事件
    def dropEvent(self,fileData):
        mimeData: QMimeData = fileData.mimeData()
        filePathList = [u.path()[1:] for u in mimeData.urls()]
        for filePath in filePathList:
            filePath:str = filePath.replace("/","//")
            if filePath.split(".")[-1] in ["tif","TIF","tiff","TIFF","GTIFF","png","jpg","pdf"]:
                self.addRasterLayer(filePath)
            elif filePath.split(".")[-1] in ["shp","SHP","gpkg","geojson","kml"]:
                self.addVectorLayer(filePath)
            elif filePath == "":
                pass
            else:
                QMessageBox.about(self, '警告', f'{filePath}为不支持的文件类型，目前支持栅格影像和shp矢量')

    def catch_exceptions(self, ty, value, trace):
        """
            捕获异常，并弹窗显示
        :param ty: 异常的类型
        :param value: 异常的对象
        :param traceback: 异常的traceback
        """
        traceback_format = traceback.format_exception(ty, value, trace)
        traceback_string = "".join(traceback_format)
        QMessageBox.about(self, 'error', traceback_string)
        self.old_hook(ty, value, trace)

    def actionOpenRasterTriggered(self):
        data_file, ext = QFileDialog.getOpenFileName(self, '添加栅格', '','GeoTiff(*.tif;*tiff;*TIF;*TIFF);;All Files(*);;JPEG(*.jpg;*.jpeg;*.JPG;*.JPEG);;*.png;;*.pdf')
        if data_file:
            self.addRasterLayer(data_file)

    def actionOpenShpTriggered(self):
        data_file, ext = QFileDialog.getOpenFileName(self, '添加矢量', '',"ShapeFile(*.shp);;All Files(*);;Other(*.gpkg;*.geojson;*.kml)")
        if data_file:
            self.addVectorLayer(data_file)

    def addRasterLayer(self,rasterFilePath):
        rasterLayer = readRasterFile(rasterFilePath)
        if self.firstAdd:
            addMapLayer(rasterLayer,self.mapCanvas,True)
            self.firstAdd = False
        else:
            addMapLayer(rasterLayer,self.mapCanvas)

    def addVectorLayer(self, vectorFilePath):
        vectorLayer = readVectorFile(vectorFilePath)
        if self.firstAdd:
            addMapLayer(vectorLayer, self.mapCanvas, True)
            self.firstAdd = False
        else:
            addMapLayer(vectorLayer, self.mapCanvas)

    def layerClicked(self):
        curLayer: QgsMapLayer = self.layerTreeView.currentLayer()
        if curLayer and type(curLayer) == QgsVectorLayer and not curLayer.readOnly():
            self.actionEditShp.setEnabled(True)
        else:
            self.actionEditShp.setEnabled(False)

    def actionEditShpTriggered(self):
        if self.actionEditShp.isChecked():
            self.editTempLayer: QgsVectorLayer = self.layerTreeView.currentLayer()
            self.editTempLayer.startEditing()
        else:
            saveShpEdit = QMessageBox.question(self, '保存编辑', "确定要将编辑内容保存到内存吗？",
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if saveShpEdit == QMessageBox.Yes:
                self.editTempLayer.commitChanges()
            else:
                self.editTempLayer.rollBack()

            self.mapCanvas.refresh()
            self.editTempLayer = None

    def selectToolIdentified(self, feature):
        print(feature.id())
        layerTemp: QgsVectorLayer = self.layerTreeView.currentLayer()
        if layerTemp.type() == QgsMapLayerType.VectorLayer:
            if feature.id() in layerTemp.selectedFeatureIds():
                layerTemp.deselect(feature.id())
            else:
                layerTemp.removeSelection()
                layerTemp.select(feature.id())

    def actionSelectFeatureTriggered(self):
        if self.actionSelectFeature.isChecked():
            if self.mapCanvas.mapTool():
                self.mapCanvas.unsetMapTool(self.mapCanvas.mapTool())
            self.selectTool = QgsMapToolIdentifyFeature(self.mapCanvas)
            self.selectTool.setCursor(Qt.ArrowCursor)
            self.selectTool.featureIdentified.connect(self.selectToolIdentified)
            layers = self.mapCanvas.layers()
            if layers:
                self.selectTool.setLayer(self.layerTreeView.currentLayer())
            self.mapCanvas.setMapTool(self.selectTool)
        else:
            if self.mapCanvas.mapTool():
                self.mapCanvas.unsetMapTool(self.mapCanvas.mapTool())

    def actionDeleteFeatureTriggered(self):
        if self.editTempLayer == None:
            QMessageBox.information(self, '警告', '您没有编辑中矢量')
            return
        if len(self.editTempLayer.selectedFeatureIds()) == 0:
            QMessageBox.information(self, '删除选中矢量', '您没有选择任何矢量')
        else:
            self.editTempLayer.deleteSelectedFeatures()

    def actionPolygonTriggered(self):
        if self.editTempLayer == None:
            QMessageBox.information(self, '警告', '您没有编辑中矢量')
            return
        if self.mapCanvas.mapTool():
            self.mapCanvas.mapTool().deactivate()
        self.polygonTool = PolygonMapTool(self.mapCanvas, self.editTempLayer, self)
        self.mapCanvas.setMapTool(self.polygonTool)

    def actionOpenTriggered(self):
        """打开工程"""
        data_file, ext = QFileDialog.getOpenFileName(self, '打开工程', '', '工程文件(*.qgs , *.qgz)')
        if data_file:
            #PROJECT.instance().readProject(data_file)
            PROJECT.read(data_file)

    def actionSaveTriggered(self):
        """保存工程"""
        data_file, ext = QFileDialog.getSaveFileName(self, '保存工程', '', '工程文件(*.qgs , *.qgz)')
        if data_file:
            PROJECT.write(data_file)

    def actionPanTriggered(self):
        self.mapTool = QgsMapToolPan(self.mapCanvas)
        self.mapCanvas.setMapTool(self.mapTool)

    def actionZoomInTriggered(self):
        self.mapTool = QgsMapToolZoom(self.mapCanvas, False)
        self.mapCanvas.setMapTool(self.mapTool)

    def actionZoomOutTriggered(self):
        self.mapTool = QgsMapToolZoom(self.mapCanvas, True)
        self.mapCanvas.setMapTool(self.mapTool)

    def actionIdentifyTriggered(self):
        # 设置识别工具
        self.identifyTool = QgsMapToolIdentifyFeature(self.mapCanvas)
        self.identifyTool.featureIdentified.connect(self.showFeatures)
        self.mapCanvas.setMapTool(self.identifyTool)

    def showFeatures(self, feature):
        print(type(feature))

        QMessageBox.information(self, '信息', ''.join(feature.attributes()))

    def actionPointTriggered(self):
        self.pointTool = PointMapTool(self.mapCanvas)
        self.mapCanvas.setMapTool(self.pointTool)