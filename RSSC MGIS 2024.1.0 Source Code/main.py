from qgis.PyQt import QtCore
from qgis.core import QgsApplication
from PyQt5.QtCore import Qt
import os
import traceback
from config import setup_env
from mainWindow import MainWindow
from splash import NewSplashScreen

if __name__ == '__main__':
    setup_env()
    # 适应高分辨率
    QgsApplication.setPrefixPath('qgis', True)
    QgsApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QgsApplication.setStyle("Fusion")
    app = QgsApplication([], True)

    t = QtCore.QTranslator()
    t.load(r'.\zh-Hans.qm')
    app.installTranslator(t)

    # 启动画面
    splash = NewSplashScreen()
    splash.show()
    app.initQgis()

    mainWindow = MainWindow()
    splash.finish(mainWindow)
    mainWindow.setWindowState(Qt.WindowMaximized) #设置窗口初始最大化
    mainWindow.show()
    #shp = r"D:\111.shp"
    #tif = r"D:\test.tif"
    #mainWindow.addVectorLayer(shp)
    #mainWindow.addRasterLayer(tif)

    app.exec_()
    app.exitQgis()