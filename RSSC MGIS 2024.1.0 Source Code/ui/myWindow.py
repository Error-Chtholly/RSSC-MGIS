# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'myWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1052, 644)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/ico/主图标2024.1.0.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.frame.setFont(font)
        self.frame.setStyleSheet("border-color: rgb(170, 170, 255);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout.addWidget(self.frame)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1052, 18))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.menubar.setFont(font)
        self.menubar.setStyleSheet("background-color:rgb(170, 170, 255);\n"
"selection-color: rgb(85, 0, 255);\n"
"alternate-background-color: rgb(170, 255, 255);\n"
"border-color: rgb(255, 255, 255);\n"
"font: 9pt \"幼圆\" ;\n"
"color: rgb(255, 255, 255);\n"
"selection-background-color: rgb(170, 255, 255);")
        self.menubar.setObjectName("menubar")
        self.menuOpen = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.menuOpen.setFont(font)
        self.menuOpen.setObjectName("menuOpen")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.menuEdit.setFont(font)
        self.menuEdit.setObjectName("menuEdit")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.statusbar.setFont(font)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget = QtWidgets.QDockWidget(MainWindow)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.dockWidget.setFont(font)
        self.dockWidget.setStyleSheet("border-color: rgb(170, 170, 255);")
        self.dockWidget.setObjectName("dockWidget")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.dockWidget.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.toolBar.setFont(font)
        self.toolBar.setStyleSheet("background-color:rgb(170, 255, 255);\n"
"font: 9pt \"幼圆\";\n"
"selection-background-color: rgb(170, 170, 255);\n"
"color: rgb(85, 85, 255);")
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionOpenShp = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/png/添加矢量.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpenShp.setIcon(icon1)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionOpenShp.setFont(font)
        self.actionOpenShp.setIconVisibleInMenu(True)
        self.actionOpenShp.setObjectName("actionOpenShp")
        self.actionOpenRaster = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/png/添加栅格.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpenRaster.setIcon(icon2)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionOpenRaster.setFont(font)
        self.actionOpenRaster.setObjectName("actionOpenRaster")
        self.actionEditShp = QtWidgets.QAction(MainWindow)
        self.actionEditShp.setCheckable(True)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/png/编辑.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionEditShp.setIcon(icon3)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionEditShp.setFont(font)
        self.actionEditShp.setObjectName("actionEditShp")
        self.actionSelectFeature = QtWidgets.QAction(MainWindow)
        self.actionSelectFeature.setCheckable(True)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/png/选择.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSelectFeature.setIcon(icon4)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionSelectFeature.setFont(font)
        self.actionSelectFeature.setObjectName("actionSelectFeature")
        self.actionDeleteFeature = QtWidgets.QAction(MainWindow)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/png/删除.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionDeleteFeature.setIcon(icon5)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionDeleteFeature.setFont(font)
        self.actionDeleteFeature.setObjectName("actionDeleteFeature")
        self.actionPolygon = QtWidgets.QAction(MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/png/polygon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPolygon.setIcon(icon6)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionPolygon.setFont(font)
        self.actionPolygon.setObjectName("actionPolygon")
        self.actionRectangle = QtWidgets.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/png/矩形.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRectangle.setIcon(icon7)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionRectangle.setFont(font)
        self.actionRectangle.setObjectName("actionRectangle")
        self.actionCircle = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/png/圆.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionCircle.setIcon(icon8)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionCircle.setFont(font)
        self.actionCircle.setObjectName("actionCircle")
        self.actionOpen = QtWidgets.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/png/打开工程.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen.setIcon(icon9)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionOpen.setFont(font)
        self.actionOpen.setObjectName("actionOpen")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/png/退出系统.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionQuit.setIcon(icon10)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionQuit.setFont(font)
        self.actionQuit.setObjectName("actionQuit")
        self.actionPan = QtWidgets.QAction(MainWindow)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(":/png/拖动.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPan.setIcon(icon11)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionPan.setFont(font)
        self.actionPan.setObjectName("actionPan")
        self.actionZoomIn = QtWidgets.QAction(MainWindow)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap(":/png/放大.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionZoomIn.setIcon(icon12)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionZoomIn.setFont(font)
        self.actionZoomIn.setObjectName("actionZoomIn")
        self.actionZoomOut = QtWidgets.QAction(MainWindow)
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap(":/png/缩小.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionZoomOut.setIcon(icon13)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionZoomOut.setFont(font)
        self.actionZoomOut.setObjectName("actionZoomOut")
        self.actionIdentify = QtWidgets.QAction(MainWindow)
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap(":/png/图片识别.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionIdentify.setIcon(icon14)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionIdentify.setFont(font)
        self.actionIdentify.setObjectName("actionIdentify")
        self.actionPoint = QtWidgets.QAction(MainWindow)
        icon15 = QtGui.QIcon()
        icon15.addPixmap(QtGui.QPixmap(":/png/绘制点.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPoint.setIcon(icon15)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionPoint.setFont(font)
        self.actionPoint.setObjectName("actionPoint")
        self.actionNDVI = QtWidgets.QAction(MainWindow)
        icon16 = QtGui.QIcon()
        icon16.addPixmap(QtGui.QPixmap(":/png/NDVI.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNDVI.setIcon(icon16)
        self.actionNDVI.setObjectName("actionNDVI")
        self.actionunspuervise = QtWidgets.QAction(MainWindow)
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap(":/png/MNDWI.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionunspuervise.setIcon(icon17)
        self.actionunspuervise.setObjectName("actionunspuervise")
        self.actionRandomForest = QtWidgets.QAction(MainWindow)
        icon18 = QtGui.QIcon()
        icon18.addPixmap(QtGui.QPixmap(":/png/随机森林.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRandomForest.setIcon(icon18)
        self.actionRandomForest.setObjectName("actionRandomForest")
        self.actionAboutQt = QtWidgets.QAction(MainWindow)
        icon19 = QtGui.QIcon()
        icon19.addPixmap(QtGui.QPixmap(":/png/关于Qt配置.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAboutQt.setIcon(icon19)
        self.actionAboutQt.setObjectName("actionAboutQt")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        icon20 = QtGui.QIcon()
        icon20.addPixmap(QtGui.QPixmap(":/png/主图标新720.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAbout.setIcon(icon20)
        self.actionAbout.setObjectName("actionAbout")
        self.actionHelp = QtWidgets.QAction(MainWindow)
        icon21 = QtGui.QIcon()
        icon21.addPixmap(QtGui.QPixmap(":/png/帮助.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionHelp.setIcon(icon21)
        self.actionHelp.setObjectName("actionHelp")
        self.actionDeveloper = QtWidgets.QAction(MainWindow)
        icon22 = QtGui.QIcon()
        icon22.addPixmap(QtGui.QPixmap(":/png/联系开发者.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionDeveloper.setIcon(icon22)
        self.actionDeveloper.setObjectName("actionDeveloper")
        self.actionSave = QtWidgets.QAction(MainWindow)
        icon23 = QtGui.QIcon()
        icon23.addPixmap(QtGui.QPixmap(":/png/保存.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave.setIcon(icon23)
        self.actionSave.setObjectName("actionSave")
        self.actionClip = QtWidgets.QAction(MainWindow)
        icon24 = QtGui.QIcon()
        icon24.addPixmap(QtGui.QPixmap(":/png/影像裁剪.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionClip.setIcon(icon24)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.actionClip.setFont(font)
        self.actionClip.setObjectName("actionClip")
        self.actionexportMap = QtWidgets.QAction(MainWindow)
        icon25 = QtGui.QIcon()
        icon25.addPixmap(QtGui.QPixmap(":/png/导出.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionexportMap.setIcon(icon25)
        self.actionexportMap.setObjectName("actionexportMap")
        self.actionReclass = QtWidgets.QAction(MainWindow)
        icon26 = QtGui.QIcon()
        icon26.addPixmap(QtGui.QPixmap(":/png/重分类.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionReclass.setIcon(icon26)
        self.actionReclass.setObjectName("actionReclass")
        self.actionGithub = QtWidgets.QAction(MainWindow)
        icon27 = QtGui.QIcon()
        icon27.addPixmap(QtGui.QPixmap(":/png/github.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionGithub.setIcon(icon27)
        self.actionGithub.setObjectName("actionGithub")
        self.actionLogs = QtWidgets.QAction(MainWindow)
        icon28 = QtGui.QIcon()
        icon28.addPixmap(QtGui.QPixmap(":/png/日志.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionLogs.setIcon(icon28)
        self.actionLogs.setObjectName("actionLogs")
        self.actionDocument = QtWidgets.QAction(MainWindow)
        icon29 = QtGui.QIcon()
        icon29.addPixmap(QtGui.QPixmap(":/png/开发文档.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionDocument.setIcon(icon29)
        self.actionDocument.setObjectName("actionDocument")
        self.menuOpen.addAction(self.actionOpenShp)
        self.menuOpen.addAction(self.actionOpenRaster)
        self.menuEdit.addAction(self.actionEditShp)
        self.menuEdit.addAction(self.actionSelectFeature)
        self.menuEdit.addAction(self.actionDeleteFeature)
        self.menuEdit.addSeparator()
        self.menuEdit.addAction(self.actionPoint)
        self.menuEdit.addAction(self.actionPolygon)
        self.menuEdit.addAction(self.actionRectangle)
        self.menuEdit.addAction(self.actionCircle)
        self.menu.addAction(self.actionOpen)
        self.menu.addAction(self.actionSave)
        self.menu.addAction(self.actionexportMap)
        self.menu.addAction(self.actionQuit)
        self.menu_2.addAction(self.actionPan)
        self.menu_2.addAction(self.actionZoomIn)
        self.menu_2.addAction(self.actionZoomOut)
        self.menu_2.addAction(self.actionIdentify)
        self.menu_3.addAction(self.actionClip)
        self.menu_3.addAction(self.actionNDVI)
        self.menu_3.addAction(self.actionReclass)
        self.menu_3.addAction(self.actionunspuervise)
        self.menu_3.addAction(self.actionRandomForest)
        self.menu_4.addAction(self.actionAboutQt)
        self.menu_4.addAction(self.actionAbout)
        self.menu_4.addAction(self.actionHelp)
        self.menu_4.addAction(self.actionDeveloper)
        self.menu_4.addAction(self.actionGithub)
        self.menu_4.addAction(self.actionLogs)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menuOpen.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addAction(self.actionexportMap)
        self.toolBar.addAction(self.actionQuit)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionPan)
        self.toolBar.addAction(self.actionZoomIn)
        self.toolBar.addAction(self.actionZoomOut)
        self.toolBar.addAction(self.actionIdentify)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionOpenShp)
        self.toolBar.addAction(self.actionOpenRaster)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionEditShp)
        self.toolBar.addAction(self.actionSelectFeature)
        self.toolBar.addAction(self.actionDeleteFeature)
        self.toolBar.addAction(self.actionPoint)
        self.toolBar.addAction(self.actionPolygon)
        self.toolBar.addAction(self.actionRectangle)
        self.toolBar.addAction(self.actionCircle)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionClip)
        self.toolBar.addAction(self.actionNDVI)
        self.toolBar.addAction(self.actionReclass)
        self.toolBar.addAction(self.actionunspuervise)
        self.toolBar.addAction(self.actionRandomForest)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionAboutQt)
        self.toolBar.addAction(self.actionAbout)
        self.toolBar.addAction(self.actionHelp)
        self.toolBar.addAction(self.actionDeveloper)
        self.toolBar.addAction(self.actionGithub)
        self.toolBar.addAction(self.actionLogs)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuOpen.setTitle(_translate("MainWindow", "添加数据(D)"))
        self.menuEdit.setTitle(_translate("MainWindow", "编辑要素(E)"))
        self.menu.setTitle(_translate("MainWindow", "工程文件(F)"))
        self.menu_2.setTitle(_translate("MainWindow", "地图操作(M)"))
        self.menu_3.setTitle(_translate("MainWindow", "遥感分类(C)"))
        self.menu_4.setTitle(_translate("MainWindow", "关于系统(A)"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionOpenShp.setText(_translate("MainWindow", "添加矢量数据(V)"))
        self.actionOpenShp.setIconText(_translate("MainWindow", "添加矢量数据"))
        self.actionOpenShp.setShortcut(_translate("MainWindow", "Ctrl+V"))
        self.actionOpenRaster.setText(_translate("MainWindow", "添加栅格数据(R)"))
        self.actionOpenRaster.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.actionEditShp.setText(_translate("MainWindow", "编辑矢量(S)"))
        self.actionEditShp.setShortcut(_translate("MainWindow", "Ctrl+Shift+E"))
        self.actionSelectFeature.setText(_translate("MainWindow", "选择要素(C)"))
        self.actionDeleteFeature.setText(_translate("MainWindow", "删除选中要素(D)"))
        self.actionDeleteFeature.setShortcut(_translate("MainWindow", "Ctrl+D"))
        self.actionPolygon.setText(_translate("MainWindow", "绘制面(L)"))
        self.actionPolygon.setShortcut(_translate("MainWindow", "Ctrl+L"))
        self.actionRectangle.setText(_translate("MainWindow", "绘制矩形(R)"))
        self.actionCircle.setText(_translate("MainWindow", "绘制圆(C)"))
        self.actionOpen.setText(_translate("MainWindow", "打开工程(O)"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionQuit.setText(_translate("MainWindow", "退出系统(E)"))
        self.actionQuit.setShortcut(_translate("MainWindow", "Ctrl+E"))
        self.actionPan.setText(_translate("MainWindow", "地图拖动(P)"))
        self.actionPan.setShortcut(_translate("MainWindow", "Ctrl+Shift+P"))
        self.actionZoomIn.setText(_translate("MainWindow", "地图放大(I)"))
        self.actionZoomIn.setShortcut(_translate("MainWindow", "Ctrl+Shift+I"))
        self.actionZoomOut.setText(_translate("MainWindow", "地图缩小(O)"))
        self.actionZoomOut.setShortcut(_translate("MainWindow", "Ctrl+Shift+O"))
        self.actionIdentify.setText(_translate("MainWindow", "地图识别(D)"))
        self.actionIdentify.setShortcut(_translate("MainWindow", "Ctrl+Shift+D"))
        self.actionPoint.setText(_translate("MainWindow", "绘制点(P)"))
        self.actionNDVI.setText(_translate("MainWindow", "计算NDVI(N)"))
        self.actionNDVI.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actionunspuervise.setText(_translate("MainWindow", "非监督分类(M)"))
        self.actionunspuervise.setShortcut(_translate("MainWindow", "Ctrl+M"))
        self.actionRandomForest.setText(_translate("MainWindow", "随机森林监督分类(R)"))
        self.actionRandomForest.setShortcut(_translate("MainWindow", "Ctrl+Shift+R"))
        self.actionAboutQt.setText(_translate("MainWindow", "关于Qt配置(Q)"))
        self.actionAboutQt.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionAbout.setText(_translate("MainWindow", "关于系统(S)"))
        self.actionAbout.setShortcut(_translate("MainWindow", "Ctrl+Shift+S"))
        self.actionHelp.setText(_translate("MainWindow", "系统帮助(H)"))
        self.actionHelp.setShortcut(_translate("MainWindow", "Ctrl+H"))
        self.actionDeveloper.setText(_translate("MainWindow", "联系开发者(E)"))
        self.actionDeveloper.setShortcut(_translate("MainWindow", "Ctrl+Alt+E"))
        self.actionSave.setText(_translate("MainWindow", "保存工程(S)"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionClip.setText(_translate("MainWindow", "影像裁剪(C)"))
        self.actionClip.setShortcut(_translate("MainWindow", "Ctrl+C"))
        self.actionexportMap.setText(_translate("MainWindow", "导出地图(EM)"))
        self.actionexportMap.setShortcut(_translate("MainWindow", "Alt+E"))
        self.actionReclass.setText(_translate("MainWindow", "NDVI重分类(NR)"))
        self.actionReclass.setShortcut(_translate("MainWindow", "Alt+R"))
        self.actionGithub.setText(_translate("MainWindow", "获取源码(G)"))
        self.actionGithub.setShortcut(_translate("MainWindow", "Ctrl+Shift+G"))
        self.actionLogs.setText(_translate("MainWindow", "开发日志(L)"))
        self.actionLogs.setShortcut(_translate("MainWindow", "Ctrl+Shift+L"))
        self.actionDocument.setText(_translate("MainWindow", "开发文档(D)"))
        self.actionDocument.setShortcut(_translate("MainWindow", "Alt+D"))
import myRc_rc