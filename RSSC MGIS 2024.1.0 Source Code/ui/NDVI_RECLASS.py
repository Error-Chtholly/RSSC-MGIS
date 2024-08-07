# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'NDVI_RECLASS.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ReClass_NDVI(object):
    def setupUi(self, ReClass_NDVI):
        ReClass_NDVI.setObjectName("ReClass_NDVI")
        ReClass_NDVI.resize(573, 210)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/ico/重分类.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        ReClass_NDVI.setWindowIcon(icon)
        ReClass_NDVI.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.txtNDVIPATH = QtWidgets.QTextEdit(ReClass_NDVI)
        self.txtNDVIPATH.setGeometry(QtCore.QRect(110, 20, 331, 31))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.txtNDVIPATH.setFont(font)
        self.txtNDVIPATH.setObjectName("txtNDVIPATH")
        self.label = QtWidgets.QLabel(ReClass_NDVI)
        self.label.setGeometry(QtCore.QRect(30, 20, 61, 31))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(ReClass_NDVI)
        self.label_2.setGeometry(QtCore.QRect(30, 90, 61, 31))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.txtPATH = QtWidgets.QTextEdit(ReClass_NDVI)
        self.txtPATH.setGeometry(QtCore.QRect(110, 90, 331, 31))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.txtPATH.setFont(font)
        self.txtPATH.setObjectName("txtPATH")
        self.ok = QtWidgets.QPushButton(ReClass_NDVI)
        self.ok.setGeometry(QtCore.QRect(370, 160, 71, 31))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.ok.setFont(font)
        self.ok.setObjectName("ok")
        self.quit = QtWidgets.QPushButton(ReClass_NDVI)
        self.quit.setGeometry(QtCore.QRect(470, 160, 71, 31))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.quit.setFont(font)
        self.quit.setObjectName("quit")
        self.chooseNDVI = QtWidgets.QPushButton(ReClass_NDVI)
        self.chooseNDVI.setGeometry(QtCore.QRect(460, 20, 81, 31))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.chooseNDVI.setFont(font)
        self.chooseNDVI.setObjectName("chooseNDVI")
        self.choosePath = QtWidgets.QPushButton(ReClass_NDVI)
        self.choosePath.setGeometry(QtCore.QRect(460, 90, 81, 31))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.choosePath.setFont(font)
        self.choosePath.setObjectName("choosePath")
        self.label_3 = QtWidgets.QLabel(ReClass_NDVI)
        self.label_3.setGeometry(QtCore.QRect(110, 60, 431, 16))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(ReClass_NDVI)
        self.label_4.setGeometry(QtCore.QRect(110, 130, 431, 16))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_2.raise_()
        self.label.raise_()
        self.txtNDVIPATH.raise_()
        self.txtPATH.raise_()
        self.ok.raise_()
        self.quit.raise_()
        self.chooseNDVI.raise_()
        self.choosePath.raise_()
        self.label_3.raise_()
        self.label_4.raise_()

        self.retranslateUi(ReClass_NDVI)
        QtCore.QMetaObject.connectSlotsByName(ReClass_NDVI)

    def retranslateUi(self, ReClass_NDVI):
        _translate = QtCore.QCoreApplication.translate
        ReClass_NDVI.setWindowTitle(_translate("ReClass_NDVI", "NDVI分类"))
        self.label.setText(_translate("ReClass_NDVI", "NDVI数据："))
        self.label_2.setText(_translate("ReClass_NDVI", "输出路径："))
        self.ok.setText(_translate("ReClass_NDVI", "NDVI分类"))
        self.quit.setText(_translate("ReClass_NDVI", "关  闭"))
        self.chooseNDVI.setText(_translate("ReClass_NDVI", "选择NDVI文件"))
        self.choosePath.setText(_translate("ReClass_NDVI", "选择目录"))
        self.label_3.setText(_translate("ReClass_NDVI", "注意：请选择提取好的NDVI数据，支持.tif格式。"))
        self.label_4.setText(_translate("ReClass_NDVI", "注意：请选择分类结果输出目录，文件会以classified_ndvi.tif后缀命名。"))
import nrRc_rc
