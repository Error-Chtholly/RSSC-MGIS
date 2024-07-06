# -*- coding: utf-8 -*-
# @Author  : Error Cohtholly
# @Time    : 2024/7/1
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import QSplashScreen


class NewSplashScreen(QSplashScreen):
    def __init__(self):
        super(NewSplashScreen, self).__init__()
        self.setPixmap(QPixmap('./resources/启动画面.png'))

    def mousePressEvent(self, event):
        pass