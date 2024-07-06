
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QDialog, QFileDialog

from ui.ndviUI import Ui_ndviDialog


class NdviDialog(QDialog,Ui_ndviDialog):

    def __init__(self,parent=None):
        super(NdviDialog,self).__init__(parent)
        self.setupUi(self)
        self.redBandPath = ""
        self.nirBandpath = ""
        self.outputPath = ""
        self.functions()

    def functions(self):
        self.redOpen.clicked.connect(self.openRedRaster)
        self.nirredOpen.clicked.connect(self.openNirRaster)
        self.pathOpen.clicked.connect(self.select_folder)
        self.ok.clicked.connect(self.accept)
        self.quit.clicked.connect(self.close)

    def openRedRaster(self):
        rasterFilePath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Raster File",
            ".",
            "Raster Files (*.tif)"
        )

        if rasterFilePath:
            self.txtRedband.setText(rasterFilePath)
            self.redBandPath = rasterFilePath

    def openNirRaster(self):
        rasterFilePath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Raster File",
            ".",
            "Raster Files (*.tif)"
        )

        if rasterFilePath:
            self.txtNirRed.setText(rasterFilePath)
            self.nirBandpath = rasterFilePath

    def select_folder(self):
        # 使用 QFileDialog.getExistingDirectory 静态方法打开文件夹选择对话框
        # 第一个参数是父窗口（可以是 None），第二个参数是初始目录（可以是 QDir.homePath() 或其他路径）
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", QDir.homePath())

        # 检查用户是否取消了选择
        if folder_path:
            self.txtPath.setText(folder_path)
            self.outputPath = folder_path