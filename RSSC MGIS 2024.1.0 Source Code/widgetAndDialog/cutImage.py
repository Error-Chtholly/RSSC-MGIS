from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QDialog, QFileDialog
from ui.ImageCutUI import Ui_CutImageDialog


class CutDialog(QDialog, Ui_CutImageDialog):
    def __init__(self, parent=None):
        super(CutDialog, self).__init__(parent)
        self.setupUi(self)
        self.rasterPath=""
        self.vectorPath=""
        self.outputPath=""
        self.functions()


    def functions(self):
        self.pbtnOpenRaster.clicked.connect(self.openRaster)
        self.pbtnOpenVector.clicked.connect(self.openVector)
        self.pbtnChooseLocation.clicked.connect(self.select_folder)
        self.pbtnCut.clicked.connect(self.accept)
        self.pbtnClose.clicked.connect(self.close)


    def openRaster(self):
        rasterFilePath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Raster File",
            ".",
            "Raster Files (*.tif)"
        )

        if rasterFilePath:
            self.txtRasterPath.setText(rasterFilePath)
            self.rasterPath=rasterFilePath

    def openVector(self):
        FilePath, _ = QFileDialog.getOpenFileName(
            self,
            "Open shapefile",
            ".",
            "Raster Files (*.shp)"
        )

        if FilePath:
            self.txtVectorPath.setText(FilePath)
            self.vectorPath=FilePath

    def select_folder(self):
        # 使用 QFileDialog.getExistingDirectory 静态方法打开文件夹选择对话框
        # 第一个参数是父窗口（可以是 None），第二个参数是初始目录（可以是 QDir.homePath() 或其他路径）
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", QDir.homePath())

        # 检查用户是否取消了选择
        if folder_path:
            self.txtoutputPath.setText(folder_path)
            self.outputPath=folder_path