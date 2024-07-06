from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QDialog, QFileDialog

from ui.NDVI_RECLASS import Ui_ReClass_NDVI



class ReClass_NDVI(QDialog,Ui_ReClass_NDVI):
    def __init__(self, parent=None):
        super(ReClass_NDVI, self).__init__(parent)
        self.setupUi(self)
        self.NdviPath = ""
        self.OutputPath = ""
        self.functions()

    def functions(self):
        self.chooseNDVI.clicked.connect(self.openNDVI)
        self.choosePath.clicked.connect(self.outputPath)
        self.ok.clicked.connect(self.accept)
        self.quit.clicked.connect(self.close)


    def openNDVI(self):
        rasterFilepath,_=QFileDialog.getOpenFileName(
            self,
            "Open Raster File",
            ".",
            "Raster Files (*.tif)"
        )
        if rasterFilepath:
            self.txtNDVIPATH.setText(rasterFilepath)
            self.NdviPath = rasterFilepath

    def outputPath(self):
        # 使用 QFileDialog.getExistingDirectory 静态方法打开文件夹选择对话框
        # 第一个参数是父窗口（可以是 None），第二个参数是初始目录（可以是 QDir.homePath() 或其他路径）
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", QDir.homePath())

        # 检查用户是否取消了选择
        if folder_path:
            self.txtPATH.setText(folder_path)
            self.OutputPath = folder_path