import os
import os.path as osp
from osgeo import gdal
import traceback
from shutil import copyfile
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QMenu, QAction, QFileDialog, QMessageBox, QTableView, QDialog
from qgis.core import QgsLayerTreeNode, QgsLayerTree, QgsMapLayerType, QgsVectorLayer, QgsProject \
    , QgsVectorFileWriter, QgsWkbTypes, Qgis, QgsFillSymbol, QgsSingleSymbolRenderer, QgsVectorLayerCache \
    , QgsMapLayer, QgsRasterLayer, QgsLayerTreeGroup, QgsLayerTreeLayer
from qgis.gui import QgsLayerTreeViewMenuProvider, QgsLayerTreeView, QgsLayerTreeViewDefaultActions, QgsMapCanvas, \
    QgsMessageBar, \
    QgsAttributeTableModel, QgsAttributeTableView, QgsAttributeTableFilterModel, QgsGui, QgsAttributeDialog, \
    QgsProjectionSelectionDialog, QgsMultiBandColorRendererWidget
import traceback
from widgetAndDialog import LayerPropWindowWidgeter, AttributeDialog

PROJECT = QgsProject.instance()

class menuProvider(QgsLayerTreeViewMenuProvider):
    def __init__(self, mainWindow, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layerTreeView: QgsLayerTreeView = mainWindow.layerTreeView
        self.mapCanvas: QgsMapCanvas = mainWindow.mapCanvas
        self.mainWindows = mainWindow

    def createContextMenu(self) -> QtWidgets.QMenu:
        try:
            menu = QMenu()
            self.actions: QgsLayerTreeViewDefaultActions = self.layerTreeView.defaultActions()
            if not self.layerTreeView.currentIndex().isValid():
                # 清除图层 deleteAllLayer
                actionDeleteAllLayer = QAction('清除图层', menu)
                actionDeleteAllLayer.triggered.connect(lambda: self.deleteAllLayer())
                menu.addAction(actionDeleteAllLayer)

                menu.addAction('展开所有图层', self.layerTreeView.expandAllNodes)
                menu.addAction('折叠所有图层', self.layerTreeView.collapseAllNodes)
                return menu


            if len(self.layerTreeView.selectedLayers()) > 1:
                # 添加组
                self.actionGroupSelected = self.actions.actionGroupSelected()
                menu.addAction(self.actionGroupSelected)

                self.actionZoomToLayer = self.actions.actionZoomToLayer(self.mapCanvas, menu)
                menu.addAction(self.actionZoomToLayer)

                self.actionMoveToTop = self.actions.actionMoveToTop(menu)
                menu.addAction(self.actionMoveToTop)

                actionDeleteSelectedLayers = QAction('删除选中图层(D)', menu)
                actionDeleteSelectedLayers.triggered.connect(self.deleteSelectedLayer)
                menu.addAction(actionDeleteSelectedLayers)

                return menu

            node: QgsLayerTreeNode = self.layerTreeView.currentNode()
            if node:
                if QgsLayerTree.isGroup(node):
                    group: QgsLayerTreeGroup = self.layerTreeView.currentGroupNode()
                    self.actionRenameGroup = self.actions.actionRenameGroupOrLayer(menu)
                    menu.addAction(self.actionRenameGroup)

                    self.actionMoveToTop = self.actions.actionMoveToTop(menu)
                    menu.addAction(self.actionMoveToTop)

                    actionDeleteGroup = QAction('删除组(D)', menu)
                    actionDeleteGroup.triggered.connect(lambda: self.deleteGroup(group))
                    menu.addAction(actionDeleteGroup)
                elif QgsLayerTree.isLayer(node):
                    layer: QgsMapLayer = self.layerTreeView.currentLayer()

                    if layer.type() == QgsMapLayerType.VectorLayer:
                        actionOpenAttributeDialog = QAction('打开属性表', menu)
                        actionOpenAttributeDialog.triggered.connect(lambda: self.openAttributeDialog(layer))
                        menu.addAction(actionOpenAttributeDialog)

                    self.actionZoomToLayer = self.actions.actionZoomToLayer(self.mapCanvas, menu)
                    menu.addAction(self.actionZoomToLayer)

                    # 添加组
                    self.actionGroupSelected = self.actions.actionGroupSelected()
                    menu.addAction(self.actionGroupSelected)

                    self.actionMoveToTop = self.actions.actionMoveToTop(menu)
                    menu.addAction(self.actionMoveToTop)

                    self.actionRenameLayer = self.actions.actionRenameGroupOrLayer(menu)
                    menu.addAction(self.actionRenameLayer)

                    actionDeleteSelectedLayers = QAction('删除选中图层(D)', menu)
                    actionDeleteSelectedLayers.triggered.connect(self.deleteSelectedLayer)
                    menu.addAction(actionDeleteSelectedLayers)

                    actionOpenLayerProp = QAction('图层属性', menu)
                    actionOpenLayerProp.triggered.connect(lambda: self.openLayerPropTriggered(layer))
                    menu.addAction(actionOpenLayerProp)
                    pass
                return menu

        except:
            print(traceback.format_exc())

    def updateRasterLayerRenderer(self, widget, layer):
        print("change")
        layer.setRenderer(widget.renderer())
        self.mapCanvas.refresh()

    def deleteSelectedLayer(self):
        deleteRes = QMessageBox.question(self.mainWindows, '信息', "确定要删除所选图层？", QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if deleteRes == QMessageBox.Yes:
            layers = self.layerTreeView.selectedLayers()
            for layer in layers:
                self.deleteLayer(layer)

    def deleteAllLayer(self):
        if len(PROJECT.mapLayers().values()) == 0:
            QMessageBox.about(None, '信息', '您的图层为空')
        else:
            deleteRes = QMessageBox.question(self.mainWindows, '信息', "确定要删除所有图层？",
                                          QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.No)
            if deleteRes == QMessageBox.Yes:
                for layer in PROJECT.mapLayers().values():
                    self.deleteLayer(layer)

    def deleteGroup(self, group: QgsLayerTreeGroup):
        deleteRes = QMessageBox.question(self.mainWindows, '信息', "确定要删除组？", QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if deleteRes == QMessageBox.Yes:
            layerTreeLayers = group.findLayers()
            for layer in layerTreeLayers:
                self.deleteLayer(layer.layer())
        PROJECT.layerTreeRoot().removeChildNode(group)

    def deleteLayer(self, layer):
        PROJECT.removeMapLayer(layer)
        self.mapCanvas.refresh()
        return 0

    def openLayerPropTriggered(self, layer):
        try:
            self.lp = LayerPropWindowWidgeter(layer, self.mainWindows)
            print(type(self.lp))
            self.lp.show()
        except:
            print(traceback.format_exc())

    def openAttributeDialog(self, layer):
        ad = AttributeDialog(self.mainWindows, layer)
        ad.show()