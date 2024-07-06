# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('D:\\Program Files\\QGIS 3.34.8\\apps\\qgis-ltr\\plugins', 'qgis\\plugins'), ('D:\\Program Files\\QGIS 3.34.8\\apps\\Python312\\Lib\\site-packages\\PyQt5\\*.pyd', 'PyQt5'), ('D:\\Program Files\\QGIS 3.34.8\\apps\\qt5\\plugins\\styles', 'PyQt5\\Qt\\plugins\\styles'), ('D:\\Program Files\\QGIS 3.34.8\\apps\\qt5\\plugins\\iconengines', 'PyQt5\\Qt\\plugins\\iconengines'), ('D:\\Program Files\\QGIS 3.34.8\\apps\\qt5\\plugins\\imageformats', 'PyQt5\\Qt\\plugins\\imageformats'), ('D:\\Program Files\\QGIS 3.34.8\\apps\\qt5\\plugins\\platforms', 'PyQt5\\Qt\\plugins\\platforms'), ('D:\\Program Files\\QGIS 3.34.8\\apps\\qt5\\plugins\\platformthemes', 'PyQt5\\Qt\\plugins\\platformthemes')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['resources\\主图标2024.1.0.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
