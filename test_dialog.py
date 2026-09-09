import sys
import os
from PyQt5.QtWidgets import QApplication, QFileDialog, QComboBox

app = QApplication(sys.argv)
dialog = QFileDialog(None, "Load LiDAR", "", "3D (*.obj *.ply *.pcd);;All (*)")
dialog.setOption(QFileDialog.DontUseNativeDialog, True)

# Make "Look in" combo boxes editable and connect to directory navigation
for cb in dialog.findChildren(QComboBox):
    # Only the directory combo box should be targeted, we can identify it by its name or just do all
    if cb.objectName() == "lookInCombo":
        cb.setEditable(True)
        def on_enter(cb=cb):
            path = cb.currentText()
            if os.path.isdir(path):
                dialog.setDirectory(path)
        cb.lineEdit().returnPressed.connect(on_enter)

dialog.show()
QApplication.processEvents()
print("Dialog opened")
