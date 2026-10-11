import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from map_viewer import MapViewerWidget

app = QApplication(sys.argv)
win = MapViewerWidget()
win.show()
print("Success!")
sys.exit(0)
