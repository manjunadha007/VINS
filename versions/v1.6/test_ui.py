import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from dashboard import HyperFusionApp

app = QApplication(sys.argv)
win = HyperFusionApp()
win.show()
print("Dashboard initialized.")
sys.exit(0)
