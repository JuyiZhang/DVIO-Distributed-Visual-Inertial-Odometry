import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class QMatplot(FigureCanvasQTAgg):
    def __init__(self, parent = None, width = 5, height = 4, dpi = 100):
        fig = Figure(figsize=(width, height), dpi = dpi)
        self.aces = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
        super(QMatplot, self).__init__(fig)