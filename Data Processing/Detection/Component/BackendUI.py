import ipaddress
from datetime import datetime
import os
import threading
import time
from PyQt5.QtWidgets import QInputDialog, QAction, QMainWindow, QMenu, QMenuBar, QWidget, QComboBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QCheckBox, QAbstractScrollArea, QTabWidget, QFileDialog, QLineEdit, QPushButton, QMessageBox, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5 import QtGui
import numpy as np
from qtrangeslider import QRangeSlider

import Component.Utility as Utility
from pyqtspinner import WaitingSpinner
from PyQt5.QtGui import QColor, QImage, QPixmap

from multiprocessing import Process, Manager
from multiprocessing.managers import DictProxy
from Component.UDPClient import broadcast_list
import matplotlib

from DataManager.Device import Device
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from Component.QMatplot import QMatplot

from Component.SessionManager import Session

import ctypes
import open3d as o3d
myappid = 'com.arkr.mutabe' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


class WorkerThread(QThread):
    update_signal = pyqtSignal(str)
    
    def run(self):
        while True:
            self.update_signal.emit(f"update")
            self.msleep(100)

class app_window(QMainWindow):
    
    device_image_dict: dict[int, list[QLabel]] = {}
    device_checkbox_dict: dict[int, QCheckBox] = {}
    device_timestamp_dict_global: dict[int, int] = {}
    predict_traj: list[list[float]] = []
    actual_traj: dict[int,dict[int,np.ndarray]] = {}
    data: dict[int, int] = None
    device_address_update: bool = False
    capture_image_update: bool = False
    device_connected: bool = False
    position_master: list[np.ndarray] = []
    person_coordinate_line = None
    device_coordinate_line:dict = {}
    device_point_cloud:dict = {}
    point_cloud_scatter = None
    
    # 2 people dataset: 1706868603698
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        manager = Manager()
        self.device_timestamp_dict = manager.dict()
        self.timestamp_detection_dict: dict[int, bool] = {}
        self.direct_feed = True
        self.session = None
        
        self.setWindowIcon(QtGui.QIcon('Assets/Logo.png'))
        self.setWindowTitle('MuTA')
        
        self.init_menu_bar()
        self.master_widget = QWidget()
        self.master_layout = QVBoxLayout()
        self.init_tab_layout()
        self.init_status_layout()
        self.init_collection_layout()
        self.init_process_layout()
        self.start_update_thread()
        self.master_widget.setLayout(self.master_layout)
        self.setCentralWidget(self.master_widget)
        self.setMinimumSize(800,960)
        self.show()
    
    def init_menu_bar(self):
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)
        
        file_menu = QMenu("&File", self)
        menu_bar.addMenu(file_menu)
        load_session_action = QAction("&Load Config", self)
        load_session_action.triggered.connect(self.load_session)
        save_session_action = QAction("&Save Config", self)
        save_session_action.triggered.connect(self.save_session)
        save_result_action = QAction("&Save Result Plot", self)
        save_result_action.triggered.connect(self.save_figure)
        save_result_data_action = QAction("&Save Result Data", self)
        save_result_data_action.triggered.connect(self.save_result)
        save_file_action = QAction("&Save Raw Data", self, checkable=True)
        save_file_action.triggered.connect(self.set_save_data)
        file_menu.addAction(load_session_action)
        file_menu.addAction(save_session_action)
        file_menu.addAction(save_result_action)
        file_menu.addAction(save_result_data_action)
        file_menu.addAction(save_file_action)
        
        
        edit_menu = QMenu("&Edit", self)
        menu_bar.addMenu(edit_menu)
        clear_result_action = QAction("&Clear result", self)
        clear_result_action.triggered.connect(self.clear_result)
        edit_menu.addAction(clear_result_action)
        
        view_menu = QMenu("&View", self)
        menu_bar.addMenu(view_menu)
        show_error_action = QAction("&Error", self)
        show_error_action.triggered.connect(self.calc_error)
        view_menu.addAction(show_error_action)
        show_offset_action = QAction("&Offset Adjustment", self)
        show_offset_action.triggered.connect(self.show_offset_adj)
        view_menu.addAction(show_offset_action)
        show_point_cloud_action = QAction("&Show Point Cloud", self)
        show_point_cloud_action.triggered.connect(self.show_point_cloud)
        view_menu.addAction(show_point_cloud_action)
    
    def calibrate_point_cloud(self):
        first_center: np.ndarray = None
        first_axis_z: np.ndarray = None
        first_point_cloud: o3d.geometry.PointCloud = None
        for id, device in self.session.devices.items():
            self.device_point_cloud[id] = device.generate_point_cloud()
            
        for id, device in self.session.devices.items():
            current_bounding_box = np.asarray(self.device_point_cloud[id].get_oriented_bounding_box().get_box_points())
            if first_point_cloud is None:
                first_point_cloud = self.device_point_cloud[id]
                """first_center = self.device_point_cloud[id].get_oriented_bounding_box().get_center()
                print(first_center)
                first_axis_z = current_bounding_box[3] - current_bounding_box[0]
                print(first_axis_z)"""
            else:
                print(id)
                """axis_z: np.ndarray = current_bounding_box[3] - current_bounding_box[0]
                rotation = Utility.vector_angle(axis_z, first_axis_z)
                print(rotation)
                self.device_point_cloud[id].rotate(Utility.rotation_matrix(rotation), np.array([0,0,0]))
                translation = - self.device_point_cloud[id].get_oriented_bounding_box().get_center() + first_center
                self.device_point_cloud[id].translate(translation)"""
                refine_registration = Utility.fast_registration(first_point_cloud, self.device_point_cloud[id])
                self.device_point_cloud[id].transform(refine_registration)
                print(refine_registration)
                rotation, translation = Utility.get_offset(refine_registration)
                device.set_offset(rotation[1], translation[0], translation[2])
        
        
    
    def show_point_cloud(self):
        self.calibrate_point_cloud()
        point_cloud = {}
        for id, o3d_pc in self.device_point_cloud.items():
            point_cloud[id] = np.asarray(o3d_pc.points)
        #self.session.devices[self.detection_device].generate_point_cloud()
        self.update_plot(point_cloud=point_cloud)
    
    def show_offset_adj(self):
        device, flag = QInputDialog.getItem(self, "Device Selection", "Select device to adjust", self.device_list, editable=False)
        if flag:
            current_offset = self.session.devices[int(ipaddress.ip_address(device))].offset_str()
            offset_str, flag_1 = QInputDialog.getText(self, "Offset Adjustment", "Offset adjustment in rotation, displacement x, displacement z, and time order, separate by comma(,)",text=current_offset)
            if flag_1:
                offsets = offset_str.split(",")
                self.offset_adj(device, offsets)
    
    def offset_adj(self, device, adj_list):
        print(adj_list)
        self.session.devices[int(ipaddress.ip_address(device))].set_offset(float(adj_list[0]),float(adj_list[1]),float(adj_list[2]),int(adj_list[3]))
    
    def save_session(self):
        fp = open('session.config',"w")
        fp.writelines([self.session_folder + "\n", str(ipaddress.ip_address(self.detection_device))  + "\n", str(self.timestamp_range_lower_index)  + "\n", str(self.timestamp_range_upper_index) + "\n", str(len(self.device_list)) + "\n"])
        for device_id, device in self.session.devices.items():
            fp.writelines([str(device_id) + "\n", device.offset_str() + "\n"])
        fp.close()
    
    def load_session(self):
        fp = open('session.config')
        data = fp.readlines()
        print(data)
        session_folder = data[0][5:-1]
        self.combobox.setCurrentText(session_folder)
        master_device = data[1][:-1]
        self.set_master_device(master_device)
        timestamp_lower_index = int(data[2][:-1])
        timestamp_upper_index = int(data[3][:-1])
        self.timestamp_range_slider.setValue((timestamp_lower_index, timestamp_upper_index))
        self.frame_count_field.setText(str(timestamp_upper_index - timestamp_lower_index))
        device_count = int(data[4][:-1])
        for i in range(0, device_count):
            self.offset_adj(int(data[i*2+5][:-1]), data[i*2+6][:-1].split(","))
        
    def init_tab_layout(self):
        self.tab_bar = QTabWidget(self)
        self.collect_widget = QWidget()
        self.process_widget = QWidget()
        self.collect_layout = QHBoxLayout()
        self.process_layout = QVBoxLayout()
        self.collect_widget.setLayout(self.collect_layout)
        self.process_widget.setLayout(self.process_layout)
        self.tab_bar.addTab(self.collect_widget, "Collect")
        self.tab_bar.addTab(self.process_widget, "Process")
        self.master_layout.addWidget(self.tab_bar)
    
    def init_status_layout(self):
        status_widget = QWidget()
        status_layout = QHBoxLayout()
        self.status_text = QLabel()
        self.status_text.setText("Waiting for connection")
        self.status_progress = QProgressBar()
        self.status_progress.setMaximumWidth(240)
        self.status_progress.setRange(0,0)
        self.status_progress.setValue(0)
        status_layout.addWidget(self.status_text)
        status_layout.addWidget(self.status_progress)
        status_widget.setLayout(status_layout)
        self.master_layout.addWidget(status_widget)
         
    def init_collection_layout(self):
        self.start_collection_button = QPushButton("Start TCP Server")
        self.start_collection_button.clicked.connect(self.start_collection)
        self.collect_layout.addWidget(self.start_collection_button)
        
    def start_collection(self):
        self.collect_layout.removeWidget(self.start_collection_button)
        self.start_collection_button.deleteLater()
        self.start_collection_button = None
        
        session_timestamp = time.time()
        session_folder = "data/Session_" + str(int(session_timestamp))
        if not os.path.isdir(session_folder):
            os.mkdir(session_folder)
        self.session = Session(session_folder)
        
        import Component.TCPServer as TCPServer
        Process(target=TCPServer.listen_for_connection_async, args=(session_folder, self.device_timestamp_dict, self.direct_feed)).start()
        
        threading.Thread(target=self.listen_for_new_frame, args=(self.device_timestamp_dict,self.session)).start()
        
        self.connection_table_init()
        self.image_widget = QWidget()
        self.image_grid = QVBoxLayout()
        self.image_widget.setLayout(self.image_grid)
        self.loading_label = QLabel("Waiting for Connection")
        self.spinner = WaitingSpinner(
                self.collect_widget,
                center_on_parent=False,
                roundness=100.0,
                fade=60.0,
                radius=6,
                lines=12,
                line_length=40,
                line_width=30,
                speed=2.0,
                color=QColor(0, 170, 255)
            )
        self.spinner.start()
        
        self.collect_layout.addWidget(self.spinner)
        self.collect_layout.addWidget(self.loading_label)
        self.collect_layout.addWidget(self.connection_table)
        self.collect_layout.addWidget(self.image_widget)
        
        self.connection_table.hide()
        self.image_widget.hide()
    
    def init_process_layout(self):
        all_session = os.listdir("data")
        
        path_text_field = QLineEdit()
        path_text_field.placeholderText = "Type relative/absolute directory or select..."
        self.combobox = QComboBox()
        self.combobox.addItems(all_session)
        self.combobox.currentTextChanged.connect(lambda: self.sel_session(self.combobox.currentText()))
        self.combobox.setLineEdit(path_text_field)
        
        select_button = QPushButton()
        select_button.setText("Select Directory")
        select_button.clicked.connect(self.sel_session_dir)
        
        directory_select_widget = self.add_widget_to_layout_batch([self.combobox, select_button])
        
        self.tree_widget = QTreeWidget()
        self.tree_widget.setColumnCount(4)
        self.tree_widget.setColumnWidth(0, 150)
        self.tree_widget.setColumnWidth(1, 150)
        self.tree_widget.itemSelectionChanged.connect(self.show_current_frame)
        self.tree_widget.setHeaderLabels(["Device/Timestamp", "Actual Time", "Time Delta", "Position"])
        self.tree_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self.show_context_menu)
        
        self.selection_label = QLabel("Select a master device to continue inferring")
        
        device_sel_widget = QWidget()
        self.device_sel_layout = QHBoxLayout()
        device_sel_widget.setLayout(self.device_sel_layout)
        
        self.selection_timestamp_label = QLabel("Select the timestamp range you would like to process")
        self.timestamp_range_slider = QRangeSlider()
        self.timestamp_range_slider.setOrientation(Qt.Orientation.Horizontal)
        self.timestamp_range_input_lower = QComboBox()
        self.timestamp_range_input_upper = QComboBox()
        self.timestamp_range_input_lower.setLineEdit(QLineEdit())
        self.timestamp_range_input_upper.setLineEdit(QLineEdit())
        self.timestamp_range_input_lower.currentIndexChanged.connect(self.set_timestamp_endpoint)
        self.timestamp_range_input_upper.currentIndexChanged.connect(self.set_timestamp_endpoint)
        
        self.timestamp_range_input_widget = self.add_widget_to_layout_batch([self.timestamp_range_input_lower, self.timestamp_range_input_upper])
        
        self.image_util_left = QLabel()
        self.image_util_right = QLabel()
        self.image_util_left.setMaximumSize(240,216)
        self.image_util_right.setMaximumSize(240,216)
        #image_util_collection = self.add_widget_to_layout_batch([self.image_util_left, self.image_util_right], False)
        
        self.result_plot = QMatplot()
        self.image_preview_widget = self.add_widget_to_layout_batch([self.image_util_left, self.image_util_right])
        self.init_result_plot()
        self.result_plot.show()
        
        self.frame_delay_field = QLineEdit()
        self.frame_delay_field.setText("0")
        self.frame_count_field = QLineEdit()
        self.infer_button = QPushButton("Generate All Result")
        self.infer_button_next = QPushButton("Generate Next Result")
        self.timestamp_detection_index = 0
        self.infer_button.clicked.connect(self.generate_result)
        self.infer_button_next.clicked.connect(self.generate_next_result)
        self.frame_count_field.editingFinished.connect(self.set_frame_count)
        self.infer_button_collection = self.add_widget_to_layout_batch([QLabel("Frame Delay: "), self.frame_delay_field, QLabel("Detect Frame Count"), self.frame_count_field, self.infer_button, self.infer_button_next])
        
        self.process_layout.addWidget(directory_select_widget)
        self.process_layout.addWidget(self.tree_widget)
        self.process_layout.addWidget(self.selection_label)
        self.process_layout.addWidget(device_sel_widget)
        self.process_layout.addWidget(self.selection_timestamp_label)
        self.process_layout.addWidget(self.timestamp_range_input_widget)
        self.process_layout.addWidget(self.timestamp_range_slider)
        self.process_layout.addWidget(self.image_preview_widget)
        self.process_layout.addWidget(self.infer_button_collection)
        
        self.selection_label.hide()
        self.selection_timestamp_label.hide()
        self.timestamp_range_input_widget.hide()
        self.timestamp_range_slider.hide()
        self.infer_button_collection.hide()
    
    def show_context_menu(self, event):
        context_menu = QMenu(self.tree_widget)
        detect = QAction("Detect Frame")
        set_start = QAction("Set Start Frame")
        set_end = QAction("Set End Frame")
        save_view = QAction("Save Tree View")
        detect.triggered.connect(self.detect_frame)
        set_start.triggered.connect(self.set_timestamp_start)
        set_end.triggered.connect(self.set_timestamp_end)
        save_view.triggered.connect(self.save_tree_view)
        context_menu.addAction(detect)
        context_menu.addAction(set_start)
        context_menu.addAction(set_end)
        context_menu.addAction(save_view)
        context_menu.exec_(self.tree_widget.mapToGlobal(event))
    
    def detect_frame(self):
        sender: QTreeWidgetItem = self.tree_widget.selectedItems()[0]
        self.detection_device = int(ipaddress.ip_address(sender.parent().text(0)))
        self.timestamp_range_lower_index = sender.parent().indexOfChild(sender)-1
        self.timestamp_detection_index = sender.parent().indexOfChild(sender)
        self.timestamp_range_upper_index = self.timestamp_detection_index
        print(self.timestamp_detection_index)
        self.generate_next_result()
        
    def set_timestamp_start(self):
        sender: QTreeWidgetItem = self.tree_widget.selectedItems()[0]
        sel_index = sender.parent().indexOfChild(sender)
        self.timestamp_range_input_lower.setCurrentIndex(sel_index)
    
    def set_timestamp_end(self):
        sender: QTreeWidgetItem = self.tree_widget.selectedItems()[0]
        sel_index = sender.parent().indexOfChild(sender)
        self.timestamp_range_input_upper.setCurrentIndex(sel_index)
        
    def save_tree_view(self):
        fp = open("tree_view.txt","w")
        for i in range(0,self.tree_widget.topLevelItemCount()):
            device_item = self.tree_widget.topLevelItem(i)
            for j in range(0, device_item.childCount()):
                timestamp_item = device_item.child(j)
                fp.write(device_item.text(0))
                for k in range(0, timestamp_item.columnCount()):
                    fp.write("\t" + timestamp_item.text(k))
                fp.write("\n")
        fp.close()          
    
    def add_widget_to_layout_batch(self, widgets: list[QWidget], direction_horizontal: bool = True):
        layout_widget = QWidget()
        layout = QHBoxLayout() if direction_horizontal else QVBoxLayout()
        layout_widget.setLayout(layout)
        for widget in widgets:
            layout.addWidget(widget)
        return layout_widget
    
    def save_figure(self):
        fname = QFileDialog.getSaveFileName(self, "Save Plot", "result.png", filter="Image files (*.png)")
        self.result_plot.figure.savefig(fname[0], transparent=False)
    
    def save_result(self):
        fname = QFileDialog.getSaveFileName(self, "Save Result", "result.txt")
        Utility.save_detection_result_coordinate(fname[0], self.timestamp_detection_dict, self.actual_traj, self.predict_traj, self.position_master)
    
    def set_save_data(self):
        self.direct_feed = not(self.sender().isChecked())
    
    def sel_session_dir(self):
        self.session_folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.combobox.setCurrentText(self.session_folder)
        self.set_progress_indicator(30, "Loading Directory List")
        self.load_directory()
    
    def sel_session(self, session_name):
        self.session_folder = "data/" + session_name
        self.set_progress_indicator(30, "Loading Directory List")
        self.load_directory()
    
    def load_directory(self):
        if self.session is not None:
            del(self.session)
            self.session = None
        self.device_timestamp_list_dict = Utility.list_all_device_timestamp(self.session_folder)
        if self.device_timestamp_list_dict is None:
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Error")
            dialog.setText("Folder invalid, no data is loaded")
            button = dialog.exec()
            if button == QMessageBox.Ok:
                self.set_progress_indicator(-1, "Waiting Command")
                return
        tree_items = []
        self.device_list: list[str] = []
        self.set_progress_indicator(70, "Displaying Directory List")
        self.tree_widget.clear()
        for device, timestamp_list in self.device_timestamp_list_dict.items():
            self.device_list.append(str(ipaddress.ip_address(device)))
            tree_item = QTreeWidgetItem([str(ipaddress.ip_address(device)), Utility.get_pose_from_timestamp(self.session_folder, device, timestamp_list[0])[2].__str__(), "", Utility.get_pose_from_timestamp(self.session_folder, device, timestamp_list[0])[3].__str__()])
            prev_timestamp = 0
            for timestamp in timestamp_list:
                tree_child = QTreeWidgetItem([str(timestamp), datetime.fromtimestamp(timestamp/1000).strftime("%m/%d/%Y %H:%M:%S"), str(timestamp - prev_timestamp), Utility.get_pose_from_timestamp(self.session_folder, device, timestamp)[0].__str__() + Utility.get_pose_from_timestamp(self.session_folder, device, timestamp)[1].__str__()])
                prev_timestamp = timestamp
                tree_item.addChild(tree_child)
            tree_items.append(tree_item)
        self.tree_widget.insertTopLevelItems(0,tree_items)
        
        self.selection_label.show()
        # The rule of nature is removing excess and appending lacking
        if self.device_sel_layout.count() < len(self.device_list):
            for i in range(self.device_sel_layout.count(), len(self.device_list)):
                device_button = QPushButton()
                self.device_sel_layout.addWidget(device_button)
        if self.device_sel_layout.count() > len(self.device_list):
            for i in range(self.device_sel_layout.count(), len(self.device_list)):
                op_device = self.device_sel_layout.itemAt(i).widget()
                self.device_sel_layout.removeWidget(op_device)
                op_device.deleteLater()
                op_device = None

        # Update all the devices
        for i in range(0, len(self.device_list)):
            device_button: QPushButton = self.device_sel_layout.itemAt(i).widget()
            device_button.setText(self.device_list[i])
            device_button.clicked.connect(self.set_master_action)
            
        self.set_progress_indicator(100, "Done")
        self.session = Session(self.session_folder, immediate_detection=True)

    def show_current_frame(self):
        if len(self.tree_widget.selectedItems()) != 1:
            return
        selected_timestamp_item = self.tree_widget.selectedItems()[0]
        if len(selected_timestamp_item.text(0).split("."))>1:
            return
        selected_timestamp = int(selected_timestamp_item.text(0))
        selected_device = selected_timestamp_item.parent().text(0)
        self.image_util_left.setPixmap(QPixmap(Utility.get_ab_image_from_timestamp(self.session_folder, int(ipaddress.ip_address(selected_device)), selected_timestamp)))
    
    def set_master_action(self):
        master_device = self.sender().text()
        self.set_master_device(master_device)
    
    def set_master_device(self, master_device = None):
        
        self.tree_widget.setMaximumHeight(600)
        self.detection_device = int(ipaddress.ip_address(master_device))
        
        
        for device, timestamp_list in self.device_timestamp_list_dict.items():
            if device == self.detection_device:
                self.session.try_add_device(self.detection_device)
                self.session.set_main_device(self.detection_device)
                self.primary_device = self.session.devices[device]
            else:
                self.session.try_add_device(device)
            for timestamp in timestamp_list:
                self.session.new_frame(timestamp, device)
            self.actual_traj[device] = {}

        
        
        timestamp_count = len(self.device_timestamp_list_dict[self.detection_device])
        timestamp_str_list = []
        for timestamp in self.device_timestamp_list_dict[self.detection_device]:
            timestamp_str_list.append(str(timestamp) + " " + (datetime.utcfromtimestamp(timestamp/1000).strftime("%m/%d %H:%M:%S")))
        self.timestamp_range_slider.setRange(0,timestamp_count-1)
        self.timestamp_range_slider.valueChanged.connect(self.set_timestamp_range)
        self.timestamp_range_slider.setValue((0, timestamp_count-2))
        self.selection_timestamp_label.show()
        self.timestamp_range_slider.show()
        self.infer_button_collection.show()
        self.timestamp_range_input_widget.show()
        self.timestamp_range_input_lower.clear()
        self.timestamp_range_input_lower.addItems(timestamp_str_list)
        self.timestamp_range_input_upper.clear()
        self.timestamp_range_input_upper.addItems(timestamp_str_list)
        
    
    def set_timestamp_range(self):
        self.timestamp_range_lower_index = int(self.timestamp_range_slider.value()[0])
        self.timestamp_detection_index = self.timestamp_range_lower_index
        self.timestamp_range_upper_index = int(self.timestamp_range_slider.value()[1])
        self.timestamp_range_input_lower.setCurrentIndex(self.timestamp_range_lower_index)
        self.timestamp_range_input_upper.setCurrentIndex(self.timestamp_range_upper_index)
        timestamp_lower = self.device_timestamp_list_dict[self.detection_device][self.timestamp_range_lower_index]
        timestamp_upper = self.device_timestamp_list_dict[self.detection_device][self.timestamp_range_upper_index]
        self.frame_count_field.setText(str(self.timestamp_range_upper_index - self.timestamp_range_lower_index))
        self.set_image_at_path(Utility.get_ab_image_from_timestamp(self.session_folder, self.detection_device, timestamp_lower))
        self.set_image_at_path(Utility.get_ab_image_from_timestamp(self.session_folder, self.detection_device, timestamp_upper), False)
    
    def set_timestamp_endpoint(self, lower: bool = True):
        self.timestamp_range_lower_index = self.timestamp_range_input_lower.currentIndex()
        self.timestamp_detection_index = self.timestamp_range_lower_index
        self.timestamp_range_upper_index = self.timestamp_range_input_upper.currentIndex()
        self.timestamp_range_slider.setValue((self.timestamp_range_lower_index, self.timestamp_range_upper_index))
        timestamp_lower = self.device_timestamp_list_dict[self.detection_device][self.timestamp_range_lower_index]
        timestamp_upper = self.device_timestamp_list_dict[self.detection_device][self.timestamp_range_upper_index]
        self.frame_count_field.setText(str(self.timestamp_range_upper_index - self.timestamp_range_lower_index))
        self.set_image_at_path(Utility.get_ab_image_from_timestamp(self.session_folder, self.detection_device, timestamp_lower))
        self.set_image_at_path(Utility.get_ab_image_from_timestamp(self.session_folder, self.detection_device, timestamp_upper), False)
    
    def set_frame_count(self):
        self.timestamp_range_upper_index = self.timestamp_range_lower_index + int(self.frame_count_field.text())
        self.timestamp_range_slider.setValue((self.timestamp_range_lower_index, self.timestamp_range_upper_index))
        self.timestamp_range_input_upper.setCurrentIndex(self.timestamp_range_upper_index)
        timestamp_upper = self.device_timestamp_list_dict[self.detection_device][self.timestamp_range_upper_index]
        self.set_image_at_path(Utility.get_ab_image_from_timestamp(self.session_folder, self.detection_device, timestamp_upper), False)
    
    def set_image_at_path(self, path: str, side_left: bool = True):
        image = QPixmap(path)
        if side_left:
            self.image_util_left.setPixmap(image)
        else:
            self.image_util_right.setPixmap(image)
    
    def generate_result(self):
        
        while(self.timestamp_detection_index <= self.timestamp_range_upper_index):
            self.generate_next_result()
            time.sleep(float(self.frame_delay_field.text()))
    
    def generate_next_result(self):
        timestamp_detection = self.device_timestamp_list_dict[self.detection_device][self.timestamp_detection_index]
        progress = (self.timestamp_detection_index - self.timestamp_range_lower_index) * 100 / (self.timestamp_range_upper_index - self.timestamp_range_lower_index)
        self.set_progress_indicator(int(progress), "Detecting frame " + str(timestamp_detection))
        self.session.detect_frame(self.detection_device, timestamp_detection)
        print(self.session.observation_history)
        self.timestamp_detection_dict[timestamp_detection] = False
        person_coordinate: list[list[float]] = []
        device_coordinate: dict[int, list[np.ndarray]] = {} # Device name : 2*n float matrix of points
        for device in self.session.devices.values():
            device_3d_coordinate = device.get_device_transform_at_timestamp(timestamp_detection)
            self.actual_traj[device.id][timestamp_detection] = np.array([device_3d_coordinate[0], device_3d_coordinate[2], 0])
        #anchor_coordinate_main_device = self.session.devices[self.detection_device].current_frame.pose
        #self.position_master.append(anchor_coordinate_main_device[0])#Utility.get_inv_transformation_of_point(anchor_coordinate_main_device[0], anchor_coordinate_main_device[2], anchor_coordinate_main_device[3])
        for timestamp, person_dict in self.session.observation_history.items():
            for person_id, person in person_dict.items():
                #Utility.get_inv_transformation_of_point(person.coordinate, anchor_coordinate_main_device[2], anchor_coordinate_main_device[3])
                converted_coordinate = self.session.devices[self.detection_device].get_point_with_offset(person.coordinate)
                person_coordinate.append([converted_coordinate[0],converted_coordinate[2],0])
                self.timestamp_detection_dict[timestamp] = True
                break
        for device_id, position_dict in self.actual_traj.items():
            device_coordinate[device_id] = []
            for timestamp, position in position_dict.items():
                device_coordinate[device_id].append(position.tolist())
        self.update_plot(person_coordinate, device_coordinate)
        self.predict_traj = person_coordinate
        self.timestamp_detection_index += 1
        if (self.timestamp_detection_index > self.timestamp_range_upper_index + 1):
            self.timestamp_detection_index = self.timestamp_range_lower_index
        self.set_image_at_path(Utility.get_ab_image_from_timestamp(self.session_folder, self.detection_device, timestamp_detection))
        self.set_image_at_path(self.session.devices[self.detection_device].latest_processed_image, False)
        self.set_progress_indicator(int(progress), "Waiting Input " + str(timestamp_detection))
    
    def init_result_plot(self):
        self.result_plot.aces.clear()
        self.result_plot.aces.set_aspect('equal')
    
    def update_plot(self, person_coordinate = None, device_coordinate = None, point_cloud: dict[int,np.ndarray] = None):
        self.result_plot.aces.set_xlabel("x (m)")
        self.result_plot.aces.set_ylabel("y (m)")
        self.result_plot.aces.set_zlabel("z (m)")
        if (person_coordinate is not None):
            if (len(person_coordinate) != 0):
                if (self.person_coordinate_line is None):
                    self.person_coordinate_line = self.result_plot.aces.plot(np.array(person_coordinate)[:,0], np.array(person_coordinate)[:,1], np.array(person_coordinate)[:,2], '-o', label="MuTA Result", markersize=2)[0]
                else:
                    self.person_coordinate_line.set_data(np.array(person_coordinate)[:,0], np.array(person_coordinate)[:,1])
                    self.person_coordinate_line.set_3d_properties(np.array(person_coordinate)[:,2])
        if (device_coordinate is not None):
            for device_id, device_pos in device_coordinate.items():
                if (len(device_pos) != 0):
                    if (self.device_coordinate_line.__contains__(device_id)):
                        print(type(self.device_coordinate_line[device_id]))
                        self.device_coordinate_line[device_id].set_data(np.array(device_pos)[:,0], -np.array(device_pos)[:,1])
                        self.device_coordinate_line[device_id].set_3d_properties(np.array(device_pos)[:,2])
                    else:
                        self.device_coordinate_line[device_id] = self.result_plot.aces.plot(np.array(device_pos)[:,0], -np.array(device_pos)[:,1], np.array(device_pos)[:,2], '-o', label="HoloLens " + str(ipaddress.ip_address(device_id)), markersize=3)[0]
        if (point_cloud is not None):
            for id, individual_point_cloud in point_cloud.items():
                x = individual_point_cloud[:, 0]
                z = individual_point_cloud[:, 1]
                y = individual_point_cloud[:, 2]
                if self.point_cloud_scatter is None:
                    self.result_plot.aces.scatter(x,y,z, s=2, marker='o', label='Point Cloud ' + str(ipaddress.ip_address(id)), alpha=.1)
        self.result_plot.aces.legend()
        self.result_plot.aces.set_aspect('equal')
        self.result_plot.draw()
        self.result_plot.flush_events()

    
    def calc_error(self):
        print(Utility.get_path_rmse(self.predict_traj, self.actual_traj))
        
    def clear_result(self):
        self.timestamp_detection_index = self.timestamp_range_lower_index
        self.session.observation_history.clear()
        self.result_plot.aces.clear()
        self.set_progress_indicator(-1, "Waiting Action")
        timestamp_lower = self.device_timestamp_list_dict[self.detection_device][self.timestamp_range_lower_index]
        timestamp_upper = self.device_timestamp_list_dict[self.detection_device][self.timestamp_range_upper_index]
        self.set_image_at_path(Utility.get_ab_image_from_timestamp(self.session_folder, self.detection_device, timestamp_lower))
        self.set_image_at_path(Utility.get_ab_image_from_timestamp(self.session_folder, self.detection_device, timestamp_upper), False)
        self.session.predicted_slope = {}
        self.session.prev_result = {}
        self.session.last_successful_timestamp = 0
        for device in self.actual_traj.keys():
            self.actual_traj[device].clear()
        self.device_coordinate_line.clear()
        self.person_coordinate_line = None
    
    # Value in terms of percentile and if value is -1 then regard as indefinite
    def set_progress_indicator(self, value: int = -1, text: str = None):
        if text is not None:
            self.status_text.setText(text)
        if value == -1:
            self.status_progress.setRange(0,0)
            self.status_progress.setValue(0)
        else:
            self.status_progress.setRange(0,100)
            self.status_progress.setValue(value)
    
    def start_update_thread(self):
        self.update_thread = WorkerThread()
        self.update_thread.update_signal.connect(self.update_ui)
        self.update_thread.start()
    
    def update_ui(self):
        
        if self.data != None and self.device_address_update:
            self.connection_table.setRowCount(len(self.data))
            self.connection_table.setColumnCount(3)
            i = 0
            for device in self.data.keys():
                device_id = QTableWidgetItem(str(ipaddress.ip_address(device)))
                device_timestamp = QTableWidgetItem(str(self.data[device]))
                device_master_set = QCheckBox()
                device_master_set.stateChanged.connect(self.change_master)
                self.connection_table.setItem(i,0,device_id)
                self.connection_table.setItem(i,1,device_timestamp)
                self.connection_table.setCellWidget(i,2,device_master_set)
                i += 1
            self.device_address_update = False
        
        if self.capture_image_update:
            self.update_capture_image()
            self.capture_image_update = False

    def connection_table_init(self):
        self.connection_table = QTableWidget()
        self.connection_table.verticalHeader().hide()
        self.connection_table.setHorizontalHeaderLabels(["IP Address", "Latest Timestamp", "Perform Detection"])
        self.connection_table.setMinimumWidth(300)
        
    def update_connection_widget_async(self, data: dict[int, int]):
        if not self.device_connected:
            self.device_connected = True
            self.collect_layout.removeWidget(self.spinner)
            self.collect_layout.removeWidget(self.loading_label)
            self.spinner.deleteLater()
            self.spinner = None
            self.loading_label.deleteLater()
            self.loading_label = None
            self.connection_table.show()
            self.image_widget.show()
        self.data = data
        self.device_address_update = True
    
    def update_capture_image_async(self):
        self.capture_image_update = True
    
    def update_capture_image(self):
        for device in self.session.devices.keys():
            if device not in self.device_image_dict.keys():
                image_raw_component = QLabel()
                image_processed_component = QLabel()
                device_layout = QVBoxLayout()
                device_name = QLabel(str(ipaddress.ip_address(device))) 
                device_image_layout = QHBoxLayout()
                device_image_layout.addWidget(image_raw_component)
                device_image_layout.addWidget(image_processed_component)
                device_layout.addWidget(device_name)
                device_layout.addLayout(device_image_layout)
                self.image_grid.addLayout(device_layout)
                self.collect_layout.update()
                self.device_image_dict[device] = [image_raw_component, image_processed_component]
            image_raw = self.session.devices[device].latest_raw_image
            q_image_raw = QImage(image_raw, 320, 288, 960, QImage.Format.Format_BGR888)
            pixmap_raw = QPixmap.fromImage(q_image_raw)
            self.device_image_dict[device][0].setPixmap(pixmap_raw)
            if self.session.devices[device].detect_data_ready:
                pixmap_processed = QPixmap(self.session.devices[device].latest_processed_image)
                self.device_image_dict[device][1].setPixmap(pixmap_processed)  
    
    def listen_for_new_frame(self, device_timestamp_dict: DictProxy, session):
        while True:
            try:
                device_addresses = self.device_timestamp_dict.keys()
                for i in range(0,len(device_addresses)):
                    dev_addr = device_addresses[i]
                    dev_addr_str = str(ipaddress.ip_address(dev_addr))
                    if dev_addr_str not in broadcast_list:
                        broadcast_list.append(dev_addr_str)
                        self.session.try_add_device(dev_addr)
                    if not(dev_addr in self.device_timestamp_dict_global) or self.device_timestamp_dict[dev_addr][0] != self.device_timestamp_dict_global[dev_addr]:
                            self.device_timestamp_dict_global[dev_addr] = self.device_timestamp_dict[dev_addr][0]
                            self.update_connection_widget_async(self.device_timestamp_dict_global)
                            self.session.new_frame(self.device_timestamp_dict[dev_addr][0], dev_addr, self.device_timestamp_dict[dev_addr][1])
                            self.update_capture_image_async()
                            #window.update_capture_image()
            except KeyboardInterrupt:
                break        
                
    
    def change_master():
        print("Changing master")