import sys
import os
import shutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QLineEdit, QMessageBox, QHBoxLayout, QProgressBar,
    QCheckBox, QComboBox, QSizePolicy, QTabWidget, QTextEdit)
from PyQt5.QtGui import QFont, QPixmap, QIntValidator
from PyQt5.QtCore import Qt, QTimer

import torch
import timm
from torchvision import transforms

from ResNet_train_2 import train_and_validate as train_ResNet
from EfficientNet_train_2 import train_and_validate as train_EfficientNet
from functions import predict_image_class_resnet, predict_image_class_efficientnet


class ImageSorterApp(QWidget):
    def __init__(self):
        super().__init__()
               
        self.classes = {
            0: "Кабарга",
            1: "Косуля",
            2: "Олень",             
            3: "Пусто",
        }

        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        self.load_ResNet()
        self.load_EfficientNet()

        self.methods = {
            "ResNet_20": self.distribute_ResNet_20,
            "EfficientNetB1": self.distribute_EfficientNet_B1,
        }

        self.initUI()
    
    def load_ResNet(self):
        self.classify_model = timm.create_model('resnext50_32x4d.a3_in1k', pretrained=False)
        model_state_dict = torch.load('best_model_ResNet_20.pth', map_location=torch.device('cpu'))
        self.classify_model.load_state_dict(model_state_dict)
        self.classify_model.eval()

        data_config = timm.data.resolve_model_data_config(self.classify_model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def load_EfficientNet(self):
        self.transforms2 = preprocess = transforms.Compose([
            transforms.Resize(240),  # Измените размер на 240x240 для EfficientNet-B1
            transforms.CenterCrop(240),
            transforms.ToTensor()])
        
        self.classify_model2 = timm.create_model("efficientnet_b1", pretrained=False, num_classes=3)
        self.classify_model2.load_state_dict(torch.load('best_model_EfficientNet.pth', map_location=torch.device('cpu')))
        self.classify_model2.to(torch.device('cpu'))
        self.classify_model2.eval()

    def initUI(self):
        self.setWindowTitle('Программа для выбора папки')
        self.setGeometry(300, 300, 800, 600)

        self.tab_widget = QTabWidget()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.tab_widget)

        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.tab_widget.addTab(self.tab1, "Классификация")
        self.tab_widget.addTab(self.tab2, "Обучение")

        self.init_tab1()
        self.init_tab2()

    def init_tab1(self):
        layout = QVBoxLayout()

        #region: choose folder
        self.folder_layout = QHBoxLayout()

        self.folder_path = QLineEdit(self)
        self.folder_path.setPlaceholderText("Выберите папку...")
        self.folder_layout.addWidget(self.folder_path)

        self.browse_button = QPushButton('Выбрать папку', self)
        self.browse_button.clicked.connect(self.browse_folder)
        self.folder_layout.addWidget(self.browse_button)

        layout.addLayout(self.folder_layout)
        #endregion
        
        #region: controls
        self.control_layout = QHBoxLayout()

        self.start_button = QPushButton('Начать', self)
        self.start_button.clicked.connect(self.start_action)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; padding: 10px;")
        self.control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton('Остановить', self)
        self.stop_button.clicked.connect(self.stop_action)
        self.stop_button.setStyleSheet("background-color: #AF4C50; color: white; font-size: 16px; padding: 10px;")
        self.control_layout.addWidget(self.stop_button)

        layout.addLayout(self.control_layout)
        #endregion

        #region: choose model
        self.model_layout = QHBoxLayout()

        self.distribution_mode_label = QLabel('Модель для классификации:', self)
        self.model_layout.addWidget(self.distribution_mode_label)

        self.distribution_mode_combo = QComboBox(self)
        self.distribution_mode_combo.addItems(self.methods.keys())
        self.model_layout.addWidget(self.distribution_mode_combo)

        layout.addLayout(self.model_layout)

        self.manual_classification_checkbox = QCheckBox('Классифицировать вручную', self)
        layout.addWidget(self.manual_classification_checkbox)
        #endregion

        self.image_count_text = 'Количество изображений: '
        self.image_count_label = QLabel(self.image_count_text + '0', self)
        self.image_count_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.image_count_label.setFont(QFont('Arial', 14))
        self.image_count_label.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self.image_count_label)

        self.subfolder_image_counts_text = 'Количество изображений в папках:\n'
        self.subfolder_image_counts_label = QLabel(self.subfolder_image_counts_text + 
                                                   "".join(map(lambda x: x + ': 0\n', self.classes.values())), self)
        self.image_count_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.subfolder_image_counts_label.setFont(QFont('Arial', 14))
        self.subfolder_image_counts_label.setAlignment(Qt.AlignHCenter)

        layout.addWidget(self.subfolder_image_counts_label)

        self.current_image_label = QLabel('Текущее изображение:', self)
        self.current_image_label.setFont(QFont('Arial', 14))
        self.current_image_label.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self.current_image_label)

        self.image_display_label = QLabel(self)
        self.image_display_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_display_label)

        self.manual_classification_layout = QHBoxLayout()
        self.manual_classification_buttons = []

        self.subfolders = self.classes.values()
        for subfolder in self.subfolders:
            button = QPushButton(subfolder.capitalize(), self)
            button.clicked.connect(lambda checked, folder=subfolder: self.move_image_manually(folder))
            self.manual_classification_buttons.append(button)
            self.manual_classification_layout.addWidget(button)

        layout.addLayout(self.manual_classification_layout)

        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.move_next_image)
        self.images_to_move = []
        self.current_image_index = 0
        self.is_paused = False
        self.is_stopped = False

        self.tab1.setLayout(layout)

        self.set_manual_classification_visible(False)

    def init_tab2(self):
        layout = QVBoxLayout()

        self.folder_layout_kabarga = QHBoxLayout()

        self.folder_path_kabarga = QLineEdit(self)
        self.folder_path_kabarga.setPlaceholderText("Выберите папку с изображениями кабарги...")
        self.folder_layout_kabarga.addWidget(self.folder_path_kabarga)

        self.browse_button_kabarga = QPushButton('Выбрать', self)
        self.browse_button_kabarga.clicked.connect(lambda: self.browse_folder(self.folder_path_kabarga))
        self.folder_layout_kabarga.addWidget(self.browse_button_kabarga)

        layout.addLayout(self.folder_layout_kabarga)

        self.folder_layout_kosulya = QHBoxLayout()

        self.folder_path_kosulya = QLineEdit(self)
        self.folder_path_kosulya.setPlaceholderText("Выберите папку с изображениями косуль...")
        self.folder_layout_kosulya.addWidget(self.folder_path_kosulya)

        self.browse_button_kosulya = QPushButton('Выбрать', self)
        self.browse_button_kosulya.clicked.connect(lambda: self.browse_folder(self.folder_path_kosulya))
        self.folder_layout_kosulya.addWidget(self.browse_button_kosulya)

        layout.addLayout(self.folder_layout_kosulya)

        self.folder_layout_olen = QHBoxLayout()

        self.folder_path_olen = QLineEdit(self)
        self.folder_path_olen.setPlaceholderText("Выберите папку с изображениями оленей...")
        self.folder_layout_olen.addWidget(self.folder_path_olen)

        self.browse_button_olen = QPushButton('Выбрать', self)
        self.browse_button_olen.clicked.connect(lambda: self.browse_folder(self.folder_path_olen))
        self.folder_layout_olen.addWidget(self.browse_button_olen)

        layout.addLayout(self.folder_layout_olen)

        self.model_train_layout = QHBoxLayout()

        self.train_epochs_label = QLabel('Кол-во эпох:', self)
        self.model_train_layout.addWidget(self.train_epochs_label)

        self.train_epochs_input = QLineEdit(self)
        self.train_epochs_input.setPlaceholderText("Введите число...")
        self.train_epochs_input.setValidator(QIntValidator(0, 10000, self))  # Пример валидатора от 0 до 10000
        self.model_train_layout.addWidget(self.train_epochs_input)

        self.train_model_label = QLabel('Модель для классификации:', self)
        self.model_train_layout.addWidget(self.train_model_label)

        self.train_model_combo = QComboBox(self)
        self.train_model_combo.addItems(list(self.methods.keys())[:-1])
        self.model_train_layout.addWidget(self.train_model_combo)

        layout.addLayout(self.model_train_layout)

        start_button2 = QPushButton('Начать', self)
        start_button2.clicked.connect(self.start_train)
        layout.addWidget(start_button2)

        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        self.tab2.setLayout(layout)

    def start_train(self):
        kabarga_path = self.folder_path_kabarga.text()
        kosulya_path = self.folder_path_kosulya.text()
        olen_path = self.folder_path_olen.text()

        epochs = int(self.train_epochs_input.text())

        if self.train_model_combo.currentText() == "ResNet_20":
            train_ResNet([kabarga_path, kosulya_path, olen_path],
                     'best_model_ResNet_20.pth',
                     'best_model_ResNet_20.pth',
                     epochs,
                     'cpu', 
                     log_method=self.log_output.append)
            self.load_ResNet()
            
        elif self.train_model_combo.currentText() == "EfficientNetB1":
            train_EfficientNet(
                [kabarga_path, kosulya_path, olen_path],
                     'best_model_EfficientNet.pth',
                     epochs,
                     'cpu', 
                     log_method=self.log_output.append)
            self.load_EfficientNet()     

    def set_manual_classification_visible(self, visible):
        self.current_image_label.setVisible(visible)
        self.image_display_label.setVisible(visible)
        for button in self.manual_classification_buttons:
            button.setVisible(visible)

    def browse_folder(self, line_edit=None):
        folder_selected = QFileDialog.getExistingDirectory(self, 'Выбрать папку')
        if folder_selected and line_edit:
            line_edit.setText(folder_selected)
        elif folder_selected:
            self.folder_path.setText(folder_selected)
            self.update_image_count()

    def update_image_count(self):
        folder = self.folder_path.text()
        if folder:
            count = self.count_images_in_folder(folder)
            self.image_count_label.setText(self.image_count_text + str(count))
        else:
            self.image_count_label.setText(self.image_count_text + '0')

    @staticmethod
    def count_images_in_folder(folder):
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        sum = 0
        for root, dirs, files in os.walk(folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    sum += 1
        return sum

    def start_action(self):
        self.is_paused = False
        self.is_stopped = False
        folder = self.folder_path.text()
        if folder:
            self.create_folders(folder)
            self.images_to_move = self.get_images(folder)
            if not len(self.images_to_move): 
                QMessageBox.warning(self, 'Предупреждение', 'Изображений не найдено!\nПожалуйста, выберите другую папку')
                return
            self.progress_bar.setMaximum(len(self.images_to_move))
            self.current_image_index = 0
            if self.manual_classification_checkbox.isChecked():
                self.set_manual_classification_visible(True)
                self.display_current_image(os.path.join(folder, self.images_to_move[self.current_image_index]))
            else:
                self.set_manual_classification_visible(False)
                self.timer.start(0)  
        else:
            QMessageBox.warning(self, 'Предупреждение', 'Пожалуйста, выберите папку')

    def distribute_image(self, method):
        image_path = self.images_to_move[self.current_image_index]
        dest_folder = self.classes[method(image_path)]
        dest_path = os.path.join(self.folder_path.text(), dest_folder, os.path.basename(image_path))
        shutil.move(image_path, dest_path)
        self.current_image_index += 1
        self.progress_bar.setValue(self.current_image_index)
        self.update_subfolder_image_counts(self.folder_path.text())

    def distribute_ResNet_20(self, image_path):
        predicted_class_idx = 3
        try:
            predicted_class_idx = predict_image_class_resnet(image_path, 
                                                      self.classify_model, 
                                                      self.transforms, 
                                                      self.yolo_model,
                                                      torch.device('cpu'),
                                                      return_not_found=True)
        except Exception as ex:
            QMessageBox.warning(self, 'Ошибка!', f'Возникла проблема при обработке изображения {image_path}')

        return predicted_class_idx

    def distribute_EfficientNet_B1(self, image_path):
        predicted_class_idx = 3
        try:
            predicted_class_idx = predict_image_class_efficientnet(image_path, 
                                                      self.classify_model2, 
                                                      self.transforms2, 
                                                      self.yolo_model,
                                                      torch.device('cpu'),
                                                      return_not_found=True)
        except Exception as ex:
            QMessageBox.warning(self, 'Ошибка!', f'Возникла проблема при обработке изображения {image_path}')

        return predicted_class_idx

    def create_folders(self, folder):
        subfolders = self.classes.values()
        for subfolder in subfolders:
            path = os.path.join(folder, subfolder)
            if not os.path.exists(path):
                os.makedirs(path)

    def get_images(self, folder):    
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        images = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    images.append(os.path.join(root, file))
        return images

    def move_next_image(self):
        if not self.is_stopped and self.current_image_index < len(self.images_to_move):
            distribution_mode = self.distribution_mode_combo.currentText()
            distribution_mode = self.distribution_mode_combo.currentText()
            distribution_method = self.methods[distribution_mode]
            self.distribute_image(distribution_method)
        else:
            self.timer.stop()
            self.update_image_count()
            if self.manual_classification_checkbox.isChecked():
                self.current_image_label.setText('Текущее изображение:')
                self.image_display_label.clear()

    def move_image_manually(self, folder):
        if self.current_image_index < len(self.images_to_move):
            image_path = self.images_to_move[self.current_image_index]
            dest_folder = folder
            dest_path = os.path.join(self.folder_path.text(), dest_folder, os.path.basename(image_path))
            shutil.move(image_path, dest_path)
            self.current_image_index += 1
            self.progress_bar.setValue(self.current_image_index)
            self.update_subfolder_image_counts(self.folder_path.text())
            if self.current_image_index < len(self.images_to_move):
                self.display_current_image(self.images_to_move[self.current_image_index])
            else:
                self.update_image_count()
                self.current_image_label.setText('Текущее изображение:')
                self.image_display_label.clear()
           
    def stop_action(self):
        self.is_stopped = True

    def display_current_image(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_display_label.setPixmap(pixmap)

    def update_subfolder_image_counts(self, folder):
        subfolders = self.classes.values()
        counts = {}
        text = self.subfolder_image_counts_text
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder, subfolder)
            counts[subfolder] = self.count_images_in_folder(subfolder_path)
            text += f'{subfolder}: {counts[subfolder]}\n'

        self.subfolder_image_counts_label.setText(text)
        self.subfolder_counts = counts

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageSorterApp()
    ex.show()
    sys.exit(app.exec_())
