import sys
import os
import random
import shutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QLineEdit, QMessageBox, QHBoxLayout, QProgressBar,
    QCheckBox, QComboBox
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt, QTimer


class ImageSorterApp(QWidget):
    def __init__(self):
        super().__init__()

        self.classes = {
            0: "Олень",
            1: "Косуля",
            2: "Кабала", 
            3: "Пусто",
        }
        
        self.methods = {
            "Случайно": self.distribute_randomly,
            "Равномерно": self.distribute_evenly,
        }

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Программа для выбора папки')
        self.setGeometry(300, 300, 800, 600)

        self.layout = QVBoxLayout()

        self.folder_layout = QHBoxLayout()
        self.folder_path = QLineEdit(self)
        self.folder_path.setPlaceholderText("Выберите папку...")
        self.folder_layout.addWidget(self.folder_path)

        self.browse_button = QPushButton('Выбрать папку', self)
        self.browse_button.clicked.connect(self.browse_folder)
        self.folder_layout.addWidget(self.browse_button)

        self.layout.addLayout(self.folder_layout)

        self.start_button = QPushButton('Начать', self)
        self.start_button.clicked.connect(self.start_action)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; padding: 10px;")
        self.layout.addWidget(self.start_button)

        self.manual_classification_checkbox = QCheckBox('Классифицировать вручную', self)
        self.layout.addWidget(self.manual_classification_checkbox)

        self.distribution_mode_label = QLabel('Режим распределения:', self)
        self.distribution_mode_label.setFont(QFont('Arial', 12))
        self.layout.addWidget(self.distribution_mode_label)

        self.distribution_mode_combo = QComboBox(self)
        self.distribution_mode_combo.addItems(self.methods.keys())
        self.layout.addWidget(self.distribution_mode_combo)

        self.image_count_label = QLabel('Количество изображений: 0', self)
        self.image_count_label.setFont(QFont('Arial', 14))
        self.image_count_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_count_label)

        self.subfolder_image_counts_label = QLabel(
            'Количество изображений в папках:\nОлень: 0\nКосуля: 0\nКабала: 0\nПусто: 0', self)
        self.subfolder_image_counts_label.setFont(QFont('Arial', 14))
        self.subfolder_image_counts_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.subfolder_image_counts_label)

        self.current_image_label = QLabel('Текущее изображение:', self)
        self.current_image_label.setFont(QFont('Arial', 14))
        self.current_image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.current_image_label)

        self.image_display_label = QLabel(self)
        self.image_display_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_display_label)

        self.manual_classification_layout = QHBoxLayout()
        self.manual_classification_buttons = []

        self.subfolders = self.classes.values()
        for subfolder in self.subfolders:
            button = QPushButton(subfolder.capitalize(), self)
            button.clicked.connect(lambda checked, folder=subfolder: self.move_image_manually(folder))
            self.manual_classification_buttons.append(button)
            self.manual_classification_layout.addWidget(button)

        self.layout.addLayout(self.manual_classification_layout)

        self.progress_bar = QProgressBar(self)
        self.layout.addWidget(self.progress_bar)

        self.control_layout = QHBoxLayout()
        
        self.pause_button = QPushButton('Пауза', self)
        self.pause_button.clicked.connect(self.pause_action)
        self.control_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton('Остановить', self)
        self.stop_button.clicked.connect(self.stop_action)
        self.control_layout.addWidget(self.stop_button)

        self.layout.addLayout(self.control_layout)

        self.setLayout(self.layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.move_next_image)
        self.images_to_move = []
        self.current_image_index = 0
        self.is_paused = False
        self.is_stopped = False

        self.set_manual_classification_visible(False)

    def set_manual_classification_visible(self, visible):
        self.current_image_label.setVisible(visible)
        self.image_display_label.setVisible(visible)
        for button in self.manual_classification_buttons:
            button.setVisible(visible)

    def browse_folder(self):
        folder_selected = QFileDialog.getExistingDirectory(self, 'Выбрать папку')
        if folder_selected:
            self.folder_path.setText(folder_selected)
            self.update_image_count()

    def update_image_count(self):
        folder = self.folder_path.text()
        if folder:
            count = self.count_images_in_folder(folder)
            self.image_count_label.setText(f'Количество изображений: {count}')
        else:
            self.image_count_label.setText('Количество изображений: 0')

    @staticmethod
    def count_images_in_folder(folder):
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        return sum(1 for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in image_extensions)

    def start_action(self):
        self.is_paused = False
        self.is_stopped = False
        folder = self.folder_path.text()
        if folder:
            self.create_folders(folder)
            self.images_to_move = self.get_images(folder)
            self.progress_bar.setMaximum(len(self.images_to_move))
            self.current_image_index = 0
            if self.manual_classification_checkbox.isChecked():
                self.set_manual_classification_visible(True)
                self.display_current_image(os.path.join(folder, self.images_to_move[self.current_image_index]))
            else:
                self.set_manual_classification_visible(False)
                self.timer.start(0)  # 100 мс задержка для плавного перемещения
        else:
            QMessageBox.warning(self, 'Предупреждение', 'Пожалуйста, выберите папку')

    def distribute_image(self, method):
        image = self.images_to_move[self.current_image_index]
        src_path = os.path.join(self.folder_path.text(), image)
        dest_folder = self.classes[method(src_path)]
        dest_path = os.path.join(self.folder_path.text(), dest_folder, image)
        shutil.move(src_path, dest_path)
        self.current_image_index += 1
        self.progress_bar.setValue(self.current_image_index)
        self.update_subfolder_image_counts(self.folder_path.text())

    def distribute_randomly(self, image_path):
        """принимает путь до изображения
        возвращает 0, 1, 2 или 3
        """

        return random.choice(list(self.classes.keys()))

    def distribute_evenly(self, image_path):
        return self.current_image_index % len(self.subfolders)

    def create_folders(self, folder):
        subfolders = self.classes.values()
        for subfolder in subfolders:
            path = os.path.join(folder, subfolder)
            if not os.path.exists(path):
                os.makedirs(path)

    def get_images(self, folder):
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        return [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in image_extensions]

    def move_next_image(self):
        if not self.is_paused and not self.is_stopped and self.current_image_index < len(self.images_to_move):
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
            if not self.is_stopped:
                QMessageBox.information(self, 'Информация', 'Изображения перемещены в папки.')

    def move_image_manually(self, folder):
        if self.current_image_index < len(self.images_to_move):
            image = self.images_to_move[self.current_image_index]
            src_path = os.path.join(self.folder_path.text(), image)
            dest_folder = folder
            dest_path = os.path.join(self.folder_path.text(), dest_folder, image)
            shutil.move(src_path, dest_path)
            self.current_image_index += 1
            self.progress_bar.setValue(self.current_image_index)
            self.update_subfolder_image_counts(self.folder_path.text())
            if self.current_image_index < len(self.images_to_move):
                self.display_current_image(os.path.join(self.folder_path.text(), self.images_to_move[self.current_image_index]))
            else:
                self.update_image_count()
                self.current_image_label.setText('Текущее изображение:')
                self.image_display_label.clear()
                QMessageBox.information(self, 'Информация', 'Изображения перемещены в папки.')

    def pause_action(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.setText('Возобновить')
        else:
            self.pause_button.setText('Пауза')

    def stop_action(self):
        self.is_stopped = True
        self.timer.stop()
        self.update_image_count()
        self.current_image_label.setText('Текущее изображение:')
        self.image_display_label.clear()
        QMessageBox.information(self, 'Информация', 'Операция остановлена.')

    def display_current_image(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_display_label.setPixmap(pixmap)

    def update_subfolder_image_counts(self, folder):
        subfolders = self.classes.values()
        counts = {}
        text = f'Количество изображений в папках:\n'
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
