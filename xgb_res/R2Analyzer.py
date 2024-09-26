import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches
import os

class R2Analyzer:
    def __init__(self, data_dicts, r2_thresholds, total_samples, addendum_init, addendum_size):
        """
        初始化类
        :param data_dicts: 不同数据集字典
        :param r2_thresholds: r2阈值列表
        :param total_samples: 每个数据集总样本数
        :param addendum_init: 初始样本数
        :param addendum_size: 每次循环增加样本数
        """
        self.data_dicts = data_dicts
        self.r2_thresholds = r2_thresholds
        self.total_samples = total_samples
        self.addendum_init = addendum_init
        self.addendum_size = addendum_size

    def find_r2_proportions(self):
        """
        返回以策略命名的字典，每个字典的key是不同数据集，value是两个列表：
        1. 达到每个r2阈值时的样本比例。
        """
        strategy_dict = {}
        for dataset_name, strategy_data in self.data_dicts.items():
            for strategy, r2_values in strategy_data.items():
                if strategy not in strategy_dict:
                    strategy_dict[strategy] = {}
                sample_counts = []
                proportions = []
                for threshold in self.r2_thresholds:
                    for i, r2_value in enumerate(r2_values):
                        if r2_value >= threshold:
                            samples_used = self.addendum_init + i * self.addendum_size
                            proportion_used = samples_used / self.total_samples
                            sample_counts.append(samples_used)
                            proportions.append(proportion_used)
                            break
                    else:
                        sample_counts.append(None)
                        proportions.append(None)
                strategy_dict[strategy][dataset_name] = proportions
        return strategy_dict

    def plot_r2_proportions(self, proportions_dict, image_folder):
        """
        根据返回的proportions字典，绘制每种策略在不同数据集下达到阈值时的数据占比图，并保存图片。
        所有图像的横坐标范围保持一致 [0, 1]。
        :param proportions_dict: 各策略在不同数据集下达到阈值时的数据比例字典。
        :param image_folder: 保存图片的文件夹路径。
        """
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        for strategy, dataset_proportions in proportions_dict.items():
            plt.figure(figsize=(8, 6))
            for dataset_name, proportions in dataset_proportions.items():
                plt.plot(proportions, self.r2_thresholds, marker='o', label=dataset_name)

            plt.xlabel('Sample Proportion')
            plt.ylabel('R2 Thresholds')
            plt.title(f'{strategy} - Proportion of Samples for Each Threshold')
            plt.legend()
            plt.grid(True)

            # 设置纵坐标只显示阈值，不显示其他刻度
            plt.yticks(self.r2_thresholds)

            # 固定横坐标范围为 [0, 1]
            plt.xlim([0, 1])

            # 保存图片
            image_path = os.path.join(image_folder, f'{strategy}_r2_proportions.png')
            plt.savefig(image_path)
            plt.close()

    def save_images_and_tables_to_word(self, result, image_folder, word_file_name='dataset_analysis_report.docx'):
        """
        将指定文件夹中的所有图片和数据比例表格插入到一个Word文档中，并设置字体格式。
        :param result: 策略命名的字典，每个字典的key是不同数据集，value是两个列表，一个是样本数列表，一个是比例列表。
        :param image_folder: 存储图片的文件夹路径。
        :param word_file_name: 生成的Word文档的文件名。
        """
        word_path = os.path.join(image_folder, word_file_name)
        doc = Document()

        # 设置文档标题
        doc.add_heading('R2 Proportions Analysis', 0)

        # 插入图片
        for strategy in result.keys():
            image_path = os.path.join(image_folder, f'{strategy}_r2_proportions.png')
            if os.path.exists(image_path):
                doc.add_heading(f'{strategy} Strategy - R2 Proportions', level=1)
                doc.add_picture(image_path, width=Inches(5.5))

        # 插入表格
        for strategy, dataset_proportions in result.items():
            doc.add_heading(f'{strategy} Strategy - Proportion Table', level=1)
            table = doc.add_table(rows=1, cols=len(self.r2_thresholds) + 1)
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Dataset'
            for i, threshold in enumerate(self.r2_thresholds):
                hdr_cells[i + 1].text = f'R2 ≥ {threshold}'

            for dataset_name, proportions in dataset_proportions.items():
                row_cells = table.add_row().cells
                row_cells[0].text = dataset_name
                for i, proportion in enumerate(proportions):
                    row_cells[i + 1].text = f'{proportion:.2f}' if proportion is not None else 'N/A'

        # 保存文档
        doc.save(word_path)
