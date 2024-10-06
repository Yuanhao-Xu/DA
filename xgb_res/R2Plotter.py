from docx import Document
from docx.shared import Pt
import os
import matplotlib.pyplot as plt

class R2Plotter:
    def __init__(self, dataset_name, data_dict, addendum_init, addendum_size):
        """
        初始化类，设置数据集名称、数据字典、初始样本数和每次添加的样本数
        :param dataset_name: 数据集的名称
        :param data_dict: 包含各策略R2分数的数据字典
        :param addendum_init: 初始样本数
        :param addendum_size: 每次循环添加的样本数
        """
        self.dataset_name = dataset_name
        self.data_dict = data_dict
        self.addendum_init = addendum_init
        self.addendum_size = addendum_size

        # 创建一个以数据集名称命名的文件夹
        if not os.path.exists(self.dataset_name):
            os.makedirs(self.dataset_name)

    def plot_all(self):
        """
        生成所有策略的总图，并保存为 PNG 文件
        """
        plt.figure(figsize=(16, 8))  # 设置图像宽度

        for strategy, r2_scores in self.data_dict.items():
            cycles = range(1, len(r2_scores) + 1)
            # 计算每次循环的数据量
            data_volume = [self.addendum_init + (cycle - 1) * self.addendum_size for cycle in cycles]
            # 绘制每个策略的图像
            plt.plot(data_volume, r2_scores, label=strategy, linestyle='-', linewidth=2, alpha=0.8)

        # 设置标题和坐标轴标签
        plt.title(f'R^2 Scores of Different Strategies on {self.dataset_name} Dataset', fontsize=24)
        plt.xlabel('Data Volume', fontsize=22)
        plt.ylabel('R^2 Score', fontsize=22)

        # 将图例位置与对比图一致，设置在右下角
        plt.legend(loc='lower right', fontsize=18)
        plt.grid(True)

        # 添加背景网格线
        plt.grid(which='both', linestyle='--', linewidth=0.5)

        # 自动调整子图参数
        plt.tight_layout()

        # 保存为 PNG 文件，文件名为简单缩写
        file_name = f"All_Strategies_on_{self.dataset_name}.png"
        plt.savefig(os.path.join(self.dataset_name, file_name))
        plt.close()  # 关闭当前图形以节省内存

    def plot_vs_RS(self):
        """
        生成每个策略与RS的对比图，并保存为 PNG 文件
        """
        for strategy, r2_scores in self.data_dict.items():
            if strategy == 'RS':
                continue  # 跳过RS本身

            plt.figure(figsize=(16, 8))  # 设置图像宽度

            cycles = range(1, len(r2_scores) + 1)
            # 计算每次循环的数据量
            data_volume = [self.addendum_init + (cycle - 1) * self.addendum_size for cycle in cycles]

            # 绘制RS的图像
            plt.plot(data_volume, self.data_dict['RS'], label='RS', linestyle='-', linewidth=2, alpha=0.8)
            # 绘制其他策略的图像
            plt.plot(data_volume, r2_scores, label=strategy, linestyle='-', linewidth=2, alpha=0.8)

            # 设置标题和坐标轴标签
            plt.title(f'R^2 Scores Comparison between {strategy} and RS on {self.dataset_name} Dataset', fontsize=24)
            plt.xlabel('Data Volume', fontsize=22)
            plt.ylabel('R^2 Score', fontsize=22)

            # 显示图例，放在右下角
            plt.legend(loc='lower right', fontsize=18)
            plt.grid(True)

            # 添加背景网格线
            plt.grid(which='both', linestyle='--', linewidth=0.5)

            # 自动调整子图参数
            plt.tight_layout()

            # 保存为 PNG 文件，文件名为策略与RS的对比
            file_name = f"{strategy}_vs_RS_on_{self.dataset_name}.png"
            plt.savefig(os.path.join(self.dataset_name, file_name))
            plt.close()  # 关闭当前图形以节省内存

    def get_r2_samples(self, r2_thresholds, total_samples):
        """
        根据给定的r2阈值列表和训练集样本总数，返回每种策略达到阈值时的样本数和比例，并打印结果
        :param r2_thresholds: r2阈值列表 (如[0.7, 0.8, 0.9])
        :param total_samples: 训练集样本总数
        :return: 一个字典，包含每种策略在达到每个r2阈值时的样本数和比例
        """
        result = {}

        # 遍历每个策略
        for strategy, r2_scores in self.data_dict.items():
            strategy_result = {}
            samples = [self.addendum_init + (cycle - 1) * self.addendum_size for cycle in range(1, len(r2_scores) + 1)]

            # 对每个r2阈值，找到首次达到该值时的样本数和占总样本数的比例
            for r2_threshold in r2_thresholds:
                threshold_met = False  # 用来记录是否达到了阈值
                for idx, r2_score in enumerate(r2_scores):
                    if r2_score >= r2_threshold:
                        sample_count = samples[idx]
                        proportion = sample_count / total_samples
                        strategy_result[r2_threshold] = {
                            'sample_count': sample_count,
                            'proportion': proportion
                        }
                        threshold_met = True
                        break  # 找到首次达到阈值的点后，跳出循环

                # 如果遍历完未达到阈值，则设为N/A
                if not threshold_met:
                    strategy_result[r2_threshold] = {
                        'sample_count': 'N/A',
                        'proportion': 'N/A'
                    }

            result[strategy] = strategy_result

            # 打印结果
            print(f"Strategy: {strategy}")
            for r2_value, info in strategy_result.items():
                if info['sample_count'] == 'N/A':
                    print(f"  R2 >= {r2_value}: N/A")
                else:
                    print(f"  R2 >= {r2_value}: Sample Count = {info['sample_count']}, Proportion = {info['proportion']:.2%}")

        return result


    def save_report(self, result, r2_thresholds, image_folder, filename=None):
        """
        保存R2报告到Word文档，包括表格和生成的图片
        :param result: get_r2_samples 返回的结果字典
        :param r2_thresholds: R2 阈值列表
        :param image_folder: 包含图片的文件夹路径
        :param filename: 保存的 Word 文档文件名，如果为 None 则默认保存到 dataset_name 文件夹中
        """
        # 如果没有指定文件名，则默认保存到 dataset_name 文件夹中
        if filename is None:
            filename = os.path.join(self.dataset_name, 'r2_report.docx')

        # 创建Word文档
        doc = Document()
        doc.add_heading(f'R2 Samples Report for {len(r2_thresholds)} Thresholds', 0)

        # 创建表格，第一行是表头，使用默认样式
        table = doc.add_table(rows=1, cols=len(r2_thresholds) + 1)

        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Strategy'

        # 动态生成每个r2阈值的列名
        for i, r2_value in enumerate(r2_thresholds):
            hdr_cells[i + 1].text = f'R2 >= {r2_value}'

        # 填充每种策略的数据
        for strategy, data in result.items():
            row_cells = table.add_row().cells
            row_cells[0].text = strategy
            for i, r2_value in enumerate(r2_thresholds):
                if data[r2_value]['sample_count'] == 'N/A':
                    row_cells[i + 1].text = 'N/A'
                else:
                    sample_count = data[r2_value]['sample_count']
                    proportion = f"{data[r2_value]['proportion']:.2%}"
                    row_cells[i + 1].text = f"{sample_count} ({proportion})"

        # 设置表格的字体样式
        for row in table.rows:
            for cell in row.cells:
                cell.text = cell.text.strip()
                cell.paragraphs[0].runs[0].font.size = Pt(10)  # 设置字体大小

        # 插入所有生成的图片
        for image_file in os.listdir(image_folder):
            if image_file.endswith('.png'):
                doc.add_page_break()
                doc.add_paragraph(image_file)
                doc.add_picture(os.path.join(image_folder, image_file), width=Pt(400))

        # 保存文档
        doc.save(filename)
        print(f"Report saved as {filename}")

