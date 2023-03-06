from data_describe import data_plot
from data_process import get_corr
from model import modelling_1, modelling_2
import matplotlib.pyplot as plt


def main():
    data_plot()
    get_corr()
    modelling_1()
    modelling_2()

    # 8.结论（未完成）

    plt.show()


if __name__ == '__main__':
    main()
