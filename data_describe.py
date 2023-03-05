import geopandas as gpd
import pandas as pd
from pyproj import CRS
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CRS_4326 = CRS('epsg:4326')

data_path = 'sun_power_dataset.csv'

# location_list = ['Camp Murray' 'Grissom' 'Hill Weber' 'JDMT' 'Kahului' 'Malmstrom'
#                  'March AFB' 'MNANG' 'Offutt' 'Peterson' 'Travis' 'USAFA']

sun_data_df = pd.read_csv(data_path, keep_default_na=False)
# sun_data_df = gpd.GeoDataFrame(
#     sun_data_df.loc[:, [c for c in sun_data_df.columns if c != "geometry"]],
#     geometry=gpd.points_from_xy(x=sun_data_df['Latitude'], y=sun_data_df['Longitude']),
#     crs=CRS_4326)

sun_data_df['Date'] = pd.to_datetime(sun_data_df['Date'], format="%Y%m%d")


def plot_sample(dataframe, axis, title=None):
    x = dataframe['Date']
    y1 = dataframe['PolyPwr']

    axis.plot(x, y1, color='tab:blue', linewidth=.8)

    axis.tick_params(axis='x', rotation=70, labelsize=7)
    axis.set_ylabel('Power Output', fontsize=12)
    axis.tick_params(axis='y', rotation=0)
    axis.grid(alpha=.4)

    axis.set_title(title)

    # axis2.set_ylabel("Temperature", fontsize=16)
    # axis2.tick_params(axis='y')

    axis.set_ylim(-10, 40)

    # 一个折线图内展示两条折线
    # y2 = dataframe['Temp']
    # axis2 = axis.twinx()
    # axis2.plot(x, y2, color='tab:red', linewidth=.8)
    # axis2.set_ylim(0, 100)


# 1.描述数据
def data_value_plot():
    print("data plot...")
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(12, 7), dpi=100)
    fig.suptitle('Power Output vs Temperature: Plotting in Secondary Y Axis')

    axes = ax.flatten()

    location_list = sun_data_df['Location'].unique()
    for i in range(len(location_list)):
        plot_sample(sun_data_df[sun_data_df['Location'] == location_list[i]], axis=axes[i], title=location_list[i])

    fig.tight_layout()


# 2.检查目标变量的分布
def data_hist_plot():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    ax.set_title('Distribution of Target Variable')
    ax.hist(sun_data_df['PolyPwr'], bins=40, density=False, facecolor="tab:blue", edgecolor="tab:orange", alpha=0.7)


def plot_is_null():
    plt.figure(figsize=(12, 7))
    sns.heatmap(sun_data_df.isnull(), cmap='viridis')
    plt.title('Visualize missing values in datasets')


def data_plot():
    data_hist_plot()
    plot_is_null()


if __name__ == '__main__':
    data_plot()
    plt.show()
