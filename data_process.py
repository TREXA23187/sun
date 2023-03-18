import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data_path = 'sun_power_dataset.csv'

# TODO: normaliztion
df = pd.read_csv(data_path, keep_default_na=False)
class_arr = []
for i in df['PolyPwr']:
    if i < 8:
        class_arr.append(1)
    elif i < 12:
        class_arr.append(2)
    elif i < 16:
        class_arr.append(3)
    elif i < 20:
        class_arr.append(4)
    elif i < 24:
        class_arr.append(5)
    else:
        class_arr.append(6)
df['power_class'] = pd.Series(class_arr)


# 3.每个因素与功率输出之间的相关性
def get_corr():
    print("get corr...")

    df_corr = df[['Location', 'Latitude', 'Longitude', 'Altitude',
                  'Month', 'Hour', 'Season', 'Humidity', 'Temp',
                  'Wind', 'Visibility', 'Pressure', 'Cloud', 'PolyPwr']].corr()
    mask = np.triu(np.ones_like(df_corr, dtype=bool))

    f1, ax1 = plt.subplots(figsize=(10, 7))
    sns.heatmap(df_corr, ax=ax1, mask=mask, cmap='rainbow', vmax=.3, center=0, annot=True, fmt='.2f',
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    ax1.tick_params(axis='x', rotation=30)

    plt.title('Correlation analysis for all sites')


# 3.特征工程
# 4.数据拆分
def data_splint_with_feature_engineering():
    # 热编码方法对分类变量即位置和季节进行编码
    df_with_location_en = pd.get_dummies(df, columns=['Location'], drop_first=True)
    df_with_loc_season_en = pd.get_dummies(df_with_location_en, columns=['Season'], drop_first=True)

    # 使用月份和小时数据创建循环特征。
    # 只有上午10点到下午3点之间的数据可用，排除不会发电的时间段。
    # 定义时间范围
    min_hour_of_interest = 10
    max_hour_of_interest = 15

    # 计算自发电开始以来的时间差
    df_with_loc_season_en['delta_hr'] = df_with_loc_season_en.Hour - min_hour_of_interest

    # 日期特征的余弦与其实际值（月份和小时）之间存在完美的相关性。
    # 因此使用其三角函数值代替其特征。
    # 计算循环月特征
    df_with_loc_season_en['sine_mon'] = np.sin((df_with_loc_season_en.Month - 1) * np.pi / 11)
    df_with_loc_season_en['cos_mon'] = np.cos((df_with_loc_season_en.Month - 1) * np.pi / 11)
    # 计算循环小时特征
    df_with_loc_season_en['sine_hr'] = np.sin(
        (df_with_loc_season_en.delta_hr * np.pi / (max_hour_of_interest - min_hour_of_interest)))
    df_with_loc_season_en['cos_hr'] = np.cos(
        (df_with_loc_season_en.delta_hr * np.pi / (max_hour_of_interest - min_hour_of_interest)))

    # Model
    selected_columns = ['Latitude', 'Humidity', 'Temp', 'power_class', 'Wind',
                        'Visibility', 'Pressure', 'Cloud', 'Location_Grissom',
                        'Location_Hill Weber', 'Location_JDMT', 'Location_Kahului',
                        'Location_MNANG', 'Location_Malmstrom', 'Location_March AFB',
                        'Location_Offutt', 'Location_Peterson', 'Location_Travis',
                        'Location_USAFA', 'Season_Spring', 'Season_Summer', 'Season_Winter',
                        'sine_mon', 'cos_mon', 'sine_hr', 'cos_hr']

    return get_data_split(df_with_loc_season_en, selected_columns, 'power_class')


def data_splint_without_feature_engineering():
    # Model
    df_with_location_en = pd.get_dummies(df, columns=['Location'], drop_first=True)
    df_with_loc_season_en = pd.get_dummies(df_with_location_en, columns=['Season'], drop_first=True)

    selected_columns = ['Latitude', 'Longitude', 'Altitude',
                        'Month', 'Hour', 'Humidity', 'Temp',
                        'Wind', 'Visibility', 'Pressure', 'Cloud', 'Location_Grissom',
                        'Location_Hill Weber', 'Location_JDMT', 'Location_Kahului',
                        'Location_MNANG', 'Location_Malmstrom', 'Location_March AFB',
                        'Location_Offutt', 'Location_Peterson', 'Location_Travis',
                        'Location_USAFA', 'Season_Spring', 'Season_Summer', 'Season_Winter', 'power_class']

    return get_data_split(df_with_loc_season_en, selected_columns, 'power_class')


def get_data_split(data_df, selected_columns, target_label):
    df_processed = data_df[selected_columns].reset_index(drop=True)
    input_feat = list(set(selected_columns).difference({target_label}))

    df_x = df_processed[input_feat].reset_index(drop=True)
    df_y = df_processed[target_label]

    transfer = MinMaxScaler(feature_range=(0, 1))

    # normalization
    df_x = transfer.fit_transform(df_x)

    return train_test_split(df_x, df_y, test_size=0.2, random_state=42)


if __name__ == '__main__':
    get_corr()

    plt.show()
