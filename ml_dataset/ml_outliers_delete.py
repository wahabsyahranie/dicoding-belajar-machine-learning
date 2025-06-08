import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ml_missing_value as ml

# import class (sekaligus saja nerapkan oop)
file = ml.Ml_clean


class ml_outliers(file):
    # memeriksa dataset kita memiliki outlier atau tidak
    # for feature in file.numeric_features:
    #     plt.figure(figsize=(10, 6))
    #     sns.boxplot(x=file.df[feature])
    #     plt.title(f'Box Plot of {feature}')
    #     plt.show()

    # contoh sederhana untuk identifikasi outliers menggunakan IQR
    Q1 = file.df[file.numeric_features].quantile(0.25)
    Q3 = file.df[file.numeric_features].quantile(0.75)
    IQR = Q3 - Q1

    # filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliers pada kolom numerik
    condition = ~((file.df[file.numeric_features] < (Q1 - 1.5 * IQR))
                  | (file.df[file.numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_filtered_numeric = file.df.loc[condition, file.numeric_features]

    # menggabungkan kembali dengan kolom kategorical
    file.categorical_features = file.df.select_dtypes(include=[object]).columns
    file.df = pd.concat(
        [df_filtered_numeric, file.df.loc[condition, file.categorical_features]], axis=1)

    # memeriksa ulang dataset yang sudah melalui proses penanganan outliers
    for feature in file.numeric_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=file.df[feature])
        plt.title(f'Box Plot of {feature}')
        plt.show()
