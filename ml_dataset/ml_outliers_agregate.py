import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ml_missing_value as ml

# import class (sekaligus saja nerapkan oop)
file = ml.Ml_clean


class ml_outliers(file):
    # Alih-alih menghapus baris yang mengandung outlier (yang bisa mengurangi jumlah data dan menyebabkan bias), Kamu mengganti (mengagregasi) nilai outlier tersebut dengan median, yaitu nilai tengah dari kolom itu.
    for col in file.numeric_features:
        Q1 = file.df[col].quantile(0.25)
        Q3 = file.df[col].quantile(0.75)
        IQR = Q3 - Q1
        median = file.df[col].median()

        file.df[col] = file.df[col].apply(
            lambda x, m=median, q1=Q1, q3=Q3, iqr=IQR: m if x < (
                q1 - 1.5 * iqr) or x > (q3 + 1.5 * iqr) else x
        )
        # file.df[col] = file.df[col].apply(lambda x: median if x < (Q1 - 1.5 * IQR) or x > (Q3 + 1.5 * IQR) else x)
        # error karena :  Python "menangkap" (capture) variabel luar ke dalam lambda hanya jika konteksnya benar. Namun dalam konteks kelas atau fungsi nested, bisa gagal

    # memeriksa ulang dataset yang sudah melalui proses penanganan outliers
    for feature in file.numeric_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=file.df[feature])
        plt.title(f'Box Plot of {feature}')
        plt.show()
