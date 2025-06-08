import pandas as pd


class Ml_clean:
    train = pd.read_csv("dataset/train.csv")

    # # menampilkan 5 baris pertama
    # print(train.head())

    # # menampilkan ringaksan informasi dari dataset
    # print(train.info())

    # # menampilkan statistik deskriptif dari dataset
    # print(train.describe(include="all"))

    # memeriksa jumlah nilai yang hilang di setiap kolom
    missing_values = train.isnull().sum()
    # print(missing_values[missing_values > 0])

    '''MENGATASI MISSING VALUE'''
    # pisahkan kolom yang memiliki missing value lebih dari 75% dan kurang dari 75%
    less = missing_values[missing_values < 1000].index
    over = missing_values[missing_values >= 1000].index
    # print(less)
    # print(over)

    # mengisi nilai yang hiland dengan median untuk kolom numerik
    # memilah nama-nama kolom dari dataframe dengan tipe data numerik
    numeric_features = train[less].select_dtypes(include=['number']).columns
    train[numeric_features] = train[numeric_features].fillna(
        # mengisi semua nilai yang hilang (NaN) pada kolom-kolom numerik tersebut dengan nilai median
        train[numeric_features].median())

    # mengisi nilai yang hilang dengan modus/mode untuk kolom kategori
    kategorical_features = train[less].select_dtypes(
        include=['object']).columns
    for column in kategorical_features:
        train[column] = train[column].fillna(train[column].mode()[0])

    # menghapus kolom dengan banyak nilai yang hilang
    # sebelumnya kita membuat batasan lebih dari dan kurang dari 75%, untuk data yang banyak hilang (diatas 75% menurut kesepakatan kita), kita bisa mempertimbangkan untuk menghapusnya.
    df = train.drop(columns=over)

    # melakukan pemeriksaan terhadap data yang sudah melewati tahapan verifikasi missing value
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    '''MENGATASI OUTLIERS'''
    '''Path: ml_outliers agregate or delete (.py)'''  # mencoba oop
    import seaborn as sns
    import matplotlib.pyplot as plt

    # #memeriksa apakah ada outliers
    # for feature in numeric_features:
    #     plt.figure(figsize=(10, 6))
    #     sns.boxplot(x=df[feature])
    #     plt.title(f'Box Plot of {feature}')
    #     plt.show()

    # Contoh sederhana untuk mengidentifikasi outliers menggunakan IQR
    Q1 = df[numeric_features].quantile(0.25)
    Q3 = df[numeric_features].quantile(0.75)
    IQR = Q3 - Q1

    # filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliers pada kolom numerik
    condition = ~((df[numeric_features] < (Q1 - 1.5 * IQR)) |
                  (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_filtered_numeric = df.loc[condition, numeric_features]

    # menggabungkan kembali dengan kolom kategorikal
    kategorical_features = df.select_dtypes(include=['object']).columns
    df = pd.concat(
        [df_filtered_numeric, df.loc[condition, kategorical_features]], axis=1)

    # # memeriksa ulang dataset yang sudah melalui proses penanganan outliers
    # for feature in numeric_features:
    #     plt.figure(figsize=(10, 6))
    #     sns.boxplot(x=df[feature])
    #     plt.title(f'Box Plot of {feature}')
    #     plt.show()

    '''NORMALISASI DAN STANDARISASI DATA'''
    from sklearn.preprocessing import StandardScaler

    # standarisasi fitur numerik
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # histogram sebelum standarisasi
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(train[numeric_features[3]], kde=True)
    plt.title("Histogram sebelum standarisasi")

    # histpgram setelah standarisasi
    plt.subplot(1, 2, 2)
    sns.histplot(df[numeric_features[3]],  kde=True)
    plt.title("Histogram setelah standarisasi")
    # plt.show()

    '''MENANGANI DUPLIKASI DATA'''
    # mengidentifikasi baris duplikat
    duplicates = df.duplicated()
    print('baris duplikat:')
    print(df[duplicates])

    # # Menghapus baris duplikat
    # df = df.drop_duplicates()
    # print("DataFrame setelah menghapus duplikat:")
    # print(df)
