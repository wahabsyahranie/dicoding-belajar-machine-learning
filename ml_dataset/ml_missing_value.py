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
    # print(missing_values[missing_values > 0])

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

    # # histogram sebelum standarisasi
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # sns.histplot(train[numeric_features[3]], kde=True)
    # plt.title("Histogram sebelum standarisasi")

    # # histpgram setelah standarisasi
    # plt.subplot(1, 2, 2)
    # sns.histplot(df[numeric_features[3]],  kde=True)
    # plt.title("Histogram setelah standarisasi")
    # plt.show()

    '''MENANGANI DUPLIKASI DATA'''
    # mengidentifikasi baris duplikat
    duplicates = df.duplicated()
    # print('baris duplikat:')
    # print(df[duplicates])

    # # Menghapus baris duplikat
    # df = df.drop_duplicates()
    # print("DataFrame setelah menghapus duplikat:")
    # print(df)

    '''MENGONVERSI TIPE DATA'''
    # data kategorikal biasanya diubah menjadi bentuk numerik yang dapat dipahami oleh model yang biasa disebut encoding
    # melihat data kategorical yang ada pada dataset
    category_features = df.select_dtypes(include=['object']).columns
    df[category_features]
    # print(df[category_features])

    # Kita akan menggunakan metode one hot encoding dan label encoding karena data kategorikal yang ada pada dataset ini tidak memiliki urutan
    # one hot encoding
    df_one_hot = pd.get_dummies(df, columns=category_features)
    df_one_hot

    # label encoding
    from sklearn.preprocessing import LabelEncoder
    # inisialisasi label encoder
    label_encoder = LabelEncoder()
    df_lencoder = pd.DataFrame(df)
    for col in category_features:
        df_lencoder[col] = label_encoder.fit_transform(df[col])
    # menampilkan hasil
    # print(df_lencoder.head())

    '''LATIHAN EXPLORATORY DAN EXPLANATORY DATA ANALYSIS'''
    # pada tahap ini data sudah harus melalui tahapan cleaning dan transformation, buktikan dengan memeriksa kembali missing value pada dataset
    missing_values = df_lencoder.isnull().sum()
    missing_presentage = (missing_values / len(df_lencoder)) * 100

    missing_data = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_presentage
    }).sort_values(by='Missing Values', ascending=False)

    missing_data[missing_data['Missing Values'] > 0]
    # print(missing_data[missing_data['Missing Values'] > 0])

    # menghitung jumlah variabel
    num_vars = df_lencoder.shape[1]

    # # menentukan jumlah baris dan kolom untuk grid subplot
    # n_cols = 4
    # n_rows = -(-num_vars // n_cols)

    # # membuat subplot
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))

    # # flatten axes array untuk memudahkan iterasi jika diperlukan
    # axes = axes.flatten()

    # # plot setiap variabel
    # for i, column in enumerate(df_lencoder.columns):
    #     df_lencoder[column].hist(ax=axes[i], bins=20, edgecolor='black')
    #     axes[i].set_title(column)
    #     axes[i].set_xlabel('Value')
    #     axes[i].set_ylabel('Frequency')

    # # menghapus subplot yang tidak terpakai (jika ada)
    # for j in range(i + 1, len(axes)):
    #     fig.delaxes(axes[j])

    # # menyesuaikan layout agar lebih rapi
    # plt.tight_layout()
    # plt.show()

    # # visualisasi distribusi data untuk beberapa kolom
    # columns_to_plot = ['OverallQual', 'YearBuilt',
    #                    'LotArea', 'SaleType', 'SaleCondition']

    # plt.figure(figsize=(10, 5))
    # for i, column in enumerate(columns_to_plot, 1):
    #     plt.subplot(2, 3, i)
    #     sns.histplot(df_lencoder[column], kde=True, bins=30)
    #     plt.title(f'Distribution of {column}')
    # plt.tight_layout()
    # plt.show()

    # visualisasi korelasi antar variabel numerik
    # plt.figure(figsize=(10, 5))
    # correlation_matrix = df_lencoder.corr()

    # sns.heatmap(correlation_matrix, annot=False,
    #             cmap='coolwarm', vmin=-1, vmax=1)
    # plt.title("Correlation Matrix")
    # plt.show()

    """lakukan analisis deskriptif"""
    # # menghitung jumlah variabel
    # num_vars = df_lencoder.shape[1]

    # # menentukan jumlah baris dan kolom untuk grid subplot
    # n_cols = 4  # jumlah kolom yang diinginkan
    # # ceiling division untuk menentukan jumlah baris
    # n_rows = -(-num_vars // n_cols)

    # # membuat subplot
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))

    # # flatten axes array untuk memudahkan iterasi jika diperlukan
    # axes = axes.flatten()

    # # plt setiap variabel
    # for i, column in enumerate(df_lencoder.columns):
    #     df_lencoder[column].hist(ax=axes[i], bins=20, edgecolor='black')
    #     axes[i].set_title(column)
    #     axes[i].set_xlabel('Value')
    #     axes[i].set_ylabel('Frequency')

    # # menghapus subplot yang tidak terpakai (jika ada)
    # for j in range(i + 1, len(axes)):
    #     fig.delaxes(axes[j])

    # # menyesuaikan layout agar lebih rapi
    # plt.tight_layout()
    # plt.show()

    # # visualisasi distribusi data untuk beberapa kolom
    # columns_to_plot = ['OverallQual', 'YearBuilt',
    #                    'LotArea', 'SaleType', 'SaleCondition']

    # plt.figure(figsize=(15, 10))
    # for i, column in enumerate(columns_to_plot, 1):
    #     plt.subplot(2, 3, i)
    #     sns.histplot(df_lencoder[column], kde=True, bins=30)
    #     plt.title(f'Distribution of {column}')

    # plt.tight_layout()
    # plt.show()

    # menghitung korelasi antara variabel target dan semua variabel lainnya
    # target_corr = df_lencoder.corr()['SalePrice']

    # # (opsional) mengurutkan hasil korelasi berdasarkan korelasi
    # target_corr_sorted = target_corr.abs().sort_values(ascending=False)

    # plt.figure(figsize=(10, 6))
    # target_corr_sorted.plot(kind='bar')
    # plt.title(f'Corelation with SalePrice')
    # plt.xlabel('Variables')
    # plt.ylabel('Correlation Coefficient')
    # plt.show()

    '''DATA SPLITTING'''
    import sklearn
    # memisahkan fitur (X) dan target (y)
    X = df_lencoder.drop(columns=['SalePrice'])
    y = df_lencoder['SalePrice']

    from sklearn.model_selection import train_test_split

    # membagi dataset menjadi training dan testing
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    # print("Jumlah data:", len(X))
    # print("Jumlah data latih:", len(x_train))
    # print("Jumlah data test:", len(x_test))

    '''MELATIH MODEL (TRAINING)'''
    # melatih model 1 dengan algoritma Least Angle Regression
    from sklearn import linear_model
    lars = linear_model.Lars(n_nonzero_coefs=1).fit(x_train, y_train)

    # melatih model 2 dengan algoritma Linear Regression
    from sklearn.linear_model import LinearRegression
    LR = LinearRegression().fit(x_train, y_train)

    # melatih model 3 dengan algoritma Gradient Boosting Regressor
    from sklearn.ensemble import GradientBoostingRegressor
    GBR = GradientBoostingRegressor(random_state=184)
    GBR.fit(x_train, y_train)

    '''EVALUASI MODEL'''
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # evaluasi pada model LARS
    pred_lars = lars.predict(x_test)
    mae_lars = mean_absolute_error(y_test, pred_lars)
    mse_lars = mean_squared_error(y_test, pred_lars)
    r2_lars = r2_score(y_test, pred_lars)

    # membuat dictionary untuk menyimpan hasil evaluasi
    data = {
        'MAE': [mae_lars],
        'MSE': [mse_lars],
        'R2': [r2_lars]
    }

    # konversi dictionary menjadi DataFrame
    df_results = pd.DataFrame(data, index=['Lars'])
    print(df_results)

    # evaluasi pada model Linear Regression
    pred_LR = LR.predict(x_test)
    mae_LR = mean_absolute_error(y_test, pred_LR)
    mse_LR = mean_squared_error(y_test, pred_LR)
    r2_LR = r2_score(y_test, pred_LR)

    # menambahkan hasil evaluasi LR ke DataFrame
    df_results.loc['Linear Regression'] = [mae_LR, mse_LR, r2_LR]
    print(df_results)

    # evaluasi pada model GradientBoostingRegressor
    pred_GBR = GBR.predict(x_test)
    mae_GBR = mean_absolute_error(y_test, pred_GBR)
    mse_GBR = mean_squared_error(y_test, pred_GBR)
    r2_GBR = r2_score(y_test, pred_GBR)

    # menambahkan hasil evaluasi LR ke DataFrame
    df_results.loc['GradientBoostingRegressor'] = [mae_GBR, mse_GBR, r2_GBR]
    print(df_results)

    '''MENYIMPAN MODEL'''
    # Joblib adalah pilihan yang disarankan untuk menyimpan model scikit-learn karena lebih efisien dalam menyimpan objek model yang besar.
    import joblib
    # menyimpan model ke dalam file
    joblib.dump(GBR, 'gbr_model.joblib')

    # Pickle adalah modul standar Python yang dapat digunakan untuk menyimpan hampir semua objek Python termasuk model machine learning.
    import pickle
    # menyimpan model ke dalam file
    with open('gbr_model.pkl', 'wb') as file:
        pickle.dump(GBR, file)
