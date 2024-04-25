import io
import streamlit as st
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import *
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
from sklearn.cluster import KMeans

# Загрузка данных
def load_data():
    # Используйте полный путь к вашему файлу
    data = pd.read_excel(r'C:\Users\Сергей\Desktop\project\wb_teapot.xlsx')
    return data

# Функция для отображения описания полей данных
def show_data_description(df):
    st.subheader("Описание полей данных:")
    st.write("Этот раздел содержит описание каждого поля в исходных данных.")
    st.write("Пожалуйста, ознакомьтесь с ним перед анализом данных.")

    # Описание полей
    df_description = {
        "id": "Номер",
        "Review": "Количество отзывов",
        "Star": "Оценка (0 - не было оценок)",
        "Value": "Стоимость с учетом скидки",
        "brandId": "Код бренда",
        "brandName": "Бренд",
        "goodsName": "Наименование товара",
        "isSoldOut": "Распродажа",
        "link": "Ссылка",
        "lowQuantity": "Мало в остатках",
        "ordersCount": "Исторические продажи, штук, с момента появления на маркетплейсе",
        "price": "Стоимость без скидок",
        "qualityRate": "Оценка качества/ удовлетворенность пользователей",
        "Вес с упаковкой (кг)": "Вес с упаковкой (кг)",
        "Длина кабеля": "Длина кабеля",
        "Количество температурных режимов": "Количество температурных режимов",
        "Материал корпуса": "Материал корпуса",
        "Модель": "Модель",
        "Мощность устройства": "Мощность устройства",
        "Объем чайника": "Объем чайника",
        "Страна бренда": "Страна бренда",
        "Страна производитель": "Страна производитель",
        "Цвет": "Цвет",
        "sale_june": "Продажи за июнь 2020 года"
    }

    # Вывод описания полей
    for column, description in df_description.items():
        st.write(f"**{column}**: {description}")

# Функция для анализа и предобработки данных
def data_analysis(df):
    st.subheader("Анализ и предобработка данных:")
    st.write("Этот раздел содержит анализ и предобработку данных перед их использованием в анализе и визуализации.")

    # Поиск нулевых значений
    st.write("### Нулевые значения:")
    st.write(df.isnull().sum())

    # Количество уникальных значений для любого столбца
    st.write("### Количество уникальных значений для каждого столбца:")
    st.write(df.nunique())

    # Описательная статистика данных
    st.write("### Описательная статистика данных:")
    st.write(df.describe())

# Функция для подготовки
def data_preporation(df):
    st.subheader("Подготовка данных:")
    st.write("Этот раздел содержит подготовку данных для кластерного анализа.")

    col=['Review', 'Star', 'ordersCount']
    st.write('Колонки, которые мы будем использовать:', col)
    pd.options.mode.chained_assignment = None 
    df[col].fillna(0, inplace=True)
    
    st.write("Таблица корреляции:")
    from pandas.plotting import scatter_matrix
    scatter_matrix(df[col], alpha=0.05, figsize=(10, 10));
    df[col].corr()   
    
def df_k(df):
    st.subheader("Метод k средних:")
    st.write("Этот раздел содержит кластерный анализ методом k средних.")
    col=['Review', 'Star', 'ordersCount']
    # загружаем библиотеку препроцесинга данных
    # эта библиотека автоматически приведен данные к нормальным значениям
    from sklearn import preprocessing
    dataNorm = preprocessing.MinMaxScaler().fit_transform(df[col].values)   

    # Вычислим расстояния между каждым набором данных,
    # т.е. строками массива data_for_clust
    # Вычисляется евклидово расстояние (по умолчанию)
    data_dist = pdist(dataNorm, 'euclidean')
    
    data_linkage = linkage(data_dist, method='average')  

    # Метод локтя. Позволячет оценить оптимальное количество сегментов.
    # Показывает сумму внутри групповых вариаций
    last = data_linkage[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)
    
    acceleration = np.diff(last, 2)  
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()
    k = acceleration_rev.argmax() + 2 
    print("Рекомендованное количество кластеров:", k)

    nClust=6

    # строим кластеризаци методом KMeans
    km = KMeans(n_clusters=nClust).fit(dataNorm)
    # выведем полученное распределение по кластерам
    # так же номер кластера, к котрому относится строка, так как нумерация начинается с нуля, выводим добавляя 1
    km.labels_ +1
    st.write("Отношение обзоров и количесва заказов.")
    x=0 # Чтобы построить диаграмму в разных осях, меняйте номера столбцов
    y=2 #
    centroids = km.cluster_centers_
    plt.figure(figsize=(10, 8))
    plt.scatter(dataNorm[:,x], dataNorm[:,y], c=km.labels_, cmap='flag')
    plt.scatter(centroids[:, x], centroids[:, y], marker='*', s=300,
                c='r', label='centroid')
    plt.xlabel(col[x])
    plt.ylabel(col[y]);
    plt.show()
    st.write("Таблица средних по кластерам.")
    # к оригинальным данным добавляем номера кластеров
    df['KMeans']=km.labels_+1
    res=df.groupby('KMeans')[col].mean()
    res['Количество']=df.groupby('KMeans').size().values
    res
# Заголовок
st.title("Кластерный анализ")
    
# Загрузка данных
data = load_data()

# Вывод данных
st.write("Демонстрация данных:")
st.write(data)

# Описание полей данных
show_data_description(data)

# Анализ и предобработка данных
data_preporation(data)

# Визуализация
df_k(data)
