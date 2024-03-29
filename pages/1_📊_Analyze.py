import pandas as pd
import streamlit as st
import numpy as np
from src.analyze_marijuana import convert_data, render, filter_data
import matplotlib.pyplot as plt


def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)

    return my_format


# Set page configuration
st.set_page_config(page_title="Marijuana Arrest In Colombia - Analyze", page_icon="chart_with_upwards_trend")

# Add title and description
st.title("Marijuana Arrest In Colombia")
st.sidebar.header("Discover Our Project.")
st.write(
    """
         Voici notre analyse.
         """
)

data = pd.read_csv("data/Marijuana_Arrests.csv")
columns = data.columns
data = filter_data(data, columns)

for column in columns:
    if column == "YEAR":
        labels = 'Nbr arrestation avant loi 2015', 'Nbr arrestation après loi 2015'
        fig, ax = plt.subplots()
        sub_data = data.loc[:, ['YEAR']]
        sub_data = sub_data.values
        sub_data = pd.DataFrame(sub_data)
        sizes = [sub_data[sub_data < 2015].count().values[0], sub_data[sub_data > 2014].count().values[0]]
        sub_data_groupby = sub_data[0].value_counts()
        median = [sub_data_groupby[sub_data_groupby.index < 2015].median(),
                  sub_data_groupby[sub_data_groupby.index > 2014].median()]
        mean = [sub_data_groupby[sub_data_groupby.index < 2015].mean(),
                sub_data_groupby[sub_data_groupby.index > 2014].mean()]
        median = ['%.2f' % elem for elem in median]
        mean = ['%.2f' % elem for elem in mean]

        ax.pie(sizes, labels=labels, autopct=autopct_format(sizes))
        st.divider()
        st.write(f"### {column} [CAT] Post traitement")
        left, right = st.columns(2)
        p = plt.gcf()
        p.gca().add_artist(plt.Circle((0, 0), 0.3, color="white"))
        left.pyplot(fig)
        right.write(
            f'Médiane pre 2015: {median[0]}  \nMoyenne pre 2015: {mean[0]}  \nMédiane post 2015: {median[1]}  \nMoyenne post 2015: {mean[1]}')

    if column == "ARREST_BLOCKX":
        fig, ax = plt.subplots()
        sub_data = data.loc[:, ['OFFENSE_BLOCKX', 'OFFENSE_BLOCKY']]
        X = sub_data.values
        plt.xlabel('OFFENSE_BLOCKX')
        plt.ylabel('OFFENSE_BLOCKY')
        plt.scatter(x=X[:, 0], y=X[:, 1])
        st.divider()
        st.write(f"### OFFENSE_BLOCK [COORDINATES] Post traitement")
        left, right = st.columns(2)
        p = plt.gcf()
        left.pyplot(fig)
        right.write(
            f'Le bloc démontrant les méfaits prouve qu\'il existe des zones récurrentes.  \nLa zone en blanc qui traverse la carte pourrait représenter des montagnes ou bien un fleuve.')

        fig, ax = plt.subplots()
        sub_data = data.loc[:, ['ARREST_BLOCKX', 'ARREST_BLOCKY']]
        X = sub_data.values
        plt.xlabel('ARREST_BLOCKX')
        plt.ylabel('ARREST_BLOCKY')
        plt.scatter(x=X[:, 0], y=X[:, 1])
        st.divider()
        st.write(f"### {column[:-1]} [COORDINATES] Post traitement")
        left, right = st.columns(2)
        p = plt.gcf()
        left.pyplot(fig)
        right.write(
            f'Le bloc au loin démontre qu\'il existe des arrestations en dehors des blocs habituels.')

        fig, ax = plt.subplots()
        sub_data = data.loc[:, ['ARREST_BLOCKX', 'ARREST_BLOCKY']]
        sub_data = sub_data[sub_data['ARREST_BLOCKX'] < 600000]
        X = sub_data.values
        plt.xlabel('ARREST_BLOCKX')
        plt.ylabel('ARREST_BLOCKY')
        plt.scatter(x=X[:, 0], y=X[:, 1])
        p = plt.gcf()
        left.pyplot(fig)
        right.write(f'Pourtant la carte est assez similaire à celle des "OFFENSE_BLOCK."')

    if column == 'TYPE':
        fig, ax = plt.subplots()
        sub_data = data.loc[:, ['TYPE']]
        sub_data = sub_data.values
        sub_data = pd.DataFrame(sub_data).replace("Public Consumption","Public consumption").value_counts()
        sub_data = sub_data.drop(['Cultivation', 'Manufacture'])
        sub_data = pd.Series(np.append(sub_data.values, 4), index=list(sub_data.index) + [('Other',)])
        indexes = [x[0].replace("'","") for x in sub_data.index.values]
        ax.pie(sub_data, labels=indexes, autopct=autopct_format(sub_data))
        st.divider()
        st.write(f"### {column} [CAT] Post traitement")
        left, right = st.columns(2)
        p = plt.gcf()
        p.gca().add_artist(plt.Circle((0, 0), 0.3, color="white"))
        left.pyplot(fig)
        right.write(f'Les catégories sont démontrées ici.')

    if column == 'SEX':
        fig, ax = plt.subplots()
        sub_data = data.loc[:, ['SEX']]
        sub_data = sub_data.values
        sub_data = pd.DataFrame(sub_data).value_counts()
        indexes = [x[0].replace("'","") for x in sub_data.index.values]
        ax.pie(sub_data, labels=indexes, autopct=autopct_format(sub_data))
        st.divider()
        st.write(f"### {column} [CAT] Post traitement")
        left, right = st.columns(2)
        p = plt.gcf()
        p.gca().add_artist(plt.Circle((0, 0), 0.3, color="white"))
        left.pyplot(fig)
        right.write(f'Les hommes sont les personnes le plus souvent arrêté.')

    if column == 'RACE':
        fig, ax = plt.subplots()
        sub_data = data.loc[:, ['RACE']].dropna()
        sub_data = sub_data.values
        sub_data = sub_data.flatten()
        sub_data = np.char.strip(np.array(sub_data, dtype=np.str_))
        sub_data = pd.DataFrame(sub_data).value_counts()
        sub_data = sub_data.drop(['A', 'P'])
        sub_data = pd.Series(np.append(sub_data.values, 57), index=list(sub_data.index) + [('O',)])
        indexes = [x[0].strip().replace("'", "") for x in sub_data.index.values]
        print(indexes)
        print(sub_data)
        ax.pie(sub_data, labels=indexes, autopct=autopct_format(sub_data))
        st.divider()
        st.write(f"### {column} [CAT] Post traitement")
        left, right = st.columns(2)
        p = plt.gcf()
        p.gca().add_artist(plt.Circle((0, 0), 0.3, color="white"))
        left.pyplot(fig)
        right.write(f'B = Black.  \nW = White.  \nU = Unknown.  \nO = Other.')

    if column == 'ETHNICITY':
        fig, ax = plt.subplots()
        sub_data = data.loc[:, ['ETHNICITY']].dropna()
        sub_data = sub_data.values
        sub_data = sub_data.flatten()
        sub_data = np.char.strip(np.array(sub_data, dtype=np.str_))
        sub_data = pd.DataFrame(sub_data).value_counts()
        indexes = [x[0].strip().replace("'", "") for x in sub_data.index.values]
        ax.pie(sub_data, labels=indexes, autopct=autopct_format(sub_data))
        st.divider()
        st.write(f"### {column} [CAT] Post traitement")
        left, right = st.columns(2)
        p = plt.gcf()
        p.gca().add_artist(plt.Circle((0, 0), 0.3, color="white"))
        left.pyplot(fig)
        right.write(f'N = Non-Hispanic.  \nH = Hispanic.  \nU = Unknown.')

    if column not in ["YEAR", "TYPE", "ADULT_JUVENILE", "CATEGORY", "OFFENSE_DISTRICT", "ADDRESS", "GIS_ID", "CREATOR", "DEFENDANT_DISTRICT", "CREATED", "EDITOR", "EDITED", "OBJECTID",
                      "GLOBALID", "OFFENSE_BLOCKX", "OFFENSE_BLOCKY", "ARREST_BLOCKX", "ARREST_BLOCKY", "CCN", "RACE", "ETHNICITY", "SEX", "DESCRIPTION"]:
        dataset, data_type = convert_data(
            data[column]
        )  # Convert data to numerical, categorical, text or index

        dataset = dataset.dropna()  # Remove NaN values

        sparsity = 1.0 - len(dataset) / float(
            len(data[column])
        )  # 1 - Size after cleaning / Size before cleaning

        render(dataset, column, sparsity, data_type)
