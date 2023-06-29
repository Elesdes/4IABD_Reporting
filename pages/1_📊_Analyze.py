import pandas as pd
import streamlit as st
from src.reporting import convert_data, render
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Marijuana Arrest In Colombia - Analyze", page_icon="chart_with_upwards_trend")

# Add title and description
st.title("Marijuana Arrest In Colombia")
st.sidebar.header("Discover Our Project.")
st.write(
    """
         Here is our analyze.
         """
)


data = pd.read_csv("data/Marijuana_Arrests.csv")
columns = data.columns

for column in columns:
    dataset, data_type = convert_data(
        data[column]
    )  # Convert data to numerical, categorical, text or index

    dataset = dataset.dropna()  # Remove NaN values

    sparsity = 1.0 - len(dataset) / float(
        len(data[column])
    )  # 1 - Size after cleaning / Size before cleaning

    render(dataset, column, sparsity, data_type)

    if column == "DATETIME":
        def autopct_format(values):
            def my_format(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return '{:.1f}%\n({v:d})'.format(pct, v=val)

            return my_format


        labels = 'Nbr arrestation avant loi 2015', 'Nbr arrestation après loi 2015'

        fig, ax = plt.subplots()
        print("Dataset:\n", type(data))

        sub_data = data.loc[:, ['YEAR']].values
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
        st.write(f"### {column} [{data_type}] Post traitement")
        left, right = st.columns(2)
        p = plt.gcf()
        p.gca().add_artist(plt.Circle((0, 0), 0.3, color="white"))
        left.pyplot(fig)
        right.write(
            f"""Médiane pre 2015: {median[0]}\n
            Médiane post 2015: {median[1]}\n
            Moyenne pre 2015: {mean[0]}\n
            Moyenne post 2015: {mean[1]}"""
        )
