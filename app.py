import datetime
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve

url = "https://data.drees.solidarites-sante.gouv.fr/api/records/1.0/download/?dataset=covid-19-resultats-par-age-issus-des-appariements-entre-si-vic-si-dep-et-vac-si"
url2 = "https://data.drees.solidarites-sante.gouv.fr/explore/dataset/covid-19-resultats-par-age-issus-des-appariements-entre-si-vic-si-dep-et-vac-si/information/"

legend_title = "Statut vaccinal"
xaxis_title = "Âge"
yaxis_title = "Cas réel sur le terrain sans aucun trucage médiatique"


def transform_status(s):
    if "Non-vaccinés" in s:
        return "[0]. Non vaccinés"
    else:
        return "[1]. vacciné"

def transform_age(s):
    d = {
        "[0,19]": "0-19 ans",
        "[20,39]": "20-39 ans",
        "[40,59]": "40-59 ans",
        "[60,79]": "60-79 ans",
        "[80;+]": "80 ans et plus",
    }
    return d[s]


@st.cache(show_spinner=False)
def get_data():
    urlretrieve(url, filename="data.csv")
    df = pd.read_csv("data.csv", sep=";", on_bad_lines="skip")

    # Keep only the last 15 days available
    dates_kept = sorted(list(set(df["date"])))[-15:]
    df = df[df["date"].isin(dates_kept)]

    return df, dates_kept


df_initial, dates_kept = get_data()
earliest, latest = dates_kept[0], dates_kept[-1]
earliest = f"{earliest[-2:]}/{earliest[5:7]}"
latest = f"{latest[-2:]}/{latest[5:7]}"

@st.cache(show_spinner=False)
def preprocess_data(df_initial):
    # Relabel the vaccination status and the age columns
    df = df_initial.copy()
    df["vac_statut"] = df["vac_statut"].map(lambda s: transform_status(s))
    df["age"] = df["age"].map(lambda s: transform_age(s))

    # Aggregate the data across all dates
    agg_dict = {"hc_pcr": "sum", "sc_pcr": "sum", "dc_pcr": "sum", "effectif": "mean"}
    df = (
        df.groupby(by=["age", "vac_statut", "date"])
        .sum()
        .groupby(by=["age", "vac_statut"])
        .agg(agg_dict)
    )
    df = df.reset_index(level=["age", "vac_statut"])

    # Compute the relative numbers of hospital admissions, ICU admissions and deaths
    df["hopital"] = df.apply(lambda x: x.hc_pcr, axis=1)
    df["critique"] = df.apply(lambda x: x.sc_pcr, axis=1)
    df["mort"] = df.apply(lambda x: x.dc_pcr, axis=1)
    df["hopital_per_1M"] = df.apply(lambda x: 1e6 * x.hc_pcr / x.effectif, axis=1)
    df["critique_per_1M"] = df.apply(lambda x: 1e6 * x.sc_pcr / x.effectif, axis=1)
    df["mort_per_1M"] = df.apply(lambda x: 1e6 * x.dc_pcr / x.effectif, axis=1)
    df["hopital_per_10M"] = df.apply(lambda x: 1e7 * x.hc_pcr / x.effectif, axis=1)
    df["critique_per_10M"] = df.apply(lambda x: 1e7 * x.sc_pcr / x.effectif, axis=1)
    df["mort_per_10M"] = df.apply(lambda x: 1e7 * x.dc_pcr / x.effectif, axis=1)

    return df


df = preprocess_data(df_initial)

titles = {
    0:  r"$\bf{Véritables\ Hospitalisations}$"
        + f" du {earliest} au {latest}",
    1:  r"$\bf{Véritables\ entrées\ en\ soins\ critiques}$"
        + f" du {earliest} au {latest}",
    2: r"$\bf{Vériables\ décès}$" + f" du {earliest} au {latest}",

    3:  r"$\bf{Hospitalisations\ simulation\ sur\ 1\ million}$"
        + f" du {earliest} au {latest}",
    4:  r"$\bf{Entrées\ en\ soins\ critiques\ simulation\ sur\ 1\ million}$"
        + f" du {earliest} au {latest}",
    5: r"$\bf{Décès\ simulation\ sur\ 1\ million}$" + f" du {earliest} au {latest}",

    6:  r"$\bf{Hospitalisations\ simulation\ sur\ 10\ millions}$"
        + f" du {earliest} au {latest}",
    7:  r"$\bf{Entrées\ en\ soins\ critiques\ simulation\ sur\ 10\ millions}$"
        + f" du {earliest} au {latest}",
    8: r"$\bf{Décès\ simulation\ sur\ 10\ millions}$" + f" du {earliest} au {latest}",
}


# @st.cache(hash_funcs={matplotlib.figure.Figure: hash}, show_spinner=False)
def create_fig(key, title):
    sns.set_context("paper")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=df,
        x="age",
        y=key,
        hue="vac_statut",
        palette=sns.color_palette("rocket"),
        ax=ax,
    )
    sns.despine()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [x[5:] for x in labels], title=legend_title)
    ax.set(
        ylabel=yaxis_title, xlabel=xaxis_title, title=title,
    )
    return fig



st.sidebar.header("COVID-19 : cas graves en fonction de l'âge et du statut vaccinal en France")
st.sidebar.markdown(f"""Les graphiques de cette page montrent les chiffres réel VS les simulations trompeuse des médias.

Les médias/politiques donnent constemment leur chiffres par million d'habitants ou par 10 million d'habitant.

Soit disant pour comparé ***equitablement*** un nombre égal de personnes vacciné et non vacciné.

Hélas ces ***simulations*** sont totalement irréalistes et permettent d'inverser de façon extrème la réalité du terrain.

La saturation des lits est-elle due au nombre de places réelles prises par les patiens (et la suppression des lits des irresponsables qui nous dirigent)
Ou se fait elle sur la base de simulation fantaisites (500 réa en réel = 20 000 en simulation) ?

Ces tableaux permettent de voir la **VÉRITABLE** responsabilité des non vaccinés et la grossierté du mensonge des simulations folles!
- non vaccinés = non piqué
- vacciné = piqué au moins une fois

Données **au cours des 15 derniers jours** pour lesquels les données nationales sont disponibles ({earliest}-{latest}) et en fonction de l'**âge** et du **statut vaccinal**.

Ces graphiques sont mis à jour quotidiennement automatiquement à partir des données de la [DREES]({url2}).
""")


st.markdown(
    """
    <style>
        div.stSelectbox{
            width: 80px !important;
        }
        .element-container {
            display: flex;
            justify-content: flex-end;
            flex-direction: row;
        }
    </style>
# La réalité sur le terrain""",
    unsafe_allow_html=True,
)


for key, title in [
    ("hopital", titles[0]),
    ("critique", titles[1]),
    ("mort", titles[2]),
]:st.pyplot(create_fig(key, title))

st.markdown(
    """
# L'arnaque médiatique sur 1 million d'habitant
chiffres qui permettent à un président d'emmerdé d'honnêtes citoyens!
    """,
    unsafe_allow_html=True,
)

for key, title in [
    ("hopital_per_1M", titles[3]),
    ("critique_per_1M", titles[4]),
    ("mort_per_1M", titles[5]),
]:st.pyplot(create_fig(key, title))

st.markdown(
    """# L'arnaque médiatique sur 10 million d'habitant
chiffres qui permettent à un président d'emmerdé VRAIMENT d'honnêtes citoyens!
    """,
    unsafe_allow_html=True,
)

for key, title in [
    ("hopital_per_10M", titles[6]),
    ("critique_per_10M", titles[7]),
    ("mort_per_10M", titles[8]),
]:st.pyplot(create_fig(key, title))
