"""
MOVIE RECOMMENDATION SYSTEM - WEB UI (Streamlit)
Collaborative Filtering using KNN | ML Mini Project

Run:
    pip install streamlit pandas numpy scikit-learn
    streamlit run app.py

Needs ratings.csv and movies.csv (MovieLens ml-latest-small) in the same folder.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

K = 10             # number of neighbours (KNN)
MIN_RATINGS = 20   # keep users/movies with at least this many ratings
MIN_SUPPORT = 3    # a recommendation needs >= 3 neighbours who rated it


# ----------------------------------------------------------------------
# DATA  (Pandas: load, clean, filter)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")[["userId", "movieId", "rating"]]
    movies = pd.read_csv("movies.csv")[["movieId", "title", "genres"]]
    ratings = ratings.dropna().drop_duplicates(["userId", "movieId"])

    mc = ratings.groupby("movieId").size()
    ratings = ratings[ratings.movieId.isin(mc[mc >= MIN_RATINGS].index)]
    uc = ratings.groupby("userId").size()
    ratings = ratings[ratings.userId.isin(uc[uc >= MIN_RATINGS].index)]
    movies = movies[movies.movieId.isin(ratings.movieId)].reset_index(drop=True)
    return ratings.reset_index(drop=True), movies


# ----------------------------------------------------------------------
# MODEL  (NumPy matrix + KNN with cosine similarity)
# ----------------------------------------------------------------------
def build_model(df, user_ids, movie_ids):
    matrix = (df.pivot_table(index="userId", columns="movieId", values="rating")
                .reindex(index=user_ids, columns=movie_ids)
                .fillna(0).values)
    user_knn = NearestNeighbors(metric="cosine", algorithm="brute").fit(matrix)
    item_knn = NearestNeighbors(metric="cosine", algorithm="brute").fit(matrix.T)
    u_d, u_n = user_knn.kneighbors(matrix, n_neighbors=K + 1)
    i_d, i_n = item_knn.kneighbors(matrix.T, n_neighbors=K + 1)
    g_mean = df.rating.mean()
    row_mean = np.array([r[r > 0].mean() if (r > 0).any() else g_mean
                         for r in matrix])
    return dict(matrix=matrix, u_nbr=u_n[:, 1:], u_sim=1 - u_d[:, 1:],
                i_nbr=i_n[:, 1:], i_sim=1 - i_d[:, 1:],
                row_mean=row_mean, g_mean=g_mean)


def predict_user_based(m, ui):
    """Predict ratings of user ui for ALL movies from similar users."""
    r = m["matrix"][m["u_nbr"][ui]]
    w = m["u_sim"][ui][:, None] * (r > 0)
    den, num = w.sum(0), (w * r).sum(0)
    pred = np.where(den > 0, num / np.where(den > 0, den, 1), m["row_mean"][ui])
    return pred, (r > 0).sum(0)


def predict_item_based_vec(m, vec, fallback):
    """Predict ratings for ALL movies from a rating vector (similar movies)."""
    r = vec[m["i_nbr"]]
    w = m["i_sim"] * (r > 0)
    den, num = w.sum(1), (w * r).sum(1)
    pred = np.where(den > 0, num / np.where(den > 0, den, 1), fallback)
    return pred, (r > 0).sum(1)


@st.cache_resource
def get_model():
    ratings, movies = load_data()
    user_ids = sorted(ratings.userId.unique())
    movie_ids = sorted(ratings.movieId.unique())
    m = build_model(ratings, user_ids, movie_ids)
    m.update(user_ids=user_ids, movie_ids=movie_ids,
             uidx={u: i for i, u in enumerate(user_ids)},
             midx={x: i for i, x in enumerate(movie_ids)})
    return m


@st.cache_data
def evaluate():
    """RMSE of Linear Regression vs user-based KNN vs item-based KNN."""
    ratings, _ = load_data()
    user_ids = sorted(ratings.userId.unique())
    movie_ids = sorted(ratings.movieId.unique())
    uidx = {u: i for i, u in enumerate(user_ids)}
    midx = {x: i for i, x in enumerate(movie_ids)}
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    m = build_model(train, user_ids, movie_ids)

    # Linear regression baseline (user mean + movie mean)
    um, mm, g = (train.groupby("userId").rating.mean(),
                 train.groupby("movieId").rating.mean(), train.rating.mean())
    feat = lambda d: np.column_stack([d.userId.map(um).fillna(g),
                                      d.movieId.map(mm).fillna(g)])
    lin = LinearRegression().fit(feat(train), train.rating)
    r_lin = np.sqrt(mean_squared_error(
        test.rating, np.clip(lin.predict(feat(test)), 0.5, 5)))

    cache_u, cache_i, pu, pi = {}, {}, [], []
    for u, mv in zip(test.userId, test.movieId):
        ui, mi = uidx[u], midx[mv]
        if ui not in cache_u:
            cache_u[ui] = predict_user_based(m, ui)[0]
            cache_i[ui] = predict_item_based_vec(
                m, m["matrix"][ui], m["row_mean"][ui])[0]
        pu.append(cache_u[ui][mi])
        pi.append(cache_i[ui][mi])
    return {"Linear Regression": r_lin,
            "User-based KNN": np.sqrt(mean_squared_error(test.rating, pu)),
            "Item-based KNN": np.sqrt(mean_squared_error(test.rating, pi))}


@st.cache_data
def cluster_users():
    """K-Means: group users by taste, and find each group's favourite genres."""
    ratings, movies = load_data()
    m = get_model()
    labels = KMeans(n_clusters=5, n_init=10, random_state=42).fit(
        m["matrix"]).labels_
    clusters = pd.Series(labels, index=m["user_ids"])
    g = movies.assign(genre=movies.genres.str.split("|")).explode("genre")
    merged = ratings.merge(g[["movieId", "genre"]], on="movieId")
    merged["cluster"] = merged.userId.map(clusters)
    stat = merged.groupby(["cluster", "genre"]).rating.agg(["mean", "count"])
    stat = stat[stat["count"] >= 100].reset_index()
    fav = {c: ", ".join(d.sort_values("mean", ascending=False).genre[:3])
           for c, d in stat.groupby("cluster")}
    return clusters, fav


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
ratings, movies = load_data()
model = get_model()
title_to_id = dict(zip(movies.title, movies.movieId))
id_to_row = movies.set_index("movieId")
stats = ratings.groupby("movieId").rating.agg(avg_rating="mean",
                                              num_ratings="count")


def show_table(ids, scores=None, sims=None):
    df = id_to_row.loc[ids, ["title", "genres"]].copy()
    if scores is not None:
        df.insert(2, "predicted ⭐", np.round(scores, 2))
    if sims is not None:
        df.insert(2, "similarity %", np.round(sims * 100, 1))
    df = df.join(stats.round({"avg_rating": 2}))
    df.columns = [c.replace("avg_rating", "avg ⭐ (all users)")
                   .replace("num_ratings", "# ratings") for c in df.columns]
    st.dataframe(df.reset_index(drop=True),
                 hide_index=False)


def top_n(scores, support, hide_mask, n, min_support=MIN_SUPPORT):
    s = scores.copy()
    s[support < min_support] = -1
    s[hide_mask] = -1
    top = np.argsort(s)[::-1][:n]
    top = [i for i in top if s[i] > 0]
    return [model["movie_ids"][i] for i in top], s[top]


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
st.sidebar.title("🎬 Movie Recommender")
st.sidebar.caption("Collaborative Filtering using KNN")
n_rec = st.sidebar.slider("How many recommendations?", 3, 20, 8)
method = st.sidebar.radio(
    "Method", ["Item-based KNN", "User-based KNN"],
    help="Item-based: movies similar to what you liked. "
         "User-based: movies liked by users with similar taste.")
st.sidebar.markdown("---")
st.sidebar.metric("Users", len(model["user_ids"]))
st.sidebar.metric("Movies", len(model["movie_ids"]))
st.sidebar.metric("Ratings", f"{len(ratings):,}")

st.title("🎬 Movie Recommendation System")
st.write("Get movie suggestions based on what **similar users** and "
         "**similar movies** tell us. Built with KNN, K-Means and Regression.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🎯 For a user", "🎞️ Similar movies", "🆕 New user (rate movies)",
     "📊 Dashboard & models", "ℹ️ About"])

# ----------------------------------------------------------------------
# TAB 1: recommendations for an existing user
# ----------------------------------------------------------------------
with tab1:
    uid = st.selectbox("Select a user ID", model["user_ids"])
    ui = model["uidx"][uid]
    rated = model["matrix"][ui] > 0

    left, right = st.columns(2)
    with left:
        st.subheader(f"⭐ Movies user {uid} loved")
        mine = ratings[ratings.userId == uid].sort_values(
            "rating", ascending=False).head(8)
        df = id_to_row.loc[mine.movieId, ["title", "genres"]].copy()
        df["rating"] = mine.rating.values
        st.dataframe(df.reset_index(drop=True))
    with right:
        clusters, fav = cluster_users()
        c = clusters[uid]
        st.subheader("🧩 Taste group (K-Means)")
        st.info(f"User {uid} is in **group {c}**. "
                f"This group's favourite genres: **{fav.get(c, '-')}**")
        st.caption(f"User has rated {int(rated.sum())} movies.")

    st.subheader(f"✅ Recommended for user {uid}")
    if method == "Item-based KNN":
        sc, sup = predict_item_based_vec(model, model["matrix"][ui],
                                         model["row_mean"][ui])
    else:
        sc, sup = predict_user_based(model, ui)
    ids, scores = top_n(sc, sup, rated, n_rec)
    show_table(ids, scores=scores)

# ----------------------------------------------------------------------
# TAB 2: similar movies
# ----------------------------------------------------------------------
with tab2:
    default = list(movies.title).index("Toy Story (1995)") \
        if "Toy Story (1995)" in set(movies.title) else 0
    pick = st.selectbox("Pick a movie you like (type to search)",
                        sorted(movies.title),
                        index=sorted(movies.title).index(movies.title[default]))
    mi = model["midx"][title_to_id[pick]]
    st.subheader(f"If you liked **{pick}**, try these:")
    ids = [model["movie_ids"][i] for i in model["i_nbr"][mi][:n_rec]]
    show_table(ids, sims=model["i_sim"][mi][:n_rec])
    st.caption("Similarity = cosine similarity of rating patterns across users.")

# ----------------------------------------------------------------------
# TAB 3: new user (cold start solved by asking for ratings)
# ----------------------------------------------------------------------
with tab3:
    st.write("New here? Pick a few movies you've watched and rate them. "
             "We'll recommend similar ones.")
    popular = stats.sort_values("num_ratings", ascending=False).index[:300]
    options = sorted(id_to_row.loc[popular, "title"])
    chosen = st.multiselect("Movies you have watched", options,
                            default=[t for t in ["Toy Story (1995)",
                                                 "Matrix, The (1999)"]
                                     if t in options])
    vec = np.zeros(len(model["movie_ids"]))
    if chosen:
        cols = st.columns(min(len(chosen), 3))
        for k, t in enumerate(chosen):
            r = cols[k % 3].slider(t, 1.0, 5.0, 4.0, 0.5, key=f"s{k}")
            vec[model["midx"][title_to_id[t]]] = r
        fb = vec[vec > 0].mean()
        sc, sup = predict_item_based_vec(model, vec, fb)
        ids, scores = top_n(sc, sup, vec > 0, n_rec, min_support=1)
        st.subheader("✅ Your recommendations")
        if ids:
            show_table(ids, scores=scores)
        else:
            st.warning("Rate a few more movies to get recommendations.")
    else:
        st.info("Select at least one movie above.")

# ----------------------------------------------------------------------
# TAB 4: dashboard (Exp 2 visualisation + model comparison)
# ----------------------------------------------------------------------
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rating distribution")
        st.bar_chart(ratings.rating.value_counts().sort_index())
    with c2:
        st.subheader("Most rated movies")
        top = stats.sort_values("num_ratings", ascending=False).head(10)
        top.index = id_to_row.loc[top.index, "title"]
        st.bar_chart(top["num_ratings"])

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Genre popularity")
        g = movies.genres.str.split("|").explode().value_counts()
        st.bar_chart(g)
    with c4:
        st.subheader("Users per taste group (K-Means)")
        clusters, fav = cluster_users()
        st.bar_chart(clusters.value_counts().sort_index())

    st.subheader("Model comparison (RMSE - lower is better)")
    with st.spinner("Evaluating models..."):
        res = evaluate()
    st.bar_chart(pd.Series(res, name="RMSE"))
    st.table(pd.DataFrame({"RMSE": res}).round(3))
    st.caption("RMSE = average size of the prediction error in stars.")

# ----------------------------------------------------------------------
# TAB 5: about
# ----------------------------------------------------------------------
with tab5:
    st.markdown("""
**How it works**
1. Build a **user-item matrix** (rows = users, columns = movies, values = ratings).
2. **User-based KNN:** find the 10 users with the most similar taste (cosine similarity)
   and average their ratings for movies you haven't seen.
3. **Item-based KNN:** find movies similar to the ones you rated highly.
4. **K-Means:** groups users into 5 taste clusters.
5. **Linear Regression:** baseline that predicts from user and movie average ratings.

**Syllabus mapping:** Exp 1 (Python, NumPy) | Exp 2 (Pandas, visualisation) |
Exp 4 (Regression) | Exp 8 (KNN) | Exp 10 (K-Means) | Exp 11 (Recommender System)

**Dataset:** MovieLens ml-latest-small (GroupLens Research).
""")
