import streamlit as st
import pandas as pd
from PIL import Image
import app_predict
import app_identify
import app_charts

modelpath = 'models/'

logo = Image.open(modelpath+'rec2logo_v2.png')

st.set_page_config(
    page_title="Rec^2", page_icon=logo, layout='centered',
)
title_col1, title_col2 = st.columns([1,4])
title_col1.image(logo, width=125)
title_col2.markdown('# Recipe Exploration')
# st.markdown('# ![](logo) RecSquared Recipe Exploration')
title_col2.markdown('A place to learn about your favorite recipes and explore new ones')

# tab1 = st.tabs(['Recipe Information'])
entry_col1, mid, entry_col2 = st.columns([2,1,2], gap='small')
test_title = entry_col1.text_input('Enter A Dish Name: ')
mid.markdown('## or')
entry_col2.text('')
entry_col2.button('Upload a picture [placeholder]')

# this decorator + function makes it so the top charts don't regenerate when changing
# from food type to ethnicity or number of tokens in t-SNE chart
@st.experimental_memo
def create_topcharts(test_title):
    # predict Ingredients based on input title
    ings = app_predict.predict_ingredients(test_title)
    ings.remove('Other')
    ingdf = app_charts.ingredient_list(ings)

    # predict cuisine based on input ingredients
    w2v_cuisines, lda_topics = app_identify.identify(ings)
    ldachart = app_charts.pie_chart(lda_topics)
    w2vchart = app_charts.pie_chart(w2v_cuisines)
    return ings, ingdf, ldachart, w2vchart


if test_title != '':
    with st.spinner('running'):
        # get outputs from above function
        ings, ingdf, ldachart, w2vchart = create_topcharts(test_title)

        # ingredient chart
        ingredients_col, pie_col = st.columns([1,1],gap='small')        
        ingredients_col.markdown('### Predicted Ingredients')
        ingredients_col.altair_chart(ingdf, use_container_width=True)

        ## pie chart
        pie_col.markdown('### Predicted Cuisine Type')
        identify_type = pie_col.radio("Select Type", ('Food Type', 'Ethnicity')) # radio button
        if identify_type == 'Food Type':
            piechart = ldachart
        elif identify_type == 'Ethnicity':
            piechart = w2vchart
        pie_col.altair_chart(piechart)

    with st.spinner('running'):
        ## t-SNE plot
        # load pre calculated df for t-SNE with n top words
        @st.cache
        def load_df(n_topwords):
            return pd.read_csv(modelpath+'tsne_plot/tsnedf_{}.csv'.format(n_topwords))
        n_topwords = st.slider('Select Number of Top Words per Label',5,30,15, step=5)
        df = load_df(n_topwords)

        # create df for predicted recipe
        recipedf = app_identify.df_plot(test_title,ings,n_topwords)
        combined, combined_notext = app_charts.tSNE_chart(df, recipedf)

        st.markdown('Shift-click multiple labels in the legend')
        st.markdown('Click in chart to reset')
        if st.checkbox('Show words', value=True):
            st.altair_chart(combined, use_container_width=True)
        else:
            st.altair_chart(combined_notext)