import streamlit as st
import pandas as pd
import app_predict
import app_identify
import app_charts

modelpath = 'models/'

st.markdown('# RecSquared Recipe Exploration')
st.text('A place to learn about your favorite recipes and explore new ones')

# tab1 = st.tabs(['Recipe Information'])
col1, col2 = st.columns(2, gap='small')
test_title = col1.text_input('Enter A Dish Name: ')
col2.button('Upload a picture [placeholder]')

# predict Ingredients based on input title
ings = app_predict.predict_ingredients(test_title)
ings.remove('Other')
col1.markdown('### Predicted Ingredients')
ingdf = app_charts.ingredient_list(ings)
col1.altair_chart(ingdf, use_container_width=True)

# predict cuisine based on input ingredients
w2v_cuisines, lda_topics = app_identify.identify(ings)
col2.markdown('### Predicted Cuisine Type')

### create charts
## pie chart
identify_type = col2.radio("Select Type", ('Food Type', 'Ethnicity'))
if identify_type == 'Food Type':
    piechart = app_charts.pie_chart(lda_topics)
elif identify_type == 'Ethnicity':
    piechart = app_charts.pie_chart(w2v_cuisines)
col2.altair_chart(piechart)


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

st.text('Shift-click multiple labels in the legend\nClick in chart to reset')
if st.checkbox('Show words', value=True):
    st.altair_chart(combined, use_container_width=True)
else:
    st.altair_chart(combined_notext)