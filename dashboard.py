import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import shap
import streamlit.components.v1 as components
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def choose_feature(option):
    if option == 'CODE_GENDER':
        fig = px.pie(values=df.CODE_GENDER.value_counts(), names=['Male','Female'],title='CODE_GENDER distribution')
        st.plotly_chart(fig)
    elif option == 'TARGET':
        fig = px.pie(values=df.TARGET.value_counts(), names=['can pay loans', 'cannot pay loans'],title='TARGET DISTRIBUTION')
        st.plotly_chart(fig)
    elif option == 'NAME_CONTRACT_TYPE':
        fig = px.pie(values=df.NAME_CONTRACT_TYPE.value_counts(), names=['cash loan','revolving loan'],title='Contract types distribution')
        plt.yticks(ticks=np.arange(2),labels=['Single',' married'])
        st.plotly_chart(fig)
    elif option == 'AMT_CREDIT':
        fig, ax = plt.subplots()
        #ax=sns.histplot(data=df, x="AMT_CREDIT",hue='TARGET')
        ax=sns.kdeplot(df.AMT_CREDIT)
        plt.axvline(df[(df.SK_ID_CURR==index)].AMT_CREDIT.item(),ymin=0, ymax=1,linestyle='--')
        plt.axvline(df.AMT_CREDIT.mean(),ymin=0, ymax=1,linestyle='--',color='red')
        plt.title('Credit amount distribution')
        st.pyplot(fig)
    elif option =='FAMILY STATUS':
        # Distribution of lenders by family status
        fig=sns.catplot(data=df, y='NAME_FAMILY_STATUS',hue='TARGET', kind='count')
        plt.xticks(rotation=60)
        plt.yticks(ticks=np.arange(6), labels=['Single',' married','civil marriage','widow','seperated','unknown'])
        plt.title('Family Status')
        st.pyplot(fig)
    elif option =='Borrowwer Occupations':
        fig, ax = plt.subplots()
        ax=sns.countplot(data=df, y='OCCUPATION_TYPE')
        st.pyplot(fig)
    elif option == 'Age':
        fig, ax = plt.subplots()
        # ax=sns.histplot(data=df, x="AMT_CREDIT",hue='TARGET')
        ax = sns.kdeplot(abs(df.DAYS_BIRTH)/365)
        plt.axvline(abs(df[(df.SK_ID_CURR == index)].DAYS_BIRTH.item())/365, ymin=0, ymax=1, linestyle='--')
        plt.axvline(abs(df.DAYS_BIRTH.mean())/365, ymin=0, ymax=1, linestyle='--', color='red')
        plt.title('Age distribution')
        st.pyplot(fig)



url = 'https://drive.google.com/file/d/1pWtKj13nsroHEzb7eWA_dv2UUUkn5Q-T/view?usp=sharing'
path1 = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
path = r'C:\Users\ROUA\OneDrive\Bureau\openclassrooms\P7\HomeCredit\application_train.csv'

@st.cache
def load_df():
    url = 'https://drive.google.com/file/d/1fK0EPuQys4fxwe50FBnZ175JvMQf4lrs/view?usp=sharing'
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url, on_bad_lines='skip')
    return df

df=load_df()
st.title('Home Credit')
st.subheader('General information about all clients')
st.sidebar.image('./Home-Credit-logo.png')
st.sidebar.title("Select client_ID")
global prediction
prediction=0
with st.form(key='my_form'):
    with st.sidebar:
        index = st.sidebar.number_input("ID", min_value=100002, max_value=10000000, key="index")
        requested_client = df[df.SK_ID_CURR == index]

        submitted = st.form_submit_button("Predict")
        session = requests.Session()
        if submitted:
            st.write("Result")
            response = requests.request(method='get', url=f'https://oc-implement-scoring-model.herokuapp.com/predict/{index}')
            prediction =response.json()["prediction"]
            st.write(prediction)


        option = st.selectbox('Choose a feature to show its distribution',('Age','CODE_GENDER','NAME_CONTRACT_TYPE','AMT_CREDIT','TARGET','FAMILY STATUS','Borrowwer Occupations'))
        choice = st.form_submit_button("choose")
        session = requests.Session()
if choice:
    choose_feature(option=option)

####indicator code
fig1 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prediction,
    number= {'font': {'color': 'green'}},
    domain = {'x': [0, 1], 'y': [0, 1]},
    gauge={
        'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
        #'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 0.5], 'color': 'green'},
            {'range': [0.5, 1], 'color': 'red'}],
    },
    title = {'text': "prediction"}))

fig2 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prediction,
    number= {'font': {'color': 'red'}},
    domain = {'x': [0, 1], 'y': [0, 1]},
    gauge={
        'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
        #'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 0.5], 'color': 'green'},
            {'range': [0.5, 1], 'color': 'red'}],
    },
    title = {'text': "prediction"}))
if prediction!=0:
    if prediction<0.5 :
        st.plotly_chart(fig1)
        st.subheader('**Model predicted 0 -> credit accepted**')
        st.balloons()
    else:
        st.plotly_chart(fig2)
        st.subheader('**Model predicted 1 -> credit not accepted**')

if st.sidebar.checkbox('Show general informations dataframe'):
    st.dataframe(df.head(5))


###############SHAP related code##################
print(df[df.SK_ID_CURR == index].AMT_CREDIT)

url_X_train='https://drive.google.com/file/d/1zBxY6cvnEvvqn7kb5OcB2t0NYhOiHsHw/view?usp=sharing'
url_X_train ='https://drive.google.com/uc?id=' + url_X_train.split('/')[-2]
X_train =pd.read_csv(url_X_train,on_bad_lines='skip')
print(df.shape)
print(X_train.shape)
print(df.columns)
model = lgb.Booster(model_file='model.txt')
model.params['objective'] = 'binary'
explainerModel = shap.TreeExplainer(model)

shap_values_Model = explainerModel.shap_values(df.drop(columns=['SK_ID_CURR', 'TARGET', 'Unnamed: 0']))

id = df.index[df.SK_ID_CURR == index]
p = shap.force_plot(explainerModel.expected_value[0], shap_values_Model[0][id],
                    df.iloc[id].drop(columns=['SK_ID_CURR', 'TARGET', 'Unnamed: 0']), matplotlib=False)

if st.sidebar.checkbox('Explain model for client ID'):
    st.subheader('Model Prediction Interpretation Plot')
    st_shap(p)

shap.initjs()

if st.sidebar.checkbox('SHAP feature importance'):
    shap_values = shap.TreeExplainer(model).shap_values(X_train.drop(columns=['Unnamed: 0']))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values[0], X_train.drop(columns=['Unnamed: 0']), plot_type='dot', show=False)
    st.pyplot(fig)



