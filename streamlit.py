import requests
from bs4 import BeautifulSoup
import csv
import os
import io
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.message import EmailMessage
from email import encoders

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px 
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.sidebar.markdown("## Enaira Anomaly Detection Model")
app_mode = st.sidebar.selectbox('Select Page',['Home','Predict_Fraud'])
if app_mode=='Home': 
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:green;padding:6px"> 
    <h3 style ="color:White;text-align:center;">Enaira Anomaly Detection Model</h3> 
    </div> 
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    # if app_mode=='Home': 
    st.title('Predict Fraudulent/Non Fraudlent Enaira Users') 
    st.markdown('Dataset :') 
#     df=pd.read_csv('enaira_frequency_validation.csv') #Read our data dataset
#     dd=df[['source_wallet_guid', 'date', 'month', 'day', 'hour', 'minute','count_trans']].copy()
#     st.write(dd.head()) 
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dd=df[['source_wallet_guid','MailGroup','kyc_status', 'bvn_flag', 'date','MailGroup_','kyc_status_', 'bvn_flag_',  'month', 'day', 'hour', 'minute','count_trans']].copy()
        
        
        dd_s=dd.head(10)
            # style
        th_props = [
          ('font-size', '18px'),
          ('text-align', 'center'),
          ('font-weight', 'bold'),
          ('color', '#6d6d6d'),
          ('background-color', '#f7ffff')
          ]

        td_props = [
          ('font-size', '15px')
          ]

        styles = [
          dict(selector="th", props=th_props),
          dict(selector="td", props=td_props)
          ]

        # table
        df2=dd_s.style.set_properties(**{'text-align': 'center'}).set_table_styles(styles)
        st.table(df2)

        st.line_chart(dd[["date", "count_trans"]].set_index("date"),width=100,height=700)
elif app_mode == 'Predict_Fraud':
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        def make_predictions(df):
            ##load model
            freq_model=pickle.load(open ('en_freq_model_new.pkl','rb'))
            ##load label encoder
            file = open("enc_v1.obj",'rb')
            enc_loaded = pickle.load(file)

            df_p=df[['MailGroup', 'kyc_status', 'bvn_flag','month', 'day', 'hour',
               'minute', 'count_trans']]

            df_p[['MailGroup_', 'kyc_status_', 'bvn_flag_']]=df_p[['MailGroup', 'kyc_status', 'bvn_flag']]

            df_u=df_p[['MailGroup_', 'kyc_status_', 'bvn_flag_','month', 'day', 'hour',
               'minute', 'count_trans']]

            col=['MailGroup_', 'kyc_status_', 'bvn_flag_']

            df_u[col]= enc_loaded.transform(df_u[col])

            predss= freq_model.predict(df_u)

            final_pred= pd.DataFrame({'source_wallet_guid':  df.source_wallet_guid,
                                       'date': df.date,
                                       'bvn_flag':df.bvn_flag,
                                       'MailGroup':df.MailGroup,
                                       'kyc_status':df.kyc_status,  
                                       'count_trans':df.count_trans,
                                       'day':df.day,
                                       'month':df.month,
                                       'hour':df.hour,
                                       'minute':df.minute,
                                       'Prediction':predss})

            th_props = [
              ('font-size', '18px'),
              ('text-align', 'center'),
              ('font-weight', 'bold'),
              ('color', '#6d6d6d'),
              ('background-color', '#f7ffff')
              ]

            td_props = [
              ('font-size', '15px')
              ]

            styles = [
              dict(selector="th", props=th_props),
              dict(selector="td", props=td_props)
              ]

            final_pred =final_pred[final_pred['Prediction']=='Fraudulent']
            final_pred=final_pred[['source_wallet_guid', 'date',
              'day', 'hour', 'minute','MailGroup','bvn_flag','kyc_status', 'count_trans', 'Prediction']]
            final_pred=final_pred.style.set_properties(**{'text-align': 'center'}).set_table_styles(styles)
            final_pred.set_properties(subset=['Prediction'],**{'background-color': 'red'})
            return final_pred 
        def send_email(user,pwd,subject):
            df_s=make_predictions(df)
            print(df_s)
            #df_s.set_properties(**{'background-color': 'red'}, subset=['Prediction'])
            try:
                df_html=df_s.hide_index().render()

                recipients=["foyelami@bluechiptech.biz","dangiwa26904@cbn.gov.ng]
#                                'raolaniyan@cbn.gov.ng',
#                                                     'amuwais@cbn.gov.ng',
#                                                     'tbadekayero@cbn.gov.ng',
#                                                     'uiisiyaku@cbn.gov.ng',
#                                                     'edetim@cbn.gov.ng',
#                                                     'aaliyu5@cbn.gov.ng',
#                                                     'raolaniyan@cbn.gov.ng',
#                                                     'bnoyekanmi@cbn.gov.ng']

                msg =MIMEMultipart('alternative')

                msg['Subject']=subject
                msg['From']=user
                msg['To']=",".join(recipients)

                html= """\
                <html>
                    <head>
                    </head>
                      <link rel='stylesheet' href='http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css'>
                       <body>
                           <p>
                           <br>Hello Ops Team</br>

                            <br>
                           The following users have been flagged for making suspicious transactions: please look into this
                           </br>
                           </br>

                           </br> 
                           </p>
                       </body>
                </html>
                """
                #html = "df_html".join((df_html,message_style))
                html += df_html
                part2 = MIMEText(html.encode('utf-8'),'html','utf-8')

                #msg.attach(dfPart2)
                msg.attach(part2)
                #df3=df_s.style.set_properties(**{'text-align': 'center'}).set_table_styles(styles)
                #st.table(df3)
                st.write(df_s)


                server=smtplib.SMTP("smtp.office365.com",587)
                server.starttls()
                server.login(user,pwd)

                server.sendmail(user, recipients, msg.as_string())
                server.close()
            

                print("Mail Sent!")

            except Exception as e:
                print(str(e))
                print("Failed to send mail")
    def main():
        send_email("foyelami@bluechiptech.biz","Goldfinch22","DataOps: Anonamly Detection System  !!!")
        print('email sent successfully')
                
    if st.button("Make Predictions"):
        if __name__ == '__main__':main()
        st.success('Mail Sent Succesffuly')
    else:
        st.write('No attached Document')
    
            #print(Result)




        
        
        
        
#         dd=df[['source_wallet_guid','MailGroup','kyc_status', 'bvn_flag', 'date','MailGroup_','kyc_status_', 'bvn_flag_',  'month', 'day', 'hour', 'minute','count_trans']].copy()

#         pickle_in = open('en_freq_model_new.pkl', 'rb') 
#         classifier = pickle.load(pickle_in)


#         dp=dd[['MailGroup_','kyc_status_', 'bvn_flag_',  'month', 'day', 'hour', 'minute','count_trans']].copy()

        
        
# if __name__=='__main__': 
#     main()



# html_temp = """ 
#             #         <div style ="background-color:green;padding:13px"> 
#             #         <h1 style ="color:White;text-align:center;">Enaira Anomaly Detection Model Prediction</h1> 
#             #         <h3 style ="color:White;text-align:center;">Here were using frequency of transaction within a period to predict a Fraudulent/Non Fraudlent Enaira Users</h3> 
#             #         </div> 
#             #         """
# #display the front end aspect
# st.markdown(html_temp, unsafe_allow_html = True) 
# if app_mode=='Home': 
#     st.title('Enaira Anomaly Prediction on Transaction Frequency') 
#     st.markdown('Dataset :') 
# #     df=pd.read_csv('enaira_frequency_validation.csv') #Read our data dataset
# #     dd=df[['source_wallet_guid', 'date', 'month', 'day', 'hour', 'minute','count_trans']].copy()
# #     st.write(dd.head()) 
#     uploaded_file = st.file_uploader("Choose a file")
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         dd=df[['source_wallet_guid','MailGroup','kyc_status', 'bvn_flag', 'date','MailGroup_','kyc_status_', 'bvn_flag_',  'month', 'day', 'hour', 'minute','count_trans']].copy()
#         #st.write(dd)
#         st.dataframe(dd.style.highlight_max(axis=0))
        
#         st.line_chart(dd[["date", "count_trans"]].set_index("date"))
        
#         # plot the time series 
# #         fig = px.line(dd, x="minute", y=["count_trans"], 
# #             title="Count of Transaction Frequency per Minute", width=1000)
# #         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# #         st.plotly_chart(fig, use_container_width=False)

#         # Add some matplotlib code !
#         fig, ax = plt.subplots()
#         df.hist(
#                     bins=8,
#                     column="count_trans",
#                     grid=False,
#                     figsize=(5, 5),
#                     color="#86bf91",
#                     zorder=2,
#                     rwidth=0.9,
#                     ax=ax,
#                   )
#         st.write(fig)

# elif app_mode == 'Predict_Fraud':
#     uploaded_file = st.file_uploader("Choose a file")
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         dd=df[['source_wallet_guid','MailGroup','kyc_status', 'bvn_flag', 'date','MailGroup_','kyc_status_', 'bvn_flag_',  'month', 'day', 'hour', 'minute','count_trans']].copy()

#         pickle_in = open('en_freq_model_new.pkl', 'rb') 
#         classifier = pickle.load(pickle_in)


#         dp=dd[['MailGroup_','kyc_status_', 'bvn_flag_',  'month', 'day', 'hour', 'minute','count_trans']].copy()

#         predss= classifier.predict(dp)
#         #pred_prob=fd_model.predict_proba(dd)
#         final_pred = pd.DataFrame({'source_wallet_guid':  dd.source_wallet_guid,
#                                'date': dd.date,
#                                'bvn_flag':dd.bvn_flag,
#                                'MailGroup':dd.MailGroup,
#                                'kyc_status':dd.kyc_status,  
#                                'count_trans':dd.count_trans,
#                                'day':dd.day,
#                                'month':dd.month,
#                                'hour':dd.hour,
#                                'minute':dd.minute,
#                                'Prediction':predss})
#         #final_pred.head()
#         st.write(final_pred)
        
        
        


#    # @st.cache()

#     # defining the function which will make the prediction using the data which the user inputs 
#     def prediction(Cnt_trans, Day, Month, Hour, Minute):   

#         # Making predictions 
#         prediction = classifier.predict( 
#             [[Cnt_trans, Minute,Hour,Day, Month]])
#         return prediction
#     # this is the main function in which we define our webpage  
#     def main():       
#         # front end elements of the web page 
#         html_temp = """ 
#         <div style ="background-color:green;padding:13px"> 
#         <h1 style ="color:White;text-align:center;">Enaira Anomaly Detection Model Prediction</h1> 
#         <h3 style ="color:White;text-align:center;">Here were using frequency of transaction within a period to predict a Fraudulent/Non Fraudlent Enaira Users</h3> 
#         </div> 
#         """

#         # display the front end aspect
#         st.markdown(html_temp, unsafe_allow_html = True) 

#         # following lines create boxes in which user can enter data required to make prediction 
#         #Transaction_Amount = st.number_input("Enter Transaction Amount")
#         Cnt_trans = st.slider('Count of Transaction', min_value=1, max_value=1000, value=18, step=1)
#         Minute = st.slider('Select Minute', min_value=1, max_value=60, value=10, step=1)
#         Hour = st.slider(' Select Hour', min_value=1, max_value=24, value=5, step=1)
#         Day = st.slider('Select Day', min_value=1, max_value=31, value=18, step=1)
#         Month = st.slider('Select Month', min_value=1, max_value=12, value=10, step=1)
#         result =""

#         # when 'Predict' is clicked, make the prediction and store it 
#         if st.button("Predict"): 
#             result = prediction(Cnt_trans, Minute,Hour,Day, Month ) 
#             st.subheader("Predicted Transaction")
#             st.success('This is a {} enaira user'.format(result))
#             #print(Result)
#     if __name__=='__main__': 
#         main()

# #@st.cache
# def load_data():
#     df = pd.read_csv("df_Wallet.csv")
#     return df
# df_wallet = load_data()

# df_wallet = df_wallet.groupby(by = ['date','dailt_tran_limit'])[['source_wallet_guid']].count().reset_index(drop=False)
# df_wallet.columns = ['date', 'dailt_tran_limit', 'count_of_wallets']


# def plot():
#     clist = df_wallet["dailt_tran_limit"].unique().tolist()

#     countries = st.multiselect("Select Tiers", clist)
#     st.header("You selected: {}".format(", ".join(countries)))

#     dfs = {country: df_wallet[df_wallet["dailt_tran_limit"] == country] for country in countries}
    
#     layout = go.Layout(
#     autosize=False,
#     width=1500,
#     height=700,

#     xaxis= go.layout.XAxis(linecolor = 'black',
#                           linewidth = 1,
#                           mirror = True),

#     yaxis= go.layout.YAxis(linecolor = 'black',
#                           linewidth = 1,
#                           mirror = True),

#     margin=go.layout.Margin(
#         l=50,
#         r=50,
#         b=100,
#         t=100,
#         pad = 4
#     )
# )

#     fig = go.Figure(layout=layout)
#     for country, df in dfs.items():
#         fig = fig.add_trace(go.Scatter(x=df_wallet["date"], y=df_wallet["count_of_wallets"], name=country))

#     st.plotly_chart(fig)


# plot()

# count_trans_day = df_wallet.groupby(by = ['date','dailt_tran_limit'])[['source_wallet_guid']].count().reset_index(drop=False)
# count_trans_day.columns = ['date', 'dailt_tran_limit', 'count_of_wallets']

# df=count_trans_day[['date','count_of_wallets']].copy()
# df = df.set_index('date') 
# st.line_chart(df)

# desired_label = st.selectbox('Filter to:', ['Above_tier2_limit', 'below_tier2_limit','below_tier3_limit','Above_tier3_limit'])
# st.write(count_trans_day[count_trans_day.dailt_tran_limit == desired_label])

# st.sidebar.markdown("### Select Date ")
# x_axis = st.sidebar.selectbox("Choose Option", count_trans_day[count_trans_day.dailt_tran_limit == desired_label])



# desired_label = st.selectbox('Filter to:', ['Above_tier2_limit', 'Above_tier3_limit','below_tier3_limit','Above_tier3_limit'])
# st.write(df_wallet[df_wallet.dailt_tran_limit == desired_label])

#measurements = df_wallet.columns.tolist()

# x_axis = st.sidebar.selectbox("X-Axis", measurements)
# y_axis = st.sidebar.selectbox("Y-Axis", measurements, index=1)

# bar_axis = st.sidebar.multiselect(label="Average Measures per Tumor Type Bar Chart",
#                                   options=measurements,
#                                   default=["source_wallet_guid","date","tier_level",'dailt_tran_limit',"AMOUNT_Cleaned"])

# if x_axis and y_axis:
#     scatter_fig = plt.figure(figsize=(6,4))
#     scatter_ax = scatter_fig.add_subplot(111)
#     malignant_df = df_wallet[df_wallet["dailt_tran_limit"] == "Above_tier2_limit"]
#     benign_df = df_wallet[df_wallet["dailt_tran_limit"] == "Above_tier3_limit"]

#     malignant_df.plot.scatter(x=x_axis, y=y_axis, s=120, c="tomato", alpha=0.6, ax=scatter_ax, label="Above_tier2_limit")
#     benign_df.plot.scatter(x=x_axis, y=y_axis, s=120, c="dodgerblue", alpha=0.6, ax=scatter_ax,
#                            title="{} vs {}".format(x_axis.capitalize(), y_axis.capitalize()), label="Above_tier3_limit");
    
# ##################### Layout Application ##################
# container1 = st.container()
# col1, col2 = st.columns(2)

# with container1:
#     with col1:
#         scatter_fig
#     with col2:
#         bar_fig


# container2 = st.container()
# col3, col4 = st.columns(2)

# with container2:
#     with col3:
#         hist_fig
#     with col4:
#         hexbin_fig


# ax = df_wallet["dailt_tran_limit"].value_counts().plot(kind="bar", color="darkcyan", figsize=[15, 10])
# plt.xticks(rotation=0, horizontalalignment="center", fontsize=20)
# plt.ylabel("Count", fontsize=20)
# plt.xlabel("Hour of the day", fontsize=20)
# plt.title("Count of Transaction with Tier Limits",fontsize=20)
# for p in ax.patches:
#     ax.annotate(
#         str(p.get_height()), xy=(p.get_x() + 0.25, p.get_height() + 0.1), fontsize=15
#     )

# #df = pd.DataFrame({"one": [1, 2, 3], "two": [4, 5, 6], "three": [7, 8, 9]})
# st.write(ax)

# # import numpy as np
# # import matplotlib.pyplot as plt

# # x = np.arange(0, 5, 0.1)
# # y = np.sin(x)
# # a=plt.plot(x, y)

# # st.write(a)

