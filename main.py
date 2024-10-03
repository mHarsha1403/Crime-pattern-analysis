import streamlit as st
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = 25,8
from IPython.core.display import HTML
sns.set()
import random

from warnings import simplefilter
simplefilter("ignore")
import os

import numpy as np # linear algebra
import pandas as pd
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from plotly.offline import download_plotlyjs, init_notebook_mode , plot,iplot
import plotly.express as px
import plotly.graph_objects as go


from plotly.colors import n_colors
from plotly.subplots import make_subplots
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()
import base64
import streamlit as st
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('./bg.jpg')
victims = pd.read_csv('C:/Users/mylav/OneDrive/Desktop/CRIMEANALYSIS/crime/20_Victims_of_rape.csv')
police_hr = pd.read_csv('C:/Users/mylav/OneDrive/Desktop/CRIMEANALYSIS/crime/35_Human_rights_violation_by_police.csv')
auto_theft = pd.read_csv('C:/Users/mylav/OneDrive/Desktop/CRIMEANALYSIS/crime/30_Auto_theft.csv')
prop_theft = pd.read_csv('C:/Users/mylav/OneDrive/Desktop/CRIMEANALYSIS/crime/10_Property_stolen_and_recovered.csv')
#crime_onwoman = pd.read_csv('C:/Users/mylav/OneDrive/Desktop/CRIMEANALYSIS/crime/42_Cases_under_crime_against_women.csv')
district_crime = pd.read_csv('C:/Users/mylav/OneDrive/Desktop/CRIMEANALYSIS/crime/crime/01_District_wise_crimes_committed_IPC_2001_2012.csv')
district_crimeonsc = pd.read_csv('C:/Users/mylav/OneDrive/Desktop/CRIMEANALYSIS/crime/crime/02_01_District_wise_crimes_committed_against_SC_2001_2012.csv')
district_crimeonst = pd.read_csv('C:/Users/mylav/OneDrive/Desktop/CRIMEANALYSIS/crime/crime/02_District_wise_crimes_committed_against_ST_2001_2012.csv')

st.title("CRIME ANALYSIS")
st.write('What kind of info you are looking for')

input=st.text_input('Enter Your Query Here')
button_clicked = st.button("Check crime in  your State")

# Check if the button is clicked
if button_clicked:
    os.system('streamlit run app2.py')
my_list = ['rape', 'harassment', 'human rights', 'torture', 'extortion','atrocities','arrest','fake encounter','false implication','property stolen','property','stolen','auto','auto theft','death','killer','murder']
penalties = {
    'rape': 'Imprisonment for 7 years to life and fine',
    'harassment': 'Imprisonment up to 3 years and/or fine',
    'human rights': 'Imprisonment up to 7 years and/or fine',
    'torture': 'Imprisonment up to 10 years and/or fine',
    'extortion': 'Imprisonment up to 3 years and/or fine',
    'atrocities': 'Imprisonment up to 10 years and/or fine',
    'arrests': 'Imprisonment up to 3 years and/or fine',
    'fake encounter': 'Life imprisonment',
    'false implication': 'Imprisonment up to 7 years and/or fine'
}
for item in my_list:
  if item in input.lower():
    if item == 'rape'or item == 'harassment' :
        st.write(victims)
        st.header('VICTIMS OF INCEST RAPE')
        rape_victims= victims[victims['Subgroup']=='Victims of Incest Rape']
        st.write(rape_victims)
        g= pd.DataFrame(rape_victims.groupby(['Year'])['Rape_Cases_Reported'].sum().reset_index())
        st.header('YEAR WISE CASES')
        st.write(g)
        fig= px.bar(g,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['blue'])
        st.plotly_chart(fig)
        st.header('AREA WISE CASES')
        g1= pd.DataFrame(rape_victims.groupby(['Area_Name'])['Rape_Cases_Reported'].sum().reset_index())
        g1.replace(to_replace='Arunachal Pradesh',value='Arunanchal Pradesh',inplace=True)
        st.write(g1)
        g1.columns=['State/UT','Cases Reported']
        shp_gdf = gpd.read_file('C:/Users/mylav/OneDrive/Desktop/CRIMEANALYSIS/map/India States/Indian_states.shp')
        merge =shp_gdf.set_index('st_nm').join(g1.set_index('State/UT'))
        fig,ax=plt.subplots(1, figsize=(10,10))

        ax.set_title('State-wise Rape-Cases Reported (2001-2010)',
                    fontdict={'fontsize': '15', 'fontweight' : '3'})
        fig = merge.plot(column='Cases Reported', cmap='Reds', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)
        plt.savefig('my_plot.png')
        st.header('INTENSITY MAP')
        st.image('my_plot.png')
        above_50 = rape_victims['Victims_Above_50_Yrs'].sum()
        ten_to_14 = rape_victims['Victims_Between_10-14_Yrs'].sum()
        fourteen_to_18 = rape_victims['Victims_Between_14-18_Yrs'].sum()
        eighteen_to_30 = rape_victims['Victims_Between_18-30_Yrs'].sum()
        thirty_to_50 = rape_victims['Victims_Between_30-50_Yrs'].sum()
        upto_10 = rape_victims['Victims_Upto_10_Yrs'].sum()
        age_grp = ['Upto 10','10 to 14','14 to 18','18 to 30','30 to 50','Above 50']
        age_group_vals = [upto_10,ten_to_14,fourteen_to_18,eighteen_to_30,thirty_to_50,above_50]

        fig = go.Figure(data=[go.Pie(labels=age_grp, values=age_group_vals,sort=True,
                                    marker=dict(colors=px.colors.qualitative.G10),textfont_size=12)])
        fig.write_image("pl2.png")
        st.header('AGE GROUPS')
        st.image('pl2.png')
        st.header('Penalties')
        st.write(penalties.get(item))

    elif item =='human rights'  or item =='torture'  or item =='extortion'  or item =='atrocities'  or item =='arrest'  or item =='fake encounter'  or item =='false implication' :
        x=item
        st.header(x.upper()+' CRIME')
        g2= pd.DataFrame(police_hr.groupby(['Area_Name'])['Cases_Registered_under_Human_Rights_Violations'].sum().reset_index())
        st.write(x)
        st.write(g2)
        st.header('YEAR WISE CASES')
        g3 = pd.DataFrame(police_hr.groupby(['Year'])['Cases_Registered_under_Human_Rights_Violations'].sum().reset_index())
        g3.columns = ['Year','Cases Registered']

        fig = px.bar(g3,x='Year',y='Cases Registered',color_discrete_sequence=['black'])
        st.plotly_chart(fig)
        st.header('GROUPING')
        st.write(police_hr.Group_Name.value_counts())
        st.header(x+'POLICE REPORT')
        g4 = pd.DataFrame(police_hr.groupby(['Year'])[['Policemen_Chargesheeted','Policemen_Convicted']].sum().reset_index())
        st.write(g4)
        year=['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010']

        fig = go.Figure(data=[
            go.Bar(name='Policemen Chargesheeted', x=year, y=g4['Policemen_Chargesheeted'],
                  marker_color='purple'),
            go.Bar(name='Policemen Convicted', x=year, y=g4['Policemen_Convicted'],
                  marker_color='red')
        ])

        fig.update_layout(barmode='group',xaxis_title='Year',yaxis_title='Number of policemen')
        st.plotly_chart(fig)
        st.header(x+'STATE WISE REPORTS')
        g2.columns= ['State/UT','Cases Reported']
        st.write(g2)
        g2.replace(to_replace='Arunachal Pradesh',value='Arunanchal Pradesh',inplace=True)
        colormaps = ['RdPu', 'viridis', 'coolwarm', 'Blues', 'Greens', 'Reds', 'PuOr', 'inferno', 'magma', 'cividis', 'cool', 'hot', 'YlOrRd', 'YlGnBu']

        random_cmap = random.choice(colormaps)
        shp_gdf = gpd.read_file('C:/Users/mylav/OneDrive/Desktop/CRIMEANALYSIS/map/India States/Indian_states.shp')
        merged = shp_gdf.set_index('st_nm').join(g2.set_index('State/UT'))
        st.write(shp_gdf)
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.axis('off')
        ax.set_title('State-wise '+x+' Cases Reported',
                    fontdict={'fontsize': '15', 'fontweight' : '3'})
        fig = merged.plot(column='Cases Reported', cmap=random_cmap, linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)
        plt.savefig('my_plot.png')
        st.header('INTENSITY MAP')
        st.image('my_plot.png')
        st.header('Penalties')
        st.write(penalties.get(item))
    elif item =='property' or item =='property stolen' or item =='stolen'or item =='Burglary':
        df = pd.read_csv('C:/Users/mylav/OneDrive/Desktop/CRIMEANALYSIS/crime/10_Property_stolen_and_recovered.csv')
        stats = df.describe()
        st.write(stats)
        plt.bar(['Recovered', 'Stolen'], [df['Cases_Property_Recovered'][0], df['Cases_Property_Stolen'][0]])
        plt.title('Cases of Property Recovered and Stolen')
        plt.xlabel('Type of Property')
        plt.ylabel('Number of Cases')
        plt.savefig('my_plot.png')
        st.image('my_plot.png')
        labels = ['Recovered', 'Stolen']
        sizes = [df['Value_of_Property_Recovered'][0], df['Value_of_Property_Stolen'][0]]
        colors = ['green', 'red']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title('Property Recovered and Stolen')
        plt.axis('equal')
        plt.savefig('my_plot.png')
        st.image('my_plot.png')
        group_data = df.groupby('Group_Name').agg({'Cases_Property_Recovered': 'sum', 'Cases_Property_Stolen': 'sum'})
        group_data.plot(kind='bar')
        plt.title('Cases of Property Recovered and Stolen by Group Name')
        plt.xlabel('Group Name')
        plt.ylabel('Number of Cases')
        plt.savefig('my_plot.png')
        st.image('my_plot.png')
        cases_by_area_year = df.pivot_table(values=['Cases_Property_Recovered', 'Cases_Property_Stolen'], index='Area_Name', columns='Year', aggfunc='sum')
        st.write(cases_by_area_year)

        
        plt.scatter(df['Value_of_Property_Recovered'], df['Value_of_Property_Stolen'])
        plt.title('Value of Property Recovered vs. Stolen')
        plt.xlabel('Value of Property Recovered')
        plt.ylabel('Value of Property Stolen')
        plt.savefig('my_plot.png')
        st.image('my_plot.png')
        top_stolen = df.sort_values(by='Cases_Property_Stolen', ascending=False).head(5)[['Sub_Group_Name', 'Cases_Property_Stolen']]
        top_stolen.rename(columns={'Sub_Group_Name': 'Sub-group', 'Cases_Property_Stolen': 'Number of Cases Stolen'}, inplace=True)
        top_stolen.reset_index(drop=True, inplace=True)
        top_stolen.index += 1
        st.write(top_stolen)

        sub_group_cases = df[['Sub_Group_Name', 'Cases_Property_Stolen']].copy()
        sub_group_cases.set_index('Sub_Group_Name', inplace=True)
        st.write(sub_group_cases)
        plt.hist([df['Value_of_Property_Recovered'], df['Value_of_Property_Stolen']], bins=5, label=['Recovered', 'Stolen'])
        plt.title('Value of Property Recovered and Stolen')
        plt.xlabel('Value of Property')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('my_plot.png')
        st.image('my_plot.png')
        year_data = df.groupby('Year').agg({'Cases_Property_Recovered': 'sum', 'Cases_Property_Stolen': 'sum'})
        year_data.plot(kind='bar')
        plt.title('Cases of Property Recovered and Stolen by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Cases')
        plt.savefig('my_plot.png')
        st.image('my_plot.png')
        summary_stats = df[['Cases_Property_Recovered', 'Cases_Property_Stolen']].describe().round(2)
        summary_stats.rename(columns={'Cases_Property_Recovered': 'Recovered Cases', 'Cases_Property_Stolen': 'Stolen Cases'}, inplace=True)
        st.write(summary_stats)
    elif item =='auto' or item == 'auto theft':
        g5 = pd.DataFrame(auto_theft.groupby(['Area_Name'])['Auto_Theft_Stolen'].sum().reset_index())
        st.write(g5)
        g5.columns = ['State/UT','Vehicle_Stolen']
        g5.replace(to_replace='Arunachal Pradesh',value='Arunanchal Pradesh',inplace=True)

        shp_gdf = gpd.read_file('C:/Users/mylav/OneDrive/Desktop/CRIMEANALYSIS/map/India States/Indian_states.shp')
        merged = shp_gdf.set_index('st_nm').join(g5.set_index('State/UT'))

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.axis('off')
        ax.set_title('State-wise Auto Theft Cases Reported(2001-2010)',
                    fontdict={'fontsize': '15', 'fontweight' : '3'})
        fig = merged.plot(column='Vehicle_Stolen', cmap='YlOrBr', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)
        plt.savefig('my_plot.png')
        st.image('my_plot.png')
        auto_theft_traced = auto_theft['Auto_Theft_Coordinated/Traced'].sum()
        auto_theft_recovered = auto_theft['Auto_Theft_Recovered'].sum()
        auto_theft_stolen = auto_theft['Auto_Theft_Stolen'].sum()

        vehicle_group = ['Vehicles Stolen','Vehicles Traced','Vehicles Recovered']
        vehicle_vals = [auto_theft_stolen,auto_theft_traced,auto_theft_recovered]

        colors = ['hotpink','purple','red']

        fig = go.Figure(data=[go.Pie(labels=vehicle_group, values=vehicle_vals,sort=False,marker=dict(colors=colors),textfont_size=12)])

        st.plotly_chart(fig)
        g5 = pd.DataFrame(auto_theft.groupby(['Year'])['Auto_Theft_Stolen'].sum().reset_index())

        g5.columns = ['Year','Vehicles Stolen']

        fig = px.bar(g5,x='Year',y='Vehicles Stolen',color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig)
        vehicle_list = ['Motor Cycles/ Scooters','Motor Car/Taxi/Jeep','Buses',
               'Goods carrying vehicles (Trucks/Tempo etc)','Other Motor vehicles']

        sr_no = [1,2,3,4,5]

        fig = go.Figure(data=[go.Table(header=dict(values=['Sr No','Vehicle type'],
                                                  fill_color='turquoise',
                                                  height=30),
                        cells=dict(values=[sr_no,vehicle_list],
                                    height=30))
                            ])
        st.plotly_chart(fig)
        motor_c = auto_theft[auto_theft['Sub_Group_Name']=='1. Motor Cycles/ Scooters']

        g8 = pd.DataFrame(motor_c.groupby(['Area_Name'])['Auto_Theft_Stolen'].sum().reset_index())
        g8_sorted = g8.sort_values(['Auto_Theft_Stolen'],ascending=True)
        fig = px.scatter(g8_sorted.iloc[-10:,:], y='Area_Name', x='Auto_Theft_Stolen',
                    orientation='h',color_discrete_sequence=["red"])
        st.plotly_chart(fig)
    elif item=='murder' or item=='killer' or item=='death' or item=='homicide' or item=='fatalities':
        murder = pd.read_csv('C:/Users/mylav/OneDrive/Desktop/CRIMEANALYSIS/crime/32_Murder_victim_age_sex.csv')
        st.write(murder.Year.unique())
        murder.Area_Name.unique()
        murder.Sub_Group_Name.unique()
        st.write(murder.head(10))
        url = "https://flo.uri.sh/visualisation/2693755/embed"

        # Render the HTML content in the Streamlit app
        st.components.v1.iframe(url, height=500)
        murdert = murder[murder['Sub_Group_Name']== '3. Total']  #keeping only total category of subgroup
        murdery = murdert.groupby(['Year'])['Victims_Total'].sum().reset_index() #grouping
        sns.set_context("talk")
        plt.style.use("fivethirtyeight")
        plt.figure(figsize = (14,10))
        #sns.palplot(sns.color_palette("hls", 8))
        ax = sns.barplot(x = 'Year' , y = 'Victims_Total' , data = murdery ,palette= 'dark') #plotting bar graph
        plt.title("Total Victims of Murder per Year")
        ax.set_ylabel('')
        for p in ax.patches:
                    ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),
                        textcoords='offset points')
        plt.savefig('my_plot.png')
        st.image('my_plot.png')
        murderg = murder.groupby(['Year' , 'Sub_Group_Name'])['Victims_Total'].sum().reset_index() # grouping with year and sub group
        murderg = murderg[murderg['Sub_Group_Name']!= '3. Total']   # we dont need total category of sub group

        plt.style.use("fivethirtyeight")
        plt.figure(figsize = (14,10))
        ax = sns.barplot( x = 'Year', y = 'Victims_Total' , hue = 'Sub_Group_Name' , data = murderg ,palette= 'bright') #plotting barplot
        plt.title('Gender Distribution of Victims per Year',size = 20)
        ax.set_ylabel('')
        plt.savefig('my_plot.png')
        st.image('my_plot.png')
        
        murderg = murder.groupby(['Year' , 'Sub_Group_Name'])['Victims_Total'].sum().reset_index() # grouping with year and sub group
        murderg = murderg[murderg['Sub_Group_Name']!= '3. Total']   # we dont need total category of sub group

        plt.style.use("fivethirtyeight")
        plt.figure(figsize = (14,10))
        ax = sns.barplot( x = 'Year', y = 'Victims_Total' , hue = 'Sub_Group_Name' , data = murderg ,palette= 'bright') #plotting barplot
        plt.title('Gender Distribution of Victims per Year',size = 20)
        ax.set_ylabel('')
        plt.savefig('my_plot.png')
        st.image('my_plot.png')
        
        murdera = murder.groupby(['Year'])[['Victims_Upto_10_15_Yrs','Victims_Above_50_Yrs',
                                          'Victims_Upto_10_Yrs', 'Victims_Upto_15_18_Yrs',
                                          'Victims_Upto_18_30_Yrs','Victims_Upto_30_50_Yrs']].sum().reset_index()  #grouby year and age group
        murdera = murdera.melt('Year', var_name='AgeGroup',  value_name='vals') #melting the dataset

        plt.style.use("fivethirtyeight")
        plt.figure(figsize = (14,10))
        ax = sns.barplot(x = 'Year' , y = 'vals',hue = 'AgeGroup' ,data = murdera ,palette= 'bright') #plotting a bar
        plt.title('Age Distribution of Victims per Year',size = 20)
        ax.get_legend().set_bbox_to_anchor((1, 1)) #anchoring the labels so that they dont show up on the graph
        ax.set_ylabel('')
        plt.savefig('my_plot.png')
        st.image('my_plot.png')
        murderag = murder.groupby(['Sub_Group_Name'])[['Victims_Upto_10_15_Yrs',
                                              'Victims_Above_50_Yrs', 'Victims_Upto_10_Yrs',
                                              'Victims_Upto_15_18_Yrs','Victims_Upto_18_30_Yrs',
                                              'Victims_Upto_30_50_Yrs',]].sum().reset_index()       #grouping with the gender and age groups

        murderag = murderag.melt('Sub_Group_Name', var_name='AgeGroup',  value_name='vals')  #melting the dataset for drawing the desired plot
        murderag= murderag[murderag['Sub_Group_Name']!= '3. Total']

        plt.style.use("fivethirtyeight")
        plt.figure(figsize = (14,10))
        ax = sns.barplot(x = 'Sub_Group_Name' , y = 'vals',hue = 'AgeGroup' ,data = murderag,palette= 'colorblind') #making barplot taking Agegroup as hue/category 
        plt.title('Age & Gender Distribution of Victims',size = 20)
        ax.get_legend().set_bbox_to_anchor((1, 1)) #using anchor so that legend doesn't show on the graph
        ax.set_ylabel('')
        ax.set_xlabel('Victims Gender')
        for p in ax.patches:
                    ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),
                        textcoords='offset points')
        plt.savefig('my_plot.png')
        st.image('my_plot.png')
        # murderst = murder[murder['Sub_Group_Name']== '3. Total']   #we need only total number of victims per state
        # murderst= murderst.groupby(['Area_Name'])['Victims_Total'].sum().sort_values(ascending = False).reset_index()
        # new_row = {'Area_Name':'Telangana', 'Victims_Total':27481}
        # murderst = pd.concat([murderst, new_row], ignore_index=True)
        # murderst.sort_values('Area_Name')
        # import geopandas as gpd
        # gdf = gpd.read_file('C:/Users/DELL/OneDrive/Desktop/CRIMEANALYSIS/map/India States/Indian_states.shp')
        # murderst.at[17, 'Area_Name'] = 'NCT of Delhi'
        # merged = gdf.merge(murderst, left_on='st_nm', right_on='Area_Name')
        # merged.drop(['Area_Name'], axis=1)
        # #merged.describe()
        # merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[:])
        # merged['coords'] = [coords[0] for coords in merged['coords']]


        # sns.set_context("talk")
        # sns.set_style("dark")
        # #plt.style.use('dark_background')
        # cmap = 'YlGn'
        # figsize = (25, 20)

        
        plt.savefig('my_plot.png')
        st.image('my_plot.png')

    elif st.button('check crime'):
        st.write('what crime can affect you')





        
        
