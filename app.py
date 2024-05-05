import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff

df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')

#Preprocessing part

def preprocess():
    global df,region_df
    df = df[df['Season'] == 'Summer']
    df = df.merge(region_df,on='NOC', how='left')
    df.drop_duplicates(inplace=True)
    df = pd.concat([df, pd.get_dummies(df['Medal'])], axis=1)
    
    df['Bronze'] = df['Bronze'].replace(False, 0)
    df['Gold'] = df['Gold'].replace(False, 0)
    df['Silver'] = df['Silver'].replace(False, 0)

    df['Bronze'] = df['Bronze'].replace(True, 1)
    df['Gold'] = df['Gold'].replace(True, 1)
    df['Silver'] = df['Silver'].replace(True, 1)

    return df

df = preprocess()

#Medal Tally Function

# def medal_tally(df):
#     medal_tally = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])

#     medal_tally = medal_tally.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending=False).reset_index()

#     medal_tally['Total'] = medal_tally['Gold'] + medal_tally['Silver'] + medal_tally['Bronze']
#     medal_tally.columns = medal_tally.columns.str.title()

#     return medal_tally

#Country and Years List Dropdown

def country_and_years_list(df):
    #for years
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0,'Overall')

    #for country's name
    country = np.unique(df['region'].dropna().tolist())
    country.sort()
    country = np.insert(country, 0, 'Overall')

    return years, country

#Fetch The Medals according to Year and Country

def fetch_medal_tally(df, year, country):
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    flag = 0
    
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    if year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]
    if year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    if year != 'Overall' and country != 'Overall':
        temp_df = medal_df[(medal_df['Year'] == int(year)) & (medal_df['region'] == country)]
        
    if flag == 1:
        x = temp_df.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year', ascending=True).reset_index()
    else:
        x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('region', ascending=True).reset_index()
        x = x.sort_values('Gold', ascending=False).reset_index(drop=True)

    x['Total'] = x['Gold'] + x['Silver'] + x['Bronze']
    x.columns = x.columns.str.title()
    
    return x

# medals = medal_tally(df) 

def some_statistics_over_time(df):
    #Nations Over Time
    nations_over_time = df.drop_duplicates(['Year', 'region'])['Year'].value_counts().reset_index().sort_values('Year')
    nations_over_time.columns = nations_over_time.columns.str.title()

    #Events Over Time
    events_over_time = df.drop_duplicates(['Year', 'Name'])['Year'].value_counts().reset_index().sort_values('Year')
    events_over_time.columns = events_over_time.columns.str.title()

    #Athletes Over Time
    athletes_over_time = df.drop_duplicates(['Year', 'Name'])['Year'].value_counts().reset_index().sort_values('Year')
    athletes_over_time.columns = athletes_over_time.columns.str.title()

    #Heatmap for The Events Happened in each sport in all olympics
    heatmap = df.drop_duplicates(['Year', 'Sport', 'Event'])

    return nations_over_time, events_over_time, athletes_over_time, heatmap

#Country Wise Analysis
def country_wise_analysis(df, country):
    temp_df = df.dropna(subset='Medal')
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    final_dataframe = new_df.groupby('Year')['Medal'].count().reset_index()

    return final_dataframe

# New Function
def heatmap_for_country(df, country):
    temp_df = df.dropna(subset='Medal')
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]

    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)

    return pt

def most_successful_countrywise(df, country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df = temp_df[temp_df['region'] == country]

    x = temp_df['Name'].value_counts().reset_index().head(10)
    
    x.rename(columns={'Name': 'Name', 'count': 'Medals'}, inplace=True)
    return x

#Weight Vs Height
def weight_v_height(df,sport):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    athlete_df['Medal'].fillna('No Medal', inplace=True)
    if sport != 'Overall':
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        return temp_df
    else:
        return athlete_df
    
#Men and Women
def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)

    final.fillna(0, inplace=True)

    return final

#Main Uer Menu
st.sidebar.header('Olympics Analyser')

user_menu = st.sidebar.radio(
    'Select an Option',
    ('Medal Tally','Overall Analysis','Country-wise Analysis','Athlete wise Analysis')
)

# st.dataframe(df)

if user_menu == "Medal Tally":
    st.sidebar.header('Medal Tally')
    years, country = country_and_years_list(df)

    selected_year = st.sidebar.selectbox('Select Year', years)
    selected_country = st.sidebar.selectbox('Select Country', country)

    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title('Overall Analysis')

    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(f'Overall Analyses for {selected_country}')

    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title(f'Overall Analyses of {selected_year} Olympics')

    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(f'Analyses of {selected_year} Olympics for {selected_country}')

    medals_tally_years = fetch_medal_tally(df, selected_year, selected_country)

    st.table(medals_tally_years)

if user_menu == 'Overall Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title('Top Statistics')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header('Editions')
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    nations_over_time, events_over_time, athletes_over_time, heatmap = some_statistics_over_time(df)

    st.header('Nations Over Time')
    fig1 = px.line(nations_over_time, x='Year', y='Count')
    st.plotly_chart(fig1)

    st.header('Events Over Time')
    fig2 = px.line(events_over_time, x='Year', y='Count')
    st.plotly_chart(fig2)

    st.header('Athletes Over Time')
    fig3 = px.line(athletes_over_time, x='Year', y='Count')
    st.plotly_chart(fig3)

    st.title("No. of Events over time(Every Sport)")
    fig,ax = plt.subplots(figsize=(20,20))
    ax = sns.heatmap(heatmap.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype(int), annot=True)
    st.pyplot(fig)

if user_menu == 'Country-wise Analysis':
    st.sidebar.header('Country Wise Analysis')
    total_countries = df['region'].dropna().unique().tolist()
    total_countries.sort()

    user_selected_country = st.sidebar.selectbox('Select a Country',total_countries)
    final_dataframe = country_wise_analysis(df, user_selected_country)

    st.header(f'{user_selected_country} Medal Tally over the years')
    fig = px.line(final_dataframe, x='Year', y='Medal')
    st.plotly_chart(fig)


    st.header(f'{user_selected_country} excels in the following sports')
    pt = heatmap_for_country(df, user_selected_country)
    fig2,ax2 = plt.subplots(figsize=(20,20))
    ax2 = sns.heatmap(pt, annot=True)
    st.pyplot(fig2)

    st.header(f'Top Ten Best Players of {user_selected_country}')
    ten_most_successfuls = most_successful_countrywise(df, user_selected_country)
    st.table(ten_most_successfuls)

if user_menu == 'Athlete wise Analysis':
    atheletes_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = atheletes_df['Age'].dropna()
    x2 = atheletes_df[atheletes_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = atheletes_df[atheletes_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = atheletes_df[atheletes_df['Medal'] == 'Bronze']['Age'].dropna()

    st.header('Age Distributions of Athletes')

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalists', 'Silver Medalists', 'Bronze Medalists'], show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=800, height=600)
    st.plotly_chart(fig)

    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    
    for sport in famous_sports:
        temp_df = atheletes_df[atheletes_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=800, height=600)
    st.title("Distribution of Age wrt Sports(Gold Medalist)")
    st.plotly_chart(fig)

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    # st.title('Height Vs Weight')
    # selected_sport = st.selectbox('Select a Sport', sport_list)
    # temp_df = weight_v_height(df,selected_sport)
    # fig,ax = plt.subplots()
    # ax = sns.scatterplot(temp_df['Weight'],temp_df['Height'],hue=temp_df['Medal'],style=temp_df['Sex'],s=60)
    # st.pyplot(fig)

    st.title("Men Vs Women Participation Over the Years")
    final = men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=800, height=600)
    st.plotly_chart(fig)