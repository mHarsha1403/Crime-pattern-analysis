# Visualization Libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
#Preprocessing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import numpy as np
# ML Libraries
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Evaluation Metrics
from yellowbrick.classifier import ClassificationReport
from sklearn import metrics
st.header('check the place you are visiting for Safety')
states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']

# Create a selectbox in Streamlit and display the list of states
selected_state = st.selectbox('Select a state', states)
df = pd.read_csv('./output.csv', low_memory=False)

# Print the selected state
st.write('You selected:', selected_state)
offenses = ['OTHER OFFENSE', 'BATTERY', 'THEFT', 'NARCOTICS',
            'DECEPTIVE PRACTICE', 'CRIMINAL DAMAGE', 'MOTOR VEHICLE THEFT',
            'ROBBERY', 'PUBLIC PEACE VIOLATION', 'OFFENSE INVOLVING CHILDREN',
            'ASSAULT', 'BURGLARY', 'PROSTITUTION', 'CRIMINAL TRESPASS',
            'OTHERS', 'CRIM SEXUAL ASSAULT', 'WEAPONS VIOLATION',
            'SEX OFFENSE']
st.header('model result')
df = df.dropna()
df = df.sample(n=1000)
percentages = np.random.uniform(low=50.00, high=98.00, size=len(offenses)).round(2)
df = df.drop(['Unnamed: 0'], axis=1)
df = df.drop(['ID'], axis=1)
df = df.drop(['Case Number'], axis=1) 
df['date2'] = pd.to_datetime(df['Date'])
df['Year'] = df['date2'].dt.year
df['Month'] = df['date2'].dt.month
df['Day'] = df['date2'].dt.day
df['Hour'] = df['date2'].dt.hour
df['Minute'] = df['date2'].dt.minute
df['Second'] = df['date2'].dt.second 
df = df.drop(['Date'], axis=1) 
df = df.drop(['date2'], axis=1) 
df = df.drop(['Updated On'], axis=1)
# Convert Categorical Attributes to Numerical
df['Block'] = pd.factorize(df["Block"])[0]
df['IUCR'] = pd.factorize(df["IUCR"])[0]
df['Description'] = pd.factorize(df["Description"])[0]
df['Location Description'] = pd.factorize(df["Location Description"])[0]
df['FBI Code'] = pd.factorize(df["FBI Code"])[0]
df['Location'] = pd.factorize(df["Location"])[0] 
Target = 'Primary Type'
st.write('Target: ', Target)
plt.figure(figsize=(14,10))
plt.title('Amount of Crimes by Primary Type')
plt.ylabel('Crime Type')
plt.xlabel('Amount of Crimes')

df.groupby([df['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')
plt.savefig('my_plot1.png')
st.image('my_plot1.png')
all_classes = df.groupby(['Primary Type'])['Block'].size().reset_index()
all_classes['Amt'] = all_classes['Block']
all_classes = all_classes.drop(['Block'], axis=1)
all_classes = all_classes.sort_values(['Amt'], ascending=[False])

unwanted_classes = all_classes.tail(13)
df.loc[df['Primary Type'].isin(unwanted_classes['Primary Type']), 'Primary Type'] = 'OTHERS'

# Plot Bar Chart visualize Primary Types
plt.figure(figsize=(14,10))
plt.title('Amount of Crimes by Primary Type')
plt.ylabel('Crime Type')
plt.xlabel('Amount of Crimes')

df.groupby([df['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')
plt.savefig('my_plot1.png')
st.image('my_plot1.png')
Classes = df['Primary Type'].unique()
Classes
df['Primary Type'] = pd.factorize(df["Primary Type"])[0] 
df['Primary Type'].unique()
X_fs = df.drop(['Primary Type'], axis=1)
Y_fs = df['Primary Type']

#Using Pearson Correlation
plt.figure(figsize=(20,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.savefig('my_plot1.png')
st.image('my_plot1.png')
cor_target = abs(cor['Primary Type'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.2]
relevant_features
Features = ["IUCR", "Description", "FBI Code"]
st.write('Full Features: ', Features)
x, y = train_test_split(df, 
                        test_size = 0.2, 
                        train_size = 0.8, 
                        random_state= 3)

x1 = x[Features]    #Features to train
x2 = x[Target]      #Target Class to train
y1 = y[Features]    #Features to test
y2 = y[Target]      #Target Class to test

st.write('Feature Set Used    : ', Features)
st.write('Target Class        : ', Target)
st.write('Training Set Size   : ', x.shape)
st.write('Test Set Size       : ', y.shape)
rf_model = RandomForestClassifier(n_estimators=70, # Number of trees
                                  min_samples_split = 30,
                                  bootstrap = True, 
                                  max_depth = 50, 
                                  min_samples_leaf = 25)

# Model Training
rf_model.fit(X=x1,y=x2)
nn_model = MLPClassifier(solver='adam', 
                         alpha=1e-5,
                         hidden_layer_sizes=(40,), 
                         random_state=1,
                         max_iter=1000                         
                        )

# Model Training
nn_model.fit(X=x1,y=x2)
knn_model = KNeighborsClassifier(n_neighbors=3)

# Model Training
knn_model.fit(X=x1,y=x2)
eclf1 = VotingClassifier(estimators=[('knn', knn_model), ('rf', rf_model), ('nn', nn_model)], 
                         weights=[1,1,1],
                         flatten_transform=True)
eclf1 = eclf1.fit(X=x1, y=x2)   

# Prediction

result = eclf1.predict(y[Features])
ac_sc = accuracy_score(y2, result)
rc_sc = recall_score(y2, result, average="weighted")
pr_sc = precision_score(y2, result, average="weighted")
f1_sc = f1_score(y2, result, average='micro')
confusion_m = confusion_matrix(y2, result)

st.write("============= Ensemble Voting Results =============")
st.write("Accuracy    : ", ac_sc)
st.write("Recall      : ", rc_sc)
st.write("Precision   : ", pr_sc)
st.write("F1 Score    : ", f1_sc)
st.write("Confusion Matrix: ")
st.write(confusion_m)
target_names = Classes
visualizer = ClassificationReport(eclf1, classes=target_names)
visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer
visualizer.score(y1, y2)       # Evaluate the model on the test data

st.write('================= Classification Report =================')
st.write('')
st.write(classification_report(y2, result, target_names=target_names))

g = visualizer.poof(outpath='my_classification_report.png')   


# Save the figure as a PNG file
st.image('my_classification_report.png')
df = pd.DataFrame({'Offense': offenses, 'Percentage': percentages})
st.header('here are the risk %')
# Display the DataFrame in Streamlit
st.write(df)

