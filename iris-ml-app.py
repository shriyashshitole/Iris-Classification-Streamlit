import streamlit as st
from PIL import Image 
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier 

st.markdown("""
# Simple Iris Flower Prediction App
         
This app predicts the **Iris Flower** type!         
""")

iris = datasets.load_iris()

#irisdf = pd.DataFrame(iris.data)
#st.write(irisdf.describe())
#st.write(iris)


st.sidebar.title('User Input Features')


def user_input_features():
    sepal_length = st.sidebar.slider('sepal length', 4.3, 7.9, 6.10)
    sepal_width = st.sidebar.slider('sepal width', 2.0, 4.4, 3.2)
    petal_length = st.sidebar.slider('petal length', 1.0, 6.9, 3.95)
    petal_width = st.sidebar.slider('petal width', 0.1, 2.5, 1.25)
    data = {"sepal_length":sepal_length,
            "sepal_width":sepal_width,
            "petal_length":petal_length,
            "petal_width":petal_width}
    features = pd.DataFrame(data, index=['Values Chosen:'])
    return features

df = user_input_features()

st.subheader("User inputed parameters")
st.write(df)
st.write("""***""")

# Features -> X, Target -> Y

X = iris.data
Y = iris.target

# Fitting the classifier

clf = RandomForestClassifier()     #clf = classifier(Initiated RandomForestClassifier on a variable, variable -> object)
clf.fit(X, Y)

# Prediction and Prediction Probability

prediction = clf.predict(df)
predict_prob = clf.predict_proba(df)

classes = pd.DataFrame(iris.target_names, columns=["Flower"])

expander = st.expander('Class labels and their corresponding index number')

expander.write(classes)

st.write("""***""")

st.subheader(f'Prediction: Index: {prediction[0]} -> Flower: **{iris.target_names[prediction][0]}**')

st.write("""***""")

st.write(f"""
### Probalility of setosa is {predict_prob[0][0]} \n
### Probalility of veriscolor is {predict_prob[0][1]} \n
### Probalility of virginica is {predict_prob[0][2]} \n
""")

st.write("""***""")


if iris.target_names[prediction][0] == "setosa":
    col1, col2 = st.columns([3,5])
    col1.image(Image.open("setosa.png"))
    col2.write('Iris setosa, the bristle-pointed iris, is a species of flowering plant in the genus Iris of the family Iridaceae. It belongs the subgenus Limniris and the series Tripetalae. It is a rhizomatous perennial from a wide range across the Arctic sea, including Alaska, Maine, Canada (including British Columbia, Newfoundland, Quebec and Yukon), Russia (including Siberia), northeastern Asia, China, Korea, and southward to Japan.')
if iris.target_names[prediction][0] == "veriscolor":
    col1, col2 = st.columns([3,5])
    col1.image(Image.open("veriscolor.png"))
    col2.write('Iris versicolor or Iris versicolour is also commonly known as the blue flag, harlequin blueflag, larger blue flag, northern blue flag, and poison flag, plus other variations of these names, and in Great Britain and Ireland as purple iris.')
if iris.target_names[prediction][0] == "virginica":
    col1, col2 = st.columns([3,5])
    col1.image(Image.open("virginica.png"))
    col2.write('Iris virginica, with the common name Virginia blueflag, Virginia iris, great blue flag, or southern blue flag, is a perennial species of flowering plant in the Iridaceae (iris) family, native to central and eastern North America.')
else:
    col1, col2, col3 = st.columns([1,1,1])
    col1.write('Setosa')
    col2.write('Veriscolor')
    col3.write('Virginica')
    col1.image(Image.open("setosa.png"))
    col2.image(Image.open("veriscolor.png"))
    col3.image(Image.open("virginica.png"))