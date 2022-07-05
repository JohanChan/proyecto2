from sklearn.tree import DecisionTreeClassifier, plot_tree
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

uploaded_file = None
dataframe = None
operacion = ''

def upload():
    st.markdown(f"# {list(page_names_to_funcs.keys())[0]}")
    global uploaded_file, dataframe, operacion

    uploaded_file = st.file_uploader("Elije un archivo")
    #dataframe = None
    if uploaded_file is not None:
        extension = uploaded_file.name.split('.')[1]
        if extension == 'csv':
            dataframe = pd.read_csv(uploaded_file)
            #st.write(dataframe)
        elif extension == 'json':
            dataframe = pd.read_json(uploaded_file)
            #st.write(dataframe)
        else:
            dataframe = pd.read_excel(uploaded_file)
            #st.write(dataframe)
    else:
        st.write('Archivo nulo')
    
    if dataframe is not None:
        st.write(dataframe)
        option = st.selectbox(
        'Elije una opcion',
        ('Regresion Lineal', 'Regresion Polinomial', 'Clasificador Gaussiano','Clasificador de arboles de Decision','Redes neuronales'))
        operacion = st.selectbox(
        'Elije una operación',
        ('Graficar Puntos', 'Funcion de Tendencia', 'Realizar prediccion'))
        if option == 'Regresion Lineal':
            st.markdown(f"# Regresión Lineal")
            lineal()
        elif option == 'Regresion Polinomial':
            st.markdown(f"# Regresión Polinomial")
            polinomial()
        elif option == 'Clasificador Gaussiano': 
            gauss()
        elif option == 'Clasificador de arboles de Decision':
            arbol()

def lineal():
    global dataframe, operacion
    inputx = st.text_input('Eje X','')
    inputy = st.text_input('Eje Y','')
    inputpred = st.text_input('Prediccion','')
    if inputy != '' and inputx != '' and inputpred != '':
        x = np.asarray(dataframe[inputx]).reshape(-1,1)
        y = dataframe[inputy]
       
        regresion = linear_model.LinearRegression()
        regresion.fit(x,y)
        if operacion == 'Graficar Puntos':
            figura, ejes = plt.subplots()
            ejes.scatter(x,y, color='blue')
            plt.title('Regresión Lineal')
            plt.xlabel(inputx)
            plt.ylabel(inputy)
            plt.grid()
            st.pyplot(figura)
        elif operacion == 'Realizar prediccion':
            st.write('Prediccion',regresion.predict([[int(inputpred)]]))
        elif operacion == 'Funcion de Tendencia':
            a = regresion.coef_
            b = regresion.intercept_
            ecuacion = 'y = '+str(a[0])+'x + '+str(b)
            st.write('Funcion de Tendencia',ecuacion)

def polinomial():
    global dataframe, operacion
    inputx = st.text_input('Eje X','')
    inputy = st.text_input('Eje Y','')
    inputpred = st.text_input('Prediccion','')
    inputgrado = st.text_input('Grado','')

    if inputx != '' and inputy != '' and inputpred != '' and inputgrado != '':
        x = dataframe[inputx]
        y = dataframe[inputy]

        x = np.asarray(x)
        y = np.asarray(y)

        x = x[:,np.newaxis]
        y = y[:,np.newaxis]

        pol = PolynomialFeatures(degree=int(inputgrado))
        xT = pol.fit_transform(x)
        modelo = linear_model.LinearRegression()
        modelo.fit(xT,y)
     
        if operacion == 'Graficar Puntos':
            figura, ejes = plt.subplots()
            ejes.scatter(x,y, color='blue')
            plt.title('Regresión Polinomial')
            plt.xlabel(inputx)
            plt.ylabel(inputy)
            plt.grid()
            st.pyplot(figura)
        elif operacion == 'Realizar prediccion':
            xMinimo = 0
            xMax = int(inputpred)

            X = np.linspace(xMinimo,xMax,100)
            X = X[:,np.newaxis]

            XT = pol.fit_transform(X)
            Y = modelo.predict(XT)
            st.write('Predicción',Y[Y.size-1])
        elif operacion == 'Funcion de Tendencia':
            a = modelo.coef_
            b = modelo.intercept_
            texto = str(b[0])+' + '
            indice = 1
            contador = 1
            for n in a:
                for m in n:
                    if m != 0:
                        if contador == len(n):
                            texto += str(m)+'x^'+str(indice)
                        else:
                            texto += str(m)+'x^'+str(indice)+' + '
                            indice = indice + 1
                    contador = contador + 1
            st.write(texto)

def gauss():
    global dataframe
    columna = st.text_input('Ingrese columna','')
    prediccion = st.text_input('Ingrese '+str(len(dataframe.columns.values)-1)+' valores separados por comas','')
    codificado = st.checkbox('Label encoder?')
    label = ''
    prueba = ''
    if columna != '' and prediccion != '':
        if codificado:
            encoders = list()
            le = preprocessing.LabelEncoder()
            for valor in dataframe.columns.values:
                literal = dataframe[valor]
                encoder = le.fit_transform(literal)
                if columna != valor:  
                    encoders.append(encoder)
                else:
                    label = encoder
        
            features = list()
            l = len(encoders[0])
            for i in range(l):
                aux = list()
                for encoder in encoders:
                    aux.append(encoder[i])
                features.append(aux)
            features = np.asarray(features)

            model = GaussianNB()
            model.fit(features,label)
            prediccion = np.fromstring(prediccion,dtype=int,sep=',')

            predicted = model.predict([prediccion])
            st.write('Prediccion',predicted[0])
        else:
            listado = list()
            for valor in dataframe:
                if columna != valor:
                    tupla = tuple(dataframe[valor])
                    listado.append(tupla)
                else:
                    label = dataframe[valor]
            model = GaussianNB()
            model.fit(listado,label)
            prediccion = np.fromstring(prediccion,dtype=int,sep=',')

            predicted = model.predict([prediccion])
            st.write('Prediccion',predicted[0])

def arbol():
    global dataframe
    columna = st.text_input('Ingrese columna','')
    codificado = st.checkbox('Label encoder?')
    label = ''
    if columna != '':
        if codificado:
            encoders = list()
            le = preprocessing.LabelEncoder()
            for valor in dataframe.columns.values:
                literal = dataframe[valor]
                encoder = le.fit_transform(literal)
                if columna != valor:   
                    encoders.append(encoder)
                else:
                    label = encoder
            #st.write(label)
        
            features = list()
            l = len(encoders[0])
            for i in range(l):
                aux = list()
                for encoder in encoders:
                    aux.append(encoder[i])
                features.append(aux)
            tree = DecisionTreeClassifier().fit(features,label)
            figura, eje = plt.subplots()
            plot_tree(tree,filled=True)
            plt.figure(figsize=(100,100))
            st.pyplot(figura)
        else:
            listado = list()
            for valor in dataframe:
                if columna != valor:
                    tupla = tuple(dataframe[valor])
                    listado.append(tupla)
                else:
                    label = dataframe[valor]
            tree = DecisionTreeClassifier().fit(listado,label)
            figura, eje = plt.subplots()
            plot_tree(tree,filled=True)
            plt.figure(figsize=(100,100))
            st.pyplot(figura)

page_names_to_funcs = {
    "Data Science": upload,
}

demo_name = st.sidebar.selectbox("Johan Leonel Chan Toledo 201603052", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()