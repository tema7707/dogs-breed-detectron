import streamlit as st
import numpy as np
import pandas as pd
import scipy
import keras
import tensorflow.keras as ks
import requests
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from keras import models
from plotnine import *
from scipy.stats import pearsonr
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

"""
# Итоговый проект
Анализируем информацию о породах собак
"""
df = pd.read_csv('dogs.csv')
BREEDS = ['boxer', 'doberman', 'husky', 'labrador', 'ovcharka', 'pitbull']

def getInfoAboutADog(img):
    with open("input.jpg", "wb") as f:
        f.write(img.read())

    # predict breed (https://github.com/dabasajay/Image-Caption-Generator/blob/master/test.py)
    orig_image = load_img('./input.jpg', target_size=(255, 255))
    image = img_to_array(orig_image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    prediction = model.predict(image)
    # end of copypaste
    predicted_label = BREEDS[np.argmax(prediction[0])]
    
    # plot predictions
    st.write(predicted_label)
    st.image(orig_image)
    
    # print info
    st.write('Порода:                  ' + predicted_label)
    st.write('Страна:                  ' + str(df[df['Breed'] == predicted_label]['Country'].values[0]))
    st.write('Происхождение:           ' + str(df[df['Breed'] == predicted_label]['Origin'].values[0]))
    st.write('Вес:                     ' + str(df[df['Breed'] == predicted_label]['Weight'].values[0]))
    st.write('Высота:                  ' + str(df[df['Breed'] == predicted_label]['Height'].values[0]))
    st.write('Продолжительность жизни: ' + str(df[df['Breed'] == predicted_label]['Life expectancy'].values[0]))


weights = []
for i in df['Weight']:
    try:
        weights.append(float(i))
    except:
        weights.append(None)        
df['Weight'] = weights

heights = []
for i in df['Height']:
    try:
        heights.append(float(i))
    except:
        heights.append(None)        
df['Height'] = heights

expectancy = []
for i in df['Life expectancy']:
    try:
        expectancy.append(float(i))
    except:
        expectancy.append(None)        
df['Life expectancy'] = expectancy

df['Height'] = df['Height'].fillna(-1)
df['Weight'] = df['Weight'].fillna(-1)
df['Life expectancy'] = df['Life expectancy'].fillna(-1)


MODS = ['Общая информация о датасете', ' Информация о конкретной породе', 'Распознать породу']

mode = st.sidebar.selectbox(
    'Выберете режим',
     MODS)

if mode == MODS[0]:
	st.write(df)
	"""
	## Продолжительность жизни
	Посмотрим на среднюю продолжительность жизни собак. Видно, что большинсво собак живет от 10 до 14 лет. Причем, очень мало собак доживает до 15 лет :(
	"""
	st.write((
	ggplot(df) +
	geom_histogram(
	    aes(x = 'Life expectancy'),
	    color = 'green',
	    fill = 'pink'
	) + 
	labs(
	    title ='Distribution of Life expectancy'
	) +
	scale_x_continuous(
	    limits = (0, df['Life expectancy'].max()+3)
	)
	).draw())
	"""
	## Вес и высота собак
	Посмотрим на взаимоотношение веса и высоты собаки. Видно, что высокие собаки весят больше, чем низкие, что логично. Заметим, что вес собак > 60 см значительно больше, чем вес низких собак.
	"""
	st.write((
	    ggplot(df[df['Weight'] > -1].dropna(subset = ['Height'])) +
	    geom_point(
		aes(x = 'Height',
		    y = 'Weight'),
		fill = 'yellow', color = 'blue'
	    ) +
	    labs(
		title ='Relationship between heights and weights of dogs',
		x = 'Height in cm',
		y = 'Weights in kg',
	    ) + 
	    scale_x_continuous(
		limits = (0, df['Height'].max()+3)
	    ) + 
	    scale_y_continuous(
		limits = (0, df['Weight'].max()+3)
	    )
	).draw())
	"""
	## Страна и высота собаки
	Посмотрим на взаимоотношение стран и высоты собак. Тут мы видим, что в Германии находятся самые высокие породы собак, а, вот, в Великобритании почти все собаки маленькие.
	"""
	st.write((
	    ggplot(df[df['Height'] > -1].dropna(subset = ['Height'])) +
	    geom_bin2d(
		aes(x = 'Height',
		    y = 'Country')
	    ) +
	    labs(
		title ='Relationship between height and country',
		x = 'Height in cm',
		y = 'Country',
	    ) +
	    theme(figure_size = (8, 8))
	).draw())
	"""
	### Используя scipy посмотрим на кореляцию признаков
	- Видно, что, чем тяжалее собака, тем она выше;
	- Можно сказать, что, страна происхождения породы влияет на высоту
	- Вес собаки почти не связан со страной происхождения породы
	"""
	st.write('Коэффициент Пирсона между высотой и весом:             	   {}'.format(pearsonr(df['Height'], df['Weight'])[0]))
	st.write('Коэффициент Пирсона между высотой и продолжительностью жизни:    {}'.format(pearsonr(df['Height'], df['Life expectancy'])[0]))
	st.write('Коэффициент Пирсона между весом и продолжительностью жизни:      {}'.format(pearsonr(df['Life expectancy'], df['Weight'])[0]))

elif mode == MODS[1]:
	option = st.selectbox(
	    'Порода',
	     df['Breed'])
	
	st.write('Порода: ' + option)
	st.write('Страна: ' + df[df['Breed'] == option]['Country'].values[0])
	st.write('Происхождение: ' + str(df[df['Breed'] == option]['Origin'].values[0]))
	st.write('Вес: ' + str(df[df['Breed'] == option]['Weight'].values[0]))
	st.write('Высота: ' + str(df[df['Breed'] == option]['Height'].values[0]))
	st.write('Продолжительность жизни: ' + str(df[df['Breed'] == option]['Life expectancy'].values[0]))

else:
	model = ks.models.load_model('dogs_calss.h5')
	uploaded_file = st.file_uploader("Choose a JPG file", type="jpg")
	if uploaded_file is not None:
		getInfoAboutADog(uploaded_file)

