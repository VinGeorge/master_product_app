import streamlit as st
import pandas as pd
import joblib
import os
import base64
from PIL import Image

# Upload the model and image

model = open(os.path.join('model/SGDClassifier_mapper.pkl'), 'rb')
predictMasterRecord = joblib.load(model)
image = Image.open('image/cat.png')


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here

    return f'<a href="data:file/csv;base64,{b64}" download="sku_mapping.csv">Скачать файл и сказать спасибо</a>'


def main():
    # Intoduction

    st.title('Welcome to the product mapping tool')

    st.subheader(
        'Если вы (или ваш интерн) устали от очередного сопоставления продуктовых справочников, то вот вам счастье')

    st.text('Счастье работает только по нашим продуктам :(')

    # Users input

    ta_placeholder = st.empty()
    ta_default_value = ""

    # Remove users imput button

    if st.button('Remove all'):
        ta_default_value = " "

    rawtext = ta_placeholder.text_area('Через точку с запятой напишите ваши SKU и нажмите Predict',
                                       ta_default_value).split(';')

    # Prediction

    if st.button('Predict', key=None):
        # Show ВЖУХ

        st.image(image, use_column_width=True)

        # Make prediction

        model_output = predictMasterRecord.predict(rawtext)

        # Put the model output and an input text into pandas dataframe

        data = pd.DataFrame(list(zip(rawtext, model_output)), columns=['Original', 'DSS'])
        st.table(data)

        # Download file

        st.markdown(get_table_download_link(data), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
