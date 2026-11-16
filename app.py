import json
import numpy as np, pandas as pd, cv2, joblib
import streamlit as st
from skimage.feature import graycomatrix, graycoprops, hog

st.set_page_config(page_title='Klasifikasi Gambar', page_icon='🔍')


# ===== Load model (sekali saja, disimpan di cache) =====
@st.cache_resource
def load_model():
    model = joblib.load('model_tulisan.pkl')
    with open('info_model.json') as f:
        info = json.load(f)
    cnn = None
    if 'cnn' in info['fitur']:        # tensorflow hanya di-load kalau modelnya pakai fitur CNN
        from tensorflow import keras
        cnn = keras.applications.EfficientNetV2B0(include_top=False, weights='imagenet',
                                                  input_shape=(info['IMG_H'], info['IMG_W'], 3), pooling='avg')
    return model, info, cnn

model, info, cnn = load_model()
IMG_H, IMG_W = info['IMG_H'], info['IMG_W']


# ===== Preprocessing & fitur (SAMA PERSIS dengan di notebook) =====
def normalize_bg(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    k = max(15, int(gray.shape[0] * 0.15)) | 1
    bg = cv2.GaussianBlur(cv2.dilate(gray, np.ones((k, k), np.uint8)), (k, k), 0)
    norm = cv2.divide(gray, bg, scale=255)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

def letterbox(img, h=IMG_H, w=IMG_W):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(int(iw * s), 1), max(int(ih * s), 1)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 255, np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas

props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
def glcm_features(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    g = graycomatrix(gray, distances=[1, 2, 3],
                     angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                     levels=256, symmetric=True, normed=True)
    return np.hstack([graycoprops(g, p).ravel() for p in props])

def hog_features(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))

def ekstrak_fitur(img):
    f = {}
    if 'cnn' in info['fitur']:  f['cnn'] = cnn.predict(img[None], verbose=0)[0]
    if 'glcm' in info['fitur']: f['glcm'] = glcm_features(img)
    if 'hog' in info['fitur']:  f['hog'] = hog_features(img)
    return np.hstack([f[k] for k in info['fitur']])[None]   # urutan harus sama dengan saat training


# ===== Tampilan =====
st.title('🔍 Klasifikasi Gambar')
st.write('Upload gambar, model akan menebak kelasnya beserta **confident value**.')

th = st.slider('Batas minimal confidence', 0.0, 1.0, 0.5, 0.05)
up = st.file_uploader('Upload gambar', type=['jpg', 'jpeg', 'png'])

if up:
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error('Gambar tidak bisa dibaca.'); st.stop()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    proc = letterbox(normalize_bg(rgb))

    st.image(rgb, caption='Gambar asli')
    st.image(proc, caption='Setelah preprocessing')

    with st.spinner('Memprediksi...'):
        proba = model.predict_proba(ekstrak_fitur(proc))[0]

    top = proba.argsort()[::-1][:5]
    kelas = [info['class_names'][model.classes_[i]] for i in top]
    conf = proba[top[0]]

    if conf >= th:
        st.success(f'Prediksi: **{kelas[0]}**')
    else:
        st.warning(f'Model kurang yakin (tebakan terdekat: **{kelas[0]}**)')
    st.metric('Confident value', f'{conf * 100:.2f}%')
    st.progress(float(conf))

    st.subheader('Top-5 prediksi')
    st.dataframe(pd.DataFrame({'Kelas': kelas, 'Confidence (%)': (proba[top] * 100).round(2)}),
                 hide_index=True)
