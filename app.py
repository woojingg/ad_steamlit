import pandas as pd
import joblib
import streamlit as st

# ✅ 캐싱 (모델은 실행할 때마다 다시 안 불러옴)
@st.cache_resource
def load_model():
    return joblib.load("randomforest_model.pkl")

model = load_model()

st.set_page_config(page_title="IVE Korea 광고 효율 예측기", layout="wide")

col1, col2, col3 = st.columns([1,4,1])
with col1:
    st.write("")
with col2:
    logo_col, title_col = st.columns([1,5])
    with logo_col:
        st.image("8634f61c-6c8f-457b-9063-e9ac14b20456.png", width=60)
    with title_col:
        st.markdown("<h1 style='margin-top: 10px;'>광고 효율 예측기</h1>", unsafe_allow_html=True)
with col3:
    st.write("") 

mda_input = st.text_input("매체번호(mda_idx) 입력", "342,396")
adv_cost = st.number_input("광고단가 (adv_cost)", min_value=0, value=1000)
ads_type = st.selectbox("타입 (ads_type)", list(range(13)), index=2)
ads_category = st.selectbox("카테고리 (ads_category)", list(range(13)), index=2)

if st.button("예측하기"):
    try:
        mda_list = [int(x.strip()) for x in mda_input.split(",") if x.strip().isdigit()]
        new_data = pd.DataFrame(
            [[ads_type, ads_category, mda, adv_cost] for mda in mda_list],
            columns=['ads_type','ads_category','mda_idx','adv_cost']
        )
        y_pred = model.predict(new_data)
        y_prob = model.predict_proba(new_data)[:, 1]

        result_df = pd.DataFrame({
            "매체번호": mda_list,
            "효율(1) / 비효율(0)": y_pred,
            "효율 확률": [f"{p:.2%}" for p in y_prob]
        })
        st.dataframe(result_df, use_container_width=True)
    except Exception as e:
        st.error(f"입력 오류: {e}")


