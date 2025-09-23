import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="IVE Korea 광고 효율 예측기", layout="wide")

# ✅ 캐싱 (모델은 실행할 때마다 다시 안 불러옴)
@st.cache_resource
def load_model():
    return joblib.load("randomforest_model.pkl")

model = load_model()

# ===== 헤더 (로고 + 제목 일자 정렬) =====
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="https://raw.githubusercontent.com/woojingg/ad_steamlit/main/아이브로고.png" 
             width="100" style="margin-right: 20px;">
        <h1 style="margin: 0;">아이브 광고 효율 예측기</h1>
    </div>
    """,
    unsafe_allow_html=True
)


# ===== 입력 폼 =====
with st.form("input_form"):
    st.subheader("📌 광고 입력값")
    mda_input = st.text_input("매체번호(mda_idx) 입력 (쉼표로 다수 입력 가능)", "342,396")
    adv_cost = st.number_input("광고단가 (adv_cost)", min_value=0, value=1000)
    ads_type = st.selectbox("광고 타입 (ads_type)", list(range(13)), index=2)
    ads_category = st.selectbox("광고 카테고리 (ads_category)", list(range(13)), index=2)
    submitted = st.form_submit_button("예측하기")

# ===== 예측 결과 =====
if submitted:
    try:
        mda_list = [int(x.strip()) for x in mda_input.split(",") if x.strip().isdigit()]
        new_data = pd.DataFrame(
            [[ads_type, ads_category, mda, adv_cost] for mda in mda_list],
            columns=['ads_type','ads_category','mda_idx','adv_cost']
        )
        y_pred = model.predict(new_data)
        y_prob = model.predict_proba(new_data)[:, 1]

        result_df = pd.DataFrame({
            "매체번호": [str(m) for m in mda_list],  # 매체번호를 문자열로 변환
            "예측": ["✅ 효율" if y == 1 else "❌ 비효율" for y in y_pred],
            "효율 확률(raw)": y_prob  # 정렬용 float
        })

        # 효율 확률 기준 내림차순 정렬
        result_df = result_df.sort_values("효율 확률(raw)", ascending=False).reset_index(drop=True)

        # 순위 추가
        result_df["순위"] = range(1, len(result_df) + 1)

        # 표시용 퍼센트 컬럼 추가
        result_df["효율 확률"] = result_df["효율 확률(raw)"].map(lambda x: f"{x:.2%}")

        # 원하는 컬럼 순서로 출력
        st.subheader("📊 효율 순위 결과")
        st.dataframe(result_df[["순위", "매체번호", "예측", "효율 확률"]], use_container_width=True)
      
        
    except Exception as e:
        st.error(f"입력 오류: {e}")



















