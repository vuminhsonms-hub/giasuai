import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from openai import OpenAI
import os

# ========================
# API KEY
# ========================
api_key = os.getenv("OPENAI_API_KEY")

client = None
if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.warning("⚠️ Chưa có API key → AI sẽ không hoạt động")

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="Gia sư Vật lí AI PRO", layout="wide")

# ========================
# STYLE
# ========================
st.markdown("""
<style>
.stButton button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ========================
# TITLE
# ========================
st.title("🔬 Gia sư Vật lí AI – Hỗ trợ học tập và thí nghiệm Vật lí thông minh")
st.write("Hệ thống học tập + phòng thí nghiệm Vật lí AI dành cho học sinh THPT")

# ========================
# MEMORY
# ========================
if "history" not in st.session_state:
    st.session_state.history = []

# ========================
# AI FUNCTION
# ========================
def ask_ai(messages):
    if client is None:
        return "⚠️ Chưa cấu hình API key"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Lỗi API: {str(e)}"

# ========================
# TABS
# ========================
tabs = st.tabs([
    "🤖 Hỏi đáp 12",
    "🧠 Giải bài",
    "📝 Trắc nghiệm",
    "🔬 Phòng thí nghiệm AI",
    "🧪 Mô phỏng",
    "📝 Chấm bài",
    "📚 Công thức",
    "📜 Lịch sử"
])

# ========================
# TAB 1: HỎI ĐÁP
# ========================
with tabs[0]:
    question = st.text_area("Nhập câu hỏi")

    if st.button("AI trả lời", key="ask_btn"):
        if question.strip():  # kiểm tra câu hỏi không rỗng
            answer = ask_ai([
                {"role": "system",
                 "content": """
                 Bạn là gia sư vật lí.
                 Nếu có công thức:
                 - Viết dạng $...$
                 - Không dùng \( \) hoặc [ ]
                 """},
                {"role": "user", "content": question}
            ])

            # Khởi tạo history nếu chưa có
            if "history" not in st.session_state:
                st.session_state.history = []

            # Lưu câu hỏi + đáp án
            st.session_state.history.append({"question": question, "answer": answer})

            st.markdown("**AI trả lời:**")
            st.markdown(answer)



# ========================
# TAB 2: GIẢI BÀI (ĐÃ FIX LỖI HIỂN THỊ & LỊCH SỬ)
# ========================
with tabs[1]:
    problem = st.text_area("Nhập bài tập", key="input_problem_tab2")

    col1, col2, col3 = st.columns(3)
    # Khởi tạo prompt_ai để tránh trùng tên với biến hệ thống
    prompt_ai = None

    if col1.button("💡 Gợi ý", key="hint_btn"):
        prompt_ai = f"Gợi ý cách làm: {problem}"
    if col2.button("🧩 Bước 1", key="step1_btn"):
        prompt_ai = f"Giải bước đầu tiên: {problem}"
    if col3.button("✅ Giải đầy đủ", key="full_btn"):
        prompt_ai = f"Giải chi tiết có công thức: {problem}"

    if prompt_ai and problem.strip():
        # Áp dụng bộ quy tắc nghiêm ngặt giống hệt Tab Hỏi đáp
        answer = ask_ai([
            {"role": "system", "content": """
                 Bạn là gia sư vật lí chuyên nghiệp.
                 Quy tắc trình bày công thức:
                 - Dùng $...$ cho công thức nằm cùng dòng.
                 - Dùng $$...$$ cho công thức cần xuống dòng riêng biệt.
                 - TUYỆT ĐỐI KHÔNG dùng ký hiệu \[ \] hoặc \( \) hoặc dấu ngoặc vuông đơn lẻ [ ] để bao quanh công thức.
                 """},
            {"role": "user", "content": prompt_ai}
        ])

        # Bước bảo hiểm cuối cùng: Ép định dạng bằng code (nếu AI lỡ quên)
        # Thay thế các biến thể ngoặc vuông/ngoặc đơn thành dấu $
        clean_answer = answer.replace(r"\[", "$$").replace(r"\]", "$$").replace(r"\(", "$").replace(r"\)", "$")
        
        # Lưu vào history theo đúng định dạng dict để Tab 8 đọc được 
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({"question": problem, "answer": clean_answer})

        # Hiển thị kết quả sạch
        st.markdown(clean_answer)

# ========================
# TAB 3: TRẮC NGHIỆM (GIAO DIỆN ĐẸP)
# ========================
with tabs[2]:
    import re

    st.markdown("""
    <style>
    .quiz-wrap {
        padding: 8px 0 4px 0;
    }
    .quiz-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px 18px 8px 18px;
        margin-bottom: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    }
    .quiz-title {
        font-size: 24px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 14px;
    }
    .quiz-meta {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 6px;
    }
    .quiz-result-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .quiz-correct {
        color: #15803d;
        font-weight: 600;
    }
    .quiz-wrong {
        color: #b91c1c;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="quiz-wrap">', unsafe_allow_html=True)
    st.subheader("📝 Tạo câu hỏi trắc nghiệm")

    topic = st.text_input("Chủ đề", placeholder="Ví dụ: Tụ điện, dòng điện xoay chiều, định luật Ohm...")
    number = st.slider("Số câu", 1, 10, 5)

    col_a, col_b = st.columns([1, 3])
    with col_a:
        create_quiz = st.button("Tạo đề", key="quiz_btn", use_container_width=True)

    if create_quiz:
        if topic.strip():
            prompt = f"""
Tạo {number} câu trắc nghiệm vật lí về chủ đề: {topic}

Yêu cầu định dạng đúng chính xác như sau:
Câu 1: Nội dung câu hỏi
A. Nội dung đáp án A
B. Nội dung đáp án B
C. Nội dung đáp án C
D. Nội dung đáp án D
Đáp án: A
Giải thích: Nội dung giải thích

Câu 2: Nội dung câu hỏi
A. Nội dung đáp án A
B. Nội dung đáp án B
C. Nội dung đáp án C
D. Nội dung đáp án D
Đáp án: B
Giải thích: Nội dung giải thích

Không thêm mở đầu.
Không thêm kết luận.
Không dùng markdown như ** hoặc -.
"""

            result = ask_ai([
                {
                    "role": "system",
                    "content": "Bạn là giáo viên vật lí. Hãy tạo đề trắc nghiệm đúng định dạng người dùng yêu cầu."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ])

            st.session_state.quiz_text = result

            # reset các lựa chọn cũ khi tạo đề mới
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith("quiz_")]
            for k in keys_to_delete:
                del st.session_state[k]

            if "quiz_submitted" in st.session_state:
                del st.session_state["quiz_submitted"]
        else:
            st.warning("Vui lòng nhập chủ đề trước khi tạo đề.")

    if "quiz_text" in st.session_state:
        text = st.session_state.quiz_text.strip()

        # Tách từng câu hỏi theo mẫu "Câu số:"
        questions = re.split(r"\n(?=Câu\s+\d+\s*:)", text)
        questions = [q.strip() for q in questions if q.strip()]

        parsed_questions = []

        for q in questions:
            lines = [line.strip() for line in q.split("\n") if line.strip()]
            if len(lines) < 7:
                continue

            first_line = lines[0]
            question_text = re.sub(r"^Câu\s+\d+\s*:\s*", "", first_line).strip()

            option_a = re.sub(r"^A\.\s*", "", lines[1]).strip()
            option_b = re.sub(r"^B\.\s*", "", lines[2]).strip()
            option_c = re.sub(r"^C\.\s*", "", lines[3]).strip()
            option_d = re.sub(r"^D\.\s*", "", lines[4]).strip()

            correct = re.sub(r"^Đáp án:\s*", "", lines[5]).strip().upper()
            explain = re.sub(r"^Giải thích:\s*", "", lines[6]).strip()

            parsed_questions.append({
                "question": question_text,
                "options": {
                    "A": option_a,
                    "B": option_b,
                    "C": option_c,
                    "D": option_d
                },
                "correct": correct,
                "explain": explain
            })

        if parsed_questions:
            st.markdown(f"""
            <div class="quiz-meta">
                Đã tạo <b>{len(parsed_questions)}</b> câu hỏi cho chủ đề: <b>{topic}</b>
            </div>
            """, unsafe_allow_html=True)

            for i, q in enumerate(parsed_questions):
                st.markdown('<div class="quiz-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="quiz-title">Câu {i+1}: {q["question"]}</div>', unsafe_allow_html=True)

                option_labels = [
                    f"A. {q['options']['A']}",
                    f"B. {q['options']['B']}",
                    f"C. {q['options']['C']}",
                    f"D. {q['options']['D']}",
                ]

                st.radio(
                    "Chọn đáp án",
                    option_labels,
                    index=None,
                    key=f"quiz_{i}",
                    label_visibility="visible"
                )

                st.markdown("</div>", unsafe_allow_html=True)

            col_submit, col_empty = st.columns([1, 3])
            with col_submit:
                submit_quiz = st.button("Nộp bài", key="submit_quiz", use_container_width=True)

            if submit_quiz:
                st.session_state.quiz_submitted = True

            if st.session_state.get("quiz_submitted", False):
                score = 0

                for i, q in enumerate(parsed_questions):
                    selected = st.session_state.get(f"quiz_{i}")
                    if selected is None:
                        continue

                    selected_letter = selected.split(".")[0].strip().upper()
                    if selected_letter == q["correct"]:
                        score += 1

                st.success(f"🎯 Điểm của bạn: {score}/{len(parsed_questions)}")

                st.markdown("### Đáp án và giải thích")

                for i, q in enumerate(parsed_questions):
                    selected = st.session_state.get(f"quiz_{i}")

                    if selected is None:
                        selected_letter = "Chưa chọn"
                        result_class = "quiz-wrong"
                        result_text = "Bạn chưa chọn đáp án"
                    else:
                        selected_letter = selected.split(".")[0].strip().upper()
                        if selected_letter == q["correct"]:
                            result_class = "quiz-correct"
                            result_text = "Bạn làm đúng"
                        else:
                            result_class = "quiz-wrong"
                            result_text = "Bạn làm sai"

                    st.markdown(f"""
                    <div class="quiz-result-card">
                        <div><b>Câu {i+1}:</b> {q["question"]}</div>
                        <div class="{result_class}">{result_text}</div>
                        <div><b>Bạn chọn:</b> {selected_letter}</div>
                        <div><b>Đáp án đúng:</b> {q["correct"]}</div>
                        <div><b>Giải thích:</b> {q["explain"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ========================
# TAB 4: PHÒNG THÍ NGHIỆM AI
# ========================
with tabs[3]:
    st.subheader("🔬 Phòng thí nghiệm Vật lí AI")

    exp_type = st.selectbox("Chọn thí nghiệm", [
        "Định luật Ohm",
        "Rơi tự do",
        "Dao động điều hòa"
    ])

    x_input = st.text_input("Dữ liệu X")
    y_input = st.text_input("Dữ liệu Y")

    if st.button("Phân tích", key="exp_btn"):
        try:
            x = np.array(list(map(float,x_input.split())))
            y = np.array(list(map(float,y_input.split())))

            slope, intercept, r, _, _ = linregress(x,y)

            fig, ax = plt.subplots()
            ax.scatter(x,y)
            ax.plot(x, slope*x + intercept)
            st.pyplot(fig)

            st.write(f"Hệ số góc: {slope:.3f}")
            st.write(f"R: {r:.3f}")

            answer = ask_ai([
                {"role":"user","content":f"Giải thích kết quả thí nghiệm {exp_type}"}
            ])

            st.markdown(answer)

        except:
            st.error("Dữ liệu không hợp lệ!")

# ========================
# TAB 5: MÔ PHỎNG
# ========================
with tabs[4]:
    st.subheader("Con lắc đơn")

    L = st.slider("Chiều dài (m)", 0.1, 2.0, 1.0)
    g = 9.8

    T = 2 * np.pi * np.sqrt(L / g)

    st.write(f"Chu kỳ T = {T:.2f} s")

# ========================
# TAB 6: CHẤM BÀI
# ========================
with tabs[5]:
    student_answer = st.text_area("Bài làm học sinh")
    correct_answer = st.text_area("Đáp án đúng")

    if st.button("Chấm bài", key="grade_btn"):
        if student_answer and correct_answer:
            answer = ask_ai([
                {"role":"user","content":f"So sánh:\n{student_answer}\nĐáp án:\n{correct_answer}"}
            ])

            st.markdown(answer)

# ========================
# TAB 7: CÔNG THỨC
# ========================
with tabs[6]:
    st.latex("v=v_0+at")
    st.latex("s=v_0t+\\frac{1}{2}at^2")
    st.latex("I=\\frac{U}{R}")
    st.latex("T=2\\pi\\sqrt{\\frac{l}{g}}")

# ========================
# TAB 8: LỊCH SỬ
# ========================
with tabs[7]:
    st.subheader("Lịch sử học tập")

    # Kiểm tra có history chưa
    if "history" in st.session_state and st.session_state.history:
        # Mỗi câu hỏi là một expander, click mở ra thấy đáp án
        for i, item in enumerate(st.session_state.history):
            with st.expander(item["question"], expanded=False):
                st.markdown(item["answer"])
    else:
        st.info("Chưa có lịch sử hỏi đáp nào.")
