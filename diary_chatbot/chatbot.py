import os, sys, time, json, random
from pathlib import Path
import streamlit as st

HERE = Path(__file__).resolve().parent
MOD_DIR = HERE / "dementia-chat-modular"
sys.path.append(str(MOD_DIR))

import main
from model import ask_gpt, load_few_shot_empathy

FEW_SHOT = load_few_shot_empathy(str(MOD_DIR / "prompts" / "few_shot_empathy.txt"))

# ===== UI 기본 =====
st.set_page_config(page_title="당신의 소중한 말벗 또랑이", page_icon="🍊", layout="wide")
st.markdown("## 🍊 당신의 소중한 말벗, 또랑이")
st.write("‘일기’라고 말하면 체크리스트 → 점수 계산 → 주제 3개. 주제당 3문항 이후에는 매 턴 동의 여부를 확인해요.")

# ===== 상태 =====
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"안녕하세요👋 오늘 하루는 어떠셨어요?"}]
if "diary_mode" not in st.session_state:
    st.session_state.diary_mode = False
if "topic_i" not in st.session_state:
    st.session_state.topic_i = 0
if "qcount_in_topic" not in st.session_state:
    st.session_state.qcount_in_topic = 0
if "used_idx_by_topic" not in st.session_state:
    st.session_state.used_idx_by_topic = []
if "awaiting_consent" not in st.session_state:
    st.session_state.awaiting_consent = False
if "diary_sess" not in st.session_state:
    st.session_state.diary_sess = None
if "topics" not in st.session_state:
    st.session_state.topics = []

# === Topic 자동 라벨링 상태 (일반 대화 전용) ===
if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""     # 확정된 현재 topic
if "candidate_topic" not in st.session_state:
    st.session_state.candidate_topic = ""   # 바뀔 후보
if "candidate_votes" not in st.session_state:
    st.session_state.candidate_votes = 0    # 후보 누적표(스무딩)

# ===== 체크리스트 정의 =====
CHECKS = [
    ("today_date",        "오늘이 몇월 며칠일까요?"),
    ("today_weather",     "오늘 날씨는 어떤가요?"),
    ("current_location",  "지금 어디에 계신가요?"),
    ("date_7days_ago",    "오늘로부터 7일 전은 몇월 며칠일까요?"),
    ("yesterday_activity","어제 뭐하셨어요?"),
]
SCORE_FN = {
    "today_date":         main.score_today_date,
    "today_weather":      main.score_today_weather,
    "current_location":   main.score_current_location,
    "date_7days_ago":     main.score_seven_days_ago,
    "yesterday_activity": main.score_yesterday_activity,
}

STOP_WORDS = ["그만","일기 끝","일기 종료","종료","끝낼래","그만할래"]
NEXT_WORDS = ["다른 단어"]

CONSENT_PROMPT = "이 주제에 대해 더 이야기해볼까요? 원하시면 이어가고, 아니면 다음 주제로 넘어갈게요."

# ===== 공감/유틸 =====
def empathetic_reply(user_text: str) -> str:
    prompt = (
        "역할: 공감형 노년 맞춤 대화 코치.\n"
        "규칙:\n- 1문장 공감\n- 이어서 구체적이고 답하기 쉬운 질문 딱 1개\n- 전체 2~3문장, 존댓말\n"
        f"{FEW_SHOT}\n\n"
        f"사용자 발화: \"{user_text.strip()}\"\n응답:"
    )
    out = ask_gpt(prompt=prompt, model=os.environ.get("CHAT_MODEL","gpt-4o-mini"),
                  temperature=0.7, max_tokens=220, response_format={"type":"text"})
    return out.strip() if out else "말씀을 들으니 마음이 쓰이네요. 혹시 그때 어떤 상황이었는지 알려주실 수 있을까요?"

def empathy_only(user_text: str) -> str:
    """질문 없는 공감만"""
    prompt = (
        "역할: 공감형 노년 맞춤 대화 코치.\n"
        "규칙:\n- 사용자의 감정을 한 문장으로 공감만 한다.\n"
        "- 질문이나 요청 금지. 오직 공감 1문장.\n"
        f"{FEW_SHOT}\n\n"
        f"사용자 발화: \"{user_text.strip()}\"\n출력:"
    )
    out = ask_gpt(prompt=prompt, model=os.environ.get("CHAT_MODEL","gpt-4o-mini"),
                  temperature=0.6, max_tokens=120, response_format={"type":"text"})
    return (out or "말씀을 들으니 마음이 쓰이네요.").strip()

def log_user_turn(user_raw: str, topic: str = "", meta: dict | None = None, ts: float | None = None):
    nrm = main.normalize_user_utterance(user_raw or "")
    std = nrm.get("standard") or (user_raw or "")
    main.conversation_memory_raw.append(user_raw)
    main.conversation_memory_std.append(std)
    main.log_event("user", content_raw=user_raw, content_std=std, topic=topic, meta=meta, ts=ts)
    try: main.check_memory_consistency(std, user_raw, nrm)
    except: pass
    return std

def log_assistant_turn(text: str, topic: str = "", meta: dict | None = None, ts: float | None = None):
    main.log_event("assistant", content_raw=text, content_std=text, topic=topic, meta=meta, ts=ts)

def classify_consent(user_std: str, topic: str) -> bool:
    """사용자 답변을 1(계속)/0(다음)으로 분류"""
    ctx = " | ".join(main.conversation_memory_std[-3:])
    prompt = (
        "당신은 화제 지속 의사 분류기입니다.\n"
        "규칙:\n- 사용자가 주제에 대해 더 얘기하고 싶으면 1,\n"
        "- 그만하거나 다른 주제로 넘어가고 싶으면 0.\n"
        "- 다른 출력 금지.\n\n"
        f"주제: {topic}\n"
        f"최근 맥락: {ctx}\n"
        f"사용자 최신 발화: {user_std}\n\n"
        "출력: 1 또는 0"
    )
    out = ask_gpt(prompt=prompt, model=os.environ.get("CHAT_MODEL","gpt-4o-mini"),
                  temperature=0.0, max_tokens=4, response_format={"type":"text"})
    return (out or "").strip().startswith("1")

# ===== (일반 대화 전용) topic 자동 라벨링 =====
TOPIC_HINTS = [
    (["잠", "불면", "깊게 못 자", "잠을 못"], "불면증"),
    (["부산", "해운대", "광안리", "서면"], "부산 여행"),
    (["여행", "떠났", "다녀왔"], "여행"),
    (["식사", "먹었", "반찬", "맛집"], "식사/음식"),
    (["가족", "손주", "아들", "딸"], "가족"),
    (["건강", "병원", "통증", "검사"], "건강"),
]

def _hint_topic(std_text: str) -> str | None:
    t = std_text
    for keywords, label in TOPIC_HINTS:
        if any(k in t for k in keywords):
            return label
    return None

def _normalize_topic_label(lbl: str) -> str:
    lbl = (lbl or "").strip().replace("주제:", "").replace("Topic:", "")
    return lbl[:18]  # 너무 길면 잘라서 1~2단어 느낌 유지

def infer_topic_label_with_llm(std_text: str, prev_topic: str, recent_ctx: str) -> tuple[str, float]:
    """
    LLM으로 현재 발화를 1~2단어 topic으로 추정.
    반환: (label, confidence[0~1])
    """
    prompt = (
        "당신은 대화의 현재 주제를 1~2단어로 요약하는 분류기입니다.\n"
        "규칙:\n"
        "- 출력은 JSON으로만: {\"label\": \"...\", \"confidence\": 0.0~1.0}\n"
        "- 직전 주제(prev_topic)를 유지하는 편이 안전하지만, 명백한 전환(예: 여행→불면증)엔 새 라벨 제안.\n"
        "- 라벨은 간결(예: \"부산 여행\", \"불면증\", \"가족\", \"식사/음식\").\n\n"
        f"prev_topic: {prev_topic or '(없음)'}\n"
        f"recent_context: {recent_ctx}\n"
        f"user_turn: {std_text}\n\n"
        "JSON만 출력:"
    )
    out = ask_gpt(prompt=prompt, model=os.environ.get("CHAT_MODEL","gpt-4o-mini"),
                  temperature=0.2, max_tokens=120, response_format={"type":"json_object"})
    try:
        data = json.loads(out) if out else {}
        label = _normalize_topic_label(data.get("label",""))
        conf = float(data.get("confidence", 0.0))
        if not label:
            label = prev_topic or ""
        return label, max(0.0, min(1.0, conf))
    except Exception:
        return prev_topic or "", 0.0

def update_topic_by_smoothing(std_text: str) -> str:
    """
    일반 대화에서 topic 자동 업데이트.
    - 1) 키워드 힌트
    - 2) LLM 분류(label, confidence)
    - 3) 스무딩: conf>=0.70이면 즉시 전환, 아니면 같은 후보 2표면 전환
    """
    prev = st.session_state.current_topic or ""
    recent_ctx = " | ".join(main.conversation_memory_std[-3:])
    hinted = _hint_topic(std_text)
    candidate = hinted or prev

    llm_label, conf = infer_topic_label_with_llm(std_text, prev_topic=prev, recent_ctx=recent_ctx)
    if llm_label and llm_label != prev:
        candidate = llm_label

    STRONG_CONF = 0.70
    if candidate == prev:
        st.session_state.candidate_topic = ""
        st.session_state.candidate_votes = 0
        return prev

    if conf >= STRONG_CONF:
        st.session_state.current_topic = candidate
        st.session_state.candidate_topic = ""
        st.session_state.candidate_votes = 0
        return candidate
    else:
        if st.session_state.candidate_topic == candidate:
            st.session_state.candidate_votes += 1
        else:
            st.session_state.candidate_topic = candidate
            st.session_state.candidate_votes = 1

        if st.session_state.candidate_votes >= 2:
            st.session_state.current_topic = candidate
            st.session_state.candidate_topic = ""
            st.session_state.candidate_votes = 0
            return candidate
        else:
            return prev

# ===== 일기장 =====
def start_diary_session():
    st.session_state.diary_sess = {
        "diary_id": f"diary_{int(time.time())}",
        "started_at": time.time(),
        "scores": {},
        "score_total": 0,
        "messages": [],
        "topics": [],
        "diary_summaries": []
    }
    st.session_state.diary_mode = True
    st.session_state.topic_i = 0
    st.session_state.qcount_in_topic = 0
    st.session_state.topics = []
    st.session_state.used_idx_by_topic = []
    st.session_state.awaiting_consent = False

def ask_check_question(i: int):
    _, q = CHECKS[i]
    ts = time.time()
    st.session_state.messages.append({"role":"assistant","content":f"[일기장] {q}"})
    log_assistant_turn(q, topic="체크리스트", ts=ts)
    st.session_state.diary_sess["messages"].append({"role":"assistant","content":q,"topic":"체크리스트","ts":ts})

def handle_check_answer(i: int, user_raw: str):
    key, q = CHECKS[i]
    ts = time.time()
    std = log_user_turn(user_raw, topic="체크리스트", meta={"tag": key}, ts=ts)
    st.session_state.diary_sess["messages"].append(
        {"role":"user","content_raw":user_raw,"content_std":std,"topic":"체크리스트","ts":ts}
    )
    score = int(SCORE_FN[key](std))
    st.session_state.diary_sess["scores"][key] = score
    st.session_state.diary_sess["score_total"] = sum(st.session_state.diary_sess["scores"].values())

def setup_topics():
    topics = main.pick_diary_topics(3)
    st.session_state.topics = topics
    st.session_state.used_idx_by_topic = [set() for _ in topics]
    st.session_state.topic_i = 0
    st.session_state.qcount_in_topic = 0
    st.session_state.awaiting_consent = False
    st.session_state.diary_sess["topics"] = topics
    msg = f"[일기장] 오늘의 주제: {', '.join(topics)}"
    st.session_state.messages.append({"role":"assistant","content":msg})
    log_assistant_turn(msg)

def pick_question_for_topic(ti: int) -> str:
    used = st.session_state.used_idx_by_topic[ti]
    all_idx = list(range(len(main.DIARY_QUESTION_TEMPLATES)))
    cand = [i for i in all_idx if i not in used]
    if not cand: used.clear(); cand = all_idx[:]
    idx = random.choice(cand)
    used.add(idx)
    t = st.session_state.topics[ti]
    return main.DIARY_QUESTION_TEMPLATES[idx].format(t=t)

def ask_topic_question():
    ti = st.session_state.topic_i
    q = pick_question_for_topic(ti)
    ts = time.time()
    msg = f"[일기장] {q}"
    st.session_state.messages.append({"role":"assistant","content":msg})
    log_assistant_turn(q, topic=st.session_state.topics[ti], ts=ts)
    st.session_state.diary_sess["messages"].append({"role":"assistant","content":q,"topic":st.session_state.topics[ti],"ts":ts})
    # 질문을 보낸 시점에만 카운트 +1
    st.session_state.qcount_in_topic += 1

def ask_consent():
    ts = time.time()
    st.session_state.awaiting_consent = True
    st.session_state.messages.append({"role":"assistant","content":f"[일기장] {CONSENT_PROMPT}"})
    log_assistant_turn(CONSENT_PROMPT, topic=st.session_state.topics[st.session_state.topic_i],
                       meta={"type":"consent"}, ts=ts)
    st.session_state.diary_sess["messages"].append(
        {"role":"assistant","content":CONSENT_PROMPT,"topic":st.session_state.topics[st.session_state.topic_i],"ts":ts}
    )

def handle_consent_input(user_raw: str):
    topic = st.session_state.topics[st.session_state.topic_i]
    ts = time.time()
    std = log_user_turn(user_raw, topic=topic, meta={"phase":"consent"}, ts=ts)
    st.session_state.diary_sess["messages"].append(
        {"role":"user","content_raw":user_raw,"content_std":std,"topic":topic,"ts":ts}
    )
    # 공감만
    empath = empathy_only(std)
    st.session_state.messages.append({"role":"assistant","content":empath})
    log_assistant_turn(empath, topic=topic, meta={"type":"empathy_after_consent"})
    st.session_state.diary_sess["messages"].append({"role":"assistant","content":empath,"topic":topic,"ts":time.time()})
    # 분류 → 계속이면 같은 주제 다음 질문 1개, 아니면 다음 주제로
    cont = classify_consent(std, topic)
    st.session_state.awaiting_consent = False
    if cont: ask_topic_question()
    else: goto_next_topic_or_finish()

def goto_next_topic_or_finish():
    st.session_state.topic_i += 1
    st.session_state.qcount_in_topic = 0
    st.session_state.awaiting_consent = False
    if st.session_state.topic_i < len(st.session_state.topics):
        ask_topic_question()
    else:
        # 정상 종료 → 요약 & 저장
        st.session_state.diary_mode = False
        st.session_state.diary_sess["ended_at"] = time.time()
        try: main.summarize_diary_session(st.session_state.diary_sess)
        except: pass
        main.diary_memory.append(st.session_state.diary_sess)
        st.session_state.messages.append({"role":"assistant","content":"[일기장] 오늘 기록이 정리되었어요. 이어서 자유롭게 이야기 나눠요. 😊"})

def handle_topic_answer(user_raw: str):
    ti = st.session_state.topic_i
    topic = st.session_state.topics[ti]
    ts = time.time()
    std = log_user_turn(user_raw, topic=topic, ts=ts)
    st.session_state.diary_sess["messages"].append({"role":"user","content_raw":user_raw,"content_std":std,"topic":topic,"ts":ts})
    # 질문 없는 공감만
    empath = empathy_only(std)
    st.session_state.messages.append({"role":"assistant","content":empath})
    log_assistant_turn(empath, topic=topic, meta={"type":"followup_empathy"})
    st.session_state.diary_sess["messages"].append({"role":"assistant","content":empath,"topic":topic,"ts":time.time()})
    # 질문은 항상 1개
    if st.session_state.qcount_in_topic < 3:
        ask_topic_question()
    else:
        ask_consent()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ===== 입력 =====
user_text = st.chat_input("편하게 이야기해 주세요.")
if user_text:
    if st.session_state.diary_mode and any(w in user_text for w in STOP_WORDS):
        if st.session_state.get("diary_sess"):
            st.session_state.diary_sess["ended_at"] = time.time()
            try:
                main.summarize_diary_session(st.session_state.diary_sess)
            except Exception:
                pass
            main.diary_memory.append(st.session_state.diary_sess)

        st.session_state.diary_mode = False
        st.session_state.messages.append(
            {"role":"assistant","content":"[일기장] 오늘 기록을 저장했어요. 오늘은 여기까지 기록할게요."}
        )
        log_assistant_turn("일기 종료(저장 완료)", topic="체크리스트", meta={"cmd":"stop"})
        st.rerun()

    st.session_state.messages.append({"role":"user","content":user_text})

    if (not st.session_state.diary_mode) and ("일기" in user_text):
        start_diary_session(); ask_check_question(0)
    elif st.session_state.diary_mode:
        answered = sum(1 for m in st.session_state.diary_sess["messages"]
                       if m.get("topic")=="체크리스트" and m.get("role")=="user")
        if answered < 5:
            handle_check_answer(answered, user_text)
            if answered+1 < 5:
                ask_check_question(answered+1)
            else:
                setup_topics()
                ask_topic_question()
        else:
            if st.session_state.awaiting_consent:
                handle_consent_input(user_text)
            else:
                handle_topic_answer(user_text)
    else:
        # ===== 일반 대화: 자동 topic 라벨링 + 스무딩 적용 =====
        std = main.normalize_user_utterance(user_text or "").get("standard") or user_text
        auto_topic = update_topic_by_smoothing(std)

        # 로그(사용자)
        main.conversation_memory_raw.append(user_text)
        main.conversation_memory_std.append(std)
        main.log_event("user", content_raw=user_text, content_std=std, topic=auto_topic, meta=None, ts=time.time())
        try:
            main.check_memory_consistency(std, user_text, {"standard": std})
        except Exception:
            pass

        # 응답
        reply = empathetic_reply(std)
        st.session_state.messages.append({"role":"assistant","content":reply})
        main.log_event("assistant", content_raw=reply, content_std=reply, topic=auto_topic, meta=None, ts=time.time())

    st.rerun()

st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("💾 conversation_log.json",
        data=json.dumps(main.conversation_log, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="conversation_log.json", mime="application/json")
with c2:
    st.download_button("🧠 fact_memory.json",
        data=json.dumps(main.fact_memory, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="fact_memory.json", mime="application/json")
with c3:
    st.download_button("📔 diary_memory.json",
        data=json.dumps(main.diary_memory, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="diary_memory.json", mime="application/json")

if st.session_state.get("diary_sess"):
    st.download_button("📝 현재 일기장 세션(JSON)",
        data=json.dumps(st.session_state.diary_sess, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="diary_session_current.json", mime="application/json")

