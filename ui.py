import streamlit as st
import json
import re
import random

# Style for the app
st.markdown("""
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 300px;
            font-size: 1rem;
        }
        .stCustomFont {
            word-spacing: 5px !important;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Intialize constants
@st.cache_resource
def init_constants():
    with open("default_bias_words.json", "r") as f:
        DEFAULT_BIAS_WORDS = json.load(f)["default_bias_words"]
    with open("example.json", "r") as f:
        EXAMPLE_DATA = json.load(f)
    return DEFAULT_BIAS_WORDS, EXAMPLE_DATA
DEFAULT_BIAS_WORDS, EXAMPLE_DATA = init_constants()

# Initialize session state
if 'sentence' not in st.session_state:
    st.session_state.sentence = None
if 'bias_words' not in st.session_state:
    st.session_state.bias_words = []

# Set the title of the app
st.title("Demo")

def submit_example():
    st.session_state.sentence = EXAMPLE_DATA[st.session_state.widget_example_option]['sentence']
    st.session_state.bias_words = EXAMPLE_DATA[st.session_state.widget_example_option]['bias_words']
    st.session_state.widget_example_option = None

def submit_sentence():
    with col02:
        if st.session_state.widget_sentence is not None:
            st.session_state.sentence = st.session_state.widget_sentence
            st.success("Sentence updated.")
    st.session_state.widget_sentence = None

def submit_bias_words():
    with col02:
        if st.session_state.widget_bias_words in DEFAULT_BIAS_WORDS:
            st.warning("Bias word already extists in the default list.")
        elif st.session_state.widget_bias_words is not None and st.session_state.widget_bias_words not in st.session_state.bias_words:
            st.session_state.bias_words.append(st.session_state.widget_bias_words)
            st.success("Bias word added.")
        else:
            st.warning("Bias word already extists. Please check the input bias words list.")
    st.session_state.widget_bias_words = None

def inference_original_model(sentence, bias_words):
    output_sentence = "Câu của model gốc"
    tags = ['<s>', 'Theo', '<mask>[0](hát x)', '<mask>[1](pan t)', 'Phó', 'chủ', 'tịch', 'tổ', 'chức', 'tư', 'vấn', '<mask>[2](óp dơ vờ rì sợt phao đây sừn)', 'trụ', 'sở', '<mask>[3](đê li)', 'trong', 'những', 'ngày', 'đầu', 'lập', 'khối,', 'Ấn', 'Độ', 'nghĩ', 'rằng', 'với', 'sự', 'giúp', 'đỡ', 'của', 'Nga,', 'họ', 'có', 'thể', 'ứng', 'phó', 'với', 'Trung', 'Quốc', 'tốt', 'hơn', '</s>']
    return output_sentence, tags

def inference_our_model(sentence, bias_words):
    output_sentence = "Theo harsh pant Phó chủ tịch tổ chức tư vấn observer research foundation trụ sở delhi trong những ngày đầu lập khối, Ấn Độ nghĩ rằng với sự giúp đỡ của Nga, họ có thể ứng phó với Trung Quốc tốt hơn"
    tags = ['<s>', 'Theo', '<mask>[0](hát x)', '<mask>[1](pan t)', 'Phó', 'chủ', 'tịch', 'tổ', 'chức', 'tư', 'vấn', '<mask>[2](óp dơ vờ rì sợt phao đây sừn)', 'trụ', 'sở', '<mask>[3](đê li)', 'trong', 'những', 'ngày', 'đầu', 'lập', 'khối,', 'Ấn', 'Độ', 'nghĩ', 'rằng', 'với', 'sự', 'giúp', 'đỡ', 'của', 'Nga,', 'họ', 'có', 'thể', 'ứng', 'phó', 'với', 'Trung', 'Quốc', 'tốt', 'hơn', '</s>']
    return output_sentence, tags

def style_output(tags):
    style_sentence = ""
    for i, tag in enumerate(tags):
        pattern = r'<mask>\[\d+\]\((.*?)\)'
        matches = re.findall(pattern, tag)
        if tag not in ['<s>', '</s>']:
            color = f"#{random.randint(0, 150):02x}{random.randint(0, 150):02x}{random.randint(0, 150):02x}"
            if matches:
                style_sentence += f"""<span style="color:{color}"><u><b>{matches[0]}</b></u></span>"""
            else:
                style_sentence += tag
            style_sentence += ' '
    return style_sentence

def display_output(sentence, bias_words):

    original_model_sentence, original_model_tags = inference_original_model(sentence, bias_words)
    our_model_sentence, our_model_tags = inference_our_model(sentence, bias_words)

    st.markdown("---")

    if sentence is not None:
        st.markdown(f"**Original sentence:**")
        st.markdown(f"""<p class="stCustomFont">{sentence}</p>""", unsafe_allow_html=True)

    st.markdown("---")

    if bias_words is not []:
        st.text_area(
            label="**Input bias words:**", value="     ".join(bias_words),
            height=100,
        )
    st.text_area(
        label="**Default bias words:**", value="     ".join(DEFAULT_BIAS_WORDS),
        height=200
    )

    st.markdown("---")

    st.markdown(f"**Original model:**")
    st.markdown(f"""<p class="stCustomFont">{style_output(original_model_tags)}</p>""", unsafe_allow_html=True)
    st.markdown(f"""<p class="stCustomFont">{original_model_sentence}</p>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(f"**Our model:**")
    st.markdown(f"""<p class="stCustomFont">{style_output(our_model_tags)}</p>""", unsafe_allow_html=True)
    st.markdown(f"""<p class="stCustomFont">{our_model_sentence}</p>""", unsafe_allow_html=True)

# Create a button with options for examples
example_option = st.selectbox(
    "**Choose an example:**",
    EXAMPLE_DATA.keys(),
    index=None,
    key="widget_example_option",
    on_change=submit_example
)
st.markdown("**OR**")

# Create an input sentence box
input_sentence = st.text_input("**Enter sentence to change:**", key="widget_sentence", on_change=submit_sentence)

# Create two columns
col01, col02 = st.columns(2)

# Create an input bias word box
with col01:
    input_bias_words = st.text_input("**Enter bias words to add:**", key="widget_bias_words", on_change=submit_bias_words)

st.markdown("---")

# Create two columns
col11, col12 = st.columns(2)

# Show selected text
with col11:
    st.markdown(f"**Input text:**")
    st.markdown(f"""<p class="stCustomFont">{st.session_state.sentence if st.session_state.sentence is not None else ''}</p>""", unsafe_allow_html=True)

# Show selected bias words
with col12:
    st.session_state.bias_words = st.multiselect(
        "**Input bias words (Optional):**", 
        st.session_state.bias_words,
        st.session_state.bias_words if len(st.session_state.bias_words) > 0 else None
    )

# Display default bias words
st.text_area(
    label="**We always include these bias words:**", value="     ".join(DEFAULT_BIAS_WORDS),
    height=200
)
st.multiselect(
    "**Check if your word is included:**", 
    DEFAULT_BIAS_WORDS,
    [],
    placeholder="Search for bias words here"
)

st.markdown("---")

# Process input data
if st.button("Process"):
    display_output(st.session_state.sentence, st.session_state.bias_words)