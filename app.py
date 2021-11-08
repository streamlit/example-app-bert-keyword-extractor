import streamlit as st
import numpy as np
from pandas import DataFrame
from keybert import KeyBERT

# For Flair (Keybert)
from flair.embeddings import TransformerDocumentEmbeddings

# roberta = TransformerDocumentEmbeddings("roberta-base")

# For table formatting
import seaborn as sns

# For download buttons
from functionforDownloadButtons import download_button
import os
import json

# region format

st.set_page_config(
    page_title="BERT Keyword Extractor",
    page_icon="🎈",
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    st.image("logo.png", width=400)
    st.header("")

with c32:
    st.header("")
    st.header("")
    st.header("")
    st.text("")
    st.text("")
    st.header("")
    st.markdown(
        "###### Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/)&nbsp, with :heart: by [@DataChaz](https://www.charlywargnier.com/) &nbsp | &nbsp [![Follow](https://img.shields.io/twitter/follow/datachaz?style=social)](https://www.twitter.com/datachaz) &nbsp | &nbsp [![this is an image link](https://i.imgur.com/thJhzOO.png)](https://www.buymeacoffee.com/cwar05)"
    )


with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   The *BERT Keyword Extractor* app is an easy-to-use interface built in Streamlit for the amazing [KeyBERT](https://github.com/MaartenGr/KeyBERT) library from Maarten Grootendorst!
-   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers] (https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
-   The tool is still in Beta. Any issues, feedback or suggestions: [![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/DataChaz/BertKeywordExtractor)
-   This app is free. If it's useful to you, you can [buy me a ☕](https://www.buymeacoffee.com/cwar05) to support my work. 🙏
	    """
    )

    st.markdown("")


# with st.expander(" Todos", expanded=True):
#
#     st.write(
#         """
# -   Check text and formatting in tooltips
#
# 	    """
#     )
#
#     st.markdown("")
#
# st.markdown("")
#
# with st.expander("📝 - Done ", expanded=True):
#     st.write(
#         """
#     - [Crash] Keep crashing on S4, issue with DistilBERT model
#     - Do more testing
#     - [Formatting] Add text about app - It's a UI for KeyBERT
#     - Add placeholders at the top
#     - [Formatting] Change logo - add  "a UI for KeyBERT"
#     - [Formatting] Change page title if my name of the app is changed
#     - Add tooltips to all fields
#     - Add max 500 words
#     - Add example for Ngram in Ngram tooltip
#     - Check default settings in screenshot
#     - change default keyword to 10
#     - REMOVED - DOESN'T WORK - Add "candidate" argument. candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s). https://github.com/MaartenGr/KeyBERT/blob/#eb6d0865c958a474f3554518539c7a37dbd9856b/keybert/model.py#L78
#
#     """
#     )
#
#     st.markdown("")

with st.expander("🔆 Coming soon!", expanded=False):

    st.write(
        """  
-   Add more embedding models.
-   Add more options from the KeyBERT API.
-   Allow for larger documents to be reviewed (currently limited to 500 words).
-   If deemed cost-effective, I may add search volume data with each retrieved keyword.

	    """
    )

    st.markdown("")

st.markdown("")

st.markdown("## **📌 Paste document **")


with st.form(key="my_form"):

    # #region cache
    #
    # ###################
    #
    # @st.cache(allow_output_mutation=True)
    # def load_model():
    #     return Summarizer()
    #
    # ###################
    #
    # import streamlit as st
    # import torch
    #
    # # Load the model (only executed once!)
    # @st.cache
    # def load_model():
    #     return torch.load("path/to/model.pt")
    #
    # model = load_model()
    #
    # # Perform a prediction.
    # question = st.text_input("What's your question?")
    # answer = model.predict(question)
    # st.write("Predicted answer:", answer)

    ###################

    # endregion cache

    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        # Model type
        ModelType = st.radio(
            "Choose your model",
            ["DistilBERT (Default)", "Flair"],
            # ["Flair", "DistilBERT (Default)", "Microsoft MiniLM"],
            help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
        )

        if ModelType == "Default (DistilBERT)":
            # kw_model = KeyBERT(model=roberta)

            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT(model=roberta)

            kw_model = load_model()

        # elif ModelType == "Microsoft MiniLM":
        #
        #    @st.cache(allow_output_mutation=True)
        #    def load_model():
        #        return KeyBERT(model="paraphrase-MiniLM-L6-v2")
        #
        #    kw_model = load_model()
        #
        #    # kw_model = KeyBERT(model='paraphrase-MiniLM-L6-v2')

        else:
            # kw_model = KeyBERT("distilbert-base-nli-mean-tokens")
            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT("distilbert-base-nli-mean-tokens")

            kw_model = load_model()

        top_N = st.slider(
            "# of results",
            min_value=1,
            max_value=30,
            value=10,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            min_value=1,
            max_value=4,
            help="""The minimum value for the ngram range.

*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
            # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
        )

        max_Ngrams = st.number_input(
            "Maximum Ngram",
            value=2,
            min_value=1,
            max_value=4,
            help="""The maximum value for the keyphrase_ngram_range.

*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
        )

        StopWordsCheckbox = st.checkbox(
            "Remove stop words",
            help="Tick this box to remove stop words from the document (currently English only)",
        )

        use_MMR = st.checkbox(
            "Use MMR",
            value=True,
            help="You can use Maximal Margin Relevance (MMR) to diversify the results. It creates keywords/keyphrases based on cosine similarity. Try high/low 'Diversity' settings below for interesting variations.",
        )

        Diversity = st.slider(
            "Keyword diversity (MMR only)",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help="""The higher the setting, the more diverse the keywords.
            
Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked.

""",
        )

    with c2:
        doc = st.text_area(
            "Paste your text below (max 500 words)",
            height=510,
        )

        MAX_WORDS = 500

        # total no of words
        import re

        res = len(re.findall(r"\w+", doc))
        # st.write(str(res))

        if res > MAX_WORDS:
            # if len(doc) > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

            doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label="✨ Get me the data!")

    if use_MMR:
        mmr = True
    else:
        mmr = False

    if StopWordsCheckbox:
        StopWords = "english"
    else:
        StopWords = None

if not submit_button:
    st.stop()

if min_Ngrams > max_Ngrams:
    st.warning("min_Ngrams can't be greater than max_Ngrams")
    st.stop()

keywords = kw_model.extract_keywords(
    doc,
    keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
    use_mmr=mmr,
    stop_words=StopWords,
    top_n=top_N,
    # highlight=True,
    diversity=Diversity,
    # candidates=["supervised"],
)


# st.markdown("---")

st.markdown("## **🎈 Check & download results **")

st.header("")

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

with c1:
    CSVButton2 = download_button(keywords, "Data.csv", "🎁 Download (.csv)")
with c2:
    CSVButton2 = download_button(keywords, "Data.txt", "🎁 Download (.txt)")
with c3:
    CSVButton2 = download_button(keywords, "Data.json", "🎁 Download (.json)")

# keywords
# st.write(type(keywords))

st.header("")

# st.markdown("## **📃 Paste document **")

df = (
    DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
    .sort_values(by="Relevancy", ascending=False)
    .reset_index(drop=True)
)

# cmGreen = sns.light_palette("green", as_cmap=True)

df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Relevancy",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Relevancy": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)
