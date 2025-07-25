import faulthandler, sys, signal
faulthandler.enable(file=sys.stderr, all_threads=True)

import os
os.environ["STREAMLIT_LOG_LEVEL"] = "debug"

import streamlit as st
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import tempfile
from auth import is_logged_in, show_login_page, show_user_info, login_required
import csv
from typing import Any, Dict, List
import pandas as _pd
from PIL import Image

options = {
    'giniCoefficient': "Gini Coefficient",
    'giniCoefficientP': "Gini Coefficient P",
    'ksStat': "KS Stat",
    'ksStatP': "KS Stat P",
    'andersonDarling': "Anderson Darling",
    'andersonDarlingNullP': "Anderson Darling Null P",
    'andersonDarlingOtherP': "Anderson Darling Other P",
    'customConcentrationIntegrateValue': "BCI",
    #'customConcentrationIntegrateValue': "BCI Integrate Value",
    'customConcentrationValueNullP': "BCI Null P",
    'customConcentrationValueOtherP': "BCI Other P",
    'pearson': "Pearson Correlation"
}

selections = {}

if st.session_state.get("trigger_rerun", False):
    st.session_state.trigger_rerun = False
    st.rerun()


def readDataFromCSV3(path: str):
    """
    Reads a simple CSV with a header row.
    Returns (headers, data_rows).
    """
    with open(path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty")
    headers = rows[0]
    data = rows[1:]
    return headers, data


class ColumnConfig:

    def __init__(self, col_num: int, original: str, sample_value: Any):
        self.col_num = col_num
        self.original = original
        self.sample_value = sample_value
        self.p_val = None
        self.t_val = None
        self.app_sdf = None
        self.use_col = True

    @property
    def new_header(self):
        return [self.p_val, self.t_val]

    def to_dict(self) -> Dict:
        return {
            "ColNum": self.col_num,
            "Original": self.original,
            "Attributes": {
                "p_val": self.p_val,
                "t_val": self.t_val,
                "app_sdf": self.app_sdf
            },
            "useCol": self.use_col
        }


class UserDataSet:

    def __init__(self,
                 new_col_headers: List[str] = None,
                 data: List[List[Any]] = None):
        self.new_col_headers = new_col_headers or []
        self.data = data or []

    def to_dataframe(self):
        return _pd.DataFrame(self.data, columns=self.new_col_headers)


ID_INSTRUCTION = "{dem_factor1[df1]}__{dem_factor2[df2]}__{demographic}__{custom}"

matches = {}

if 'dataIn' not in st.session_state:
    st.session_state.dataIn = False

st.set_page_config(page_title="AI Fairness Pipeline", layout="wide")

st.title("AI Fairness Pipeline")
st.markdown("Run bias detection analysis on machine learning models")

x = st.file_uploader("Upload a CSV file", type="csv")

if x is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(x.read())
        tmp_file_path = tmp_file.name
    df_cols = pd.read_csv(tmp_file_path)

    st.session_state.dataIn = True
    tests_run = {"correlation": True, "gini": True, "ks": True}

    if st.session_state.dataIn:
        tag_button = st.button("Collect Data",
                               key="ver3_tag",
                               use_container_width=True)
        if tag_button:
            headers, data = readDataFromCSV3(tmp_file_path)

            base_df = pd.read_csv(tmp_file_path)

            st.session_state.raw_headers = list(base_df.columns)
            st.session_state.raw_headers2 = headers
            st.session_state.raw_headers2.insert(0, "")
            st.session_state.raw_data = data
            configs = []
            sample_row = data[0] if data else []
            for i, h in enumerate(headers):
                cfg = ColumnConfig(
                    i, h, sample_row[i] if i < len(sample_row) else None)
                configs.append(cfg)
            st.session_state.configs = configs
            st.session_state.num_f = 1

        if st.session_state.get("configs"):
            #bc1, bc2 = st.columns(2)
            #with bc1:
            with st.container(border=True):
                st.subheader("Column Select")

                st.session_state.sdf = st.selectbox(
                    "Sociodemographic Factor",
                    st.session_state.raw_headers,
                    key="app_sdf")

                st.session_state.base = st.selectbox(
                    "Base", st.session_state.raw_headers)

                st.subheader("Pairs")
                new_match = st.button("New Column Set",
                                      use_container_width=True)
                header_cols = st.columns([3, 3, 1])
                header_cols[0].write("Predicted Values")
                header_cols[1].write("True Values")
                header_cols[2].write("Use")

                for i in range(st.session_state.num_f):
                    cfg = st.session_state.configs[i]
                    c1, c2, c3 = st.columns([3, 3, 1])
                    cfg.p_val = c1.selectbox("Select Column",
                                             st.session_state.raw_headers2,
                                             key=f"p_{cfg.col_num}")
                    cfg.t_val = c2.selectbox("Select Column",
                                             st.session_state.raw_headers2,
                                             key=f"t_{cfg.col_num}")
                    c3.write(" ")
                    c3.write(" ")
                    checkbox_key = f"use_{cfg.col_num}"

                    # Initialize checkbox state once (only if missing)
                    if checkbox_key not in st.session_state:
                        st.session_state[
                            checkbox_key] = True  # default checked

                    cfg.use_col = st.session_state[checkbox_key]

                    c3.checkbox("Use", key=checkbox_key)

                if new_match:

                    st.session_state.num_f += 1
                    cfg = st.session_state.configs[st.session_state.num_f]
                    c1, c2, c3 = st.columns([3, 3, 1])
                    cfg.p_val = c1.selectbox("Select Column",
                                             st.session_state.raw_headers2,
                                             key=f"p_{cfg.col_num}")
                    cfg.t_val = c2.selectbox("Select Column",
                                             st.session_state.raw_headers2,
                                             key=f"t_{cfg.col_num}")
                    c3.write(" ")
                    c3.write(" ")
                    checkbox_key = f"use_{cfg.col_num}"

                    # Initialize checkbox state once (only if missing)
                    if checkbox_key not in st.session_state:
                        st.session_state[
                            checkbox_key] = True  # default checked

                    # Sync cfg.use_col to current checkbox state
                    cfg.use_col = st.session_state[checkbox_key]

                    # Render the checkbox with the stored state
                    c3.checkbox("Use", value=cfg.use_col, key=checkbox_key)

                    #st.session_state.trigger_rerun = True

            #with bc2:
            with st.container(border=True):

                st.header("Metrics")
                selections['pearson'] = st.checkbox("Correlation", value=True)
                selections["giniCoefficient"] = st.checkbox("Gini Coefficient",
                                                            value=True)
                selections["customConcentrationIntegrateValue"] = st.checkbox(
                    "BCI", value=True)
                selections["ksStat"] = st.checkbox("KS Test", value=True)
                selections["andersonDarling"] = st.checkbox("Anderson Darling",
                                                            value=True)

                #st.header("Graphs")
                #ksG = st.checkbox("KS Test ", value=True)
                #bciG = st.checkbox("BCI ", value=True)
                #scatter = st.checkbox("Scatterplots ", value=True)

            save_and_run = st.button("Run Analysis", key="ver3_save")
            if save_and_run:
                print("DEBUG: Save Mapping & Run Analysis V3 button clicked")
                try:

                    print("DEBUG: raw_data length:",
                          len(st.session_state.raw_data))
                    raw_df = pd.DataFrame(st.session_state.raw_data,
                                          columns=st.session_state.raw_headers)
                    print("DEBUG: raw_df shape:", raw_df.shape)
                    keep_cfgs = [
                        cfg for cfg in st.session_state.configs if cfg.use_col
                        and cfg.t_val is not None and cfg.p_val is not None
                    ]
                    matches = [
                        keep_cfgs[i].new_header
                        for i in range(st.session_state.num_f)
                    ]

                    for m in range(len(matches)):
                        if matches[m][0] == "" or matches[m][
                                1] == "" or matches[m][1] is None or matches[
                                    m][0] is None:
                            matches.pop(m)

                    #for cfg in st.session_state.configs:
                    #st.write(cfg.t_val)
                    #st.write(cfg.use_col)

                    #print("DEBUG: matches =", matches)
                    #st.write("DEBUG: matches =", matches)
                    invalid = [m for m in matches if None in m or "" in m]
                    if invalid:
                        st.error(f"Incomplete column selection: {invalid}")
                        st.stop()

                    from aiFairnessPipeline.src.ParsePredictionsByDem import iterateOverData4

                    results_df = iterateOverData4(raw_df, matches,
                                                  st.session_state.sdf,
                                                  st.session_state.base,
                                                  tests_run)
                    print("DEBUG: results_df shape:", results_df.shape)

                    results_df = results_df.reset_index(drop=True)

                    st.header("Bias Scores")
                    #st.dataframe(results_df)

                    res_col_count = 0
                    for index, row in results_df.iterrows():
                        res_col_count += 1
                    res_cols = st.columns(res_col_count)

                    for i in range(len(res_cols)):
                        with res_cols[i]:
                            st.subheader(results_df["outcome"].iloc[i])

                    for key in options:
                        if key in selections and selections[key]:
                            for index, row in results_df.iterrows():
                                with res_cols[index]:
                                    st.metric(label=options[key],
                                              value=round(row[key], 3))

                    original_dir = os.getcwd()
                    os.chdir('aiFairnessPipeline')
                    sys.path.append('.')
                    os.chdir(original_dir)

                    print("DEBUG: Analysis complete, loading graphs")
                    st.header("Graphs")
                    pic_file = "aiFairnessPipeline/ConcentrationCurve.png"
                    print("DEBUG: Opening image file:", pic_file)
                    image = Image.open(pic_file)
                    st.image(image,
                             caption="Uploaded PNG",
                             use_container_width=True)
                    print("DEBUG: Image displayed successfully")

                except Exception as e:
                    print("DEBUG: Exception in analysis or graph block:", e)
                    st.error(f"Error saving/running V3 analysis: {e}")
                    st.code(str(e), language='text')
