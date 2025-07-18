from re import T
import streamlit as st
import subprocess
import sys
import os
import time
from pathlib import Path
import pandas as pd
import tempfile
from auth import is_logged_in, show_login_page, show_user_info, login_required
import csv
from typing import Any, Dict, List
import pandas as _pd


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
        parts = [p for p in (self.p_val, self.t_val) if p]

        x = [self.p_val, self.t_val]
        #st.write(x)

        return x

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

    #step0_button = st.button("Collect Data", use_container_width=True)
    #if step0_button:
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

            #st.session_state.raw_headers = headers
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
            #st.session_state.num_f = int(len(headers) / 2) + 1

        if st.session_state.get("configs"):
            with st.container(border=True):
                st.subheader("Column Select")

                st.session_state.sdf = st.selectbox(
                    "Sociodemographic Factor",
                    st.session_state.raw_headers,
                    key="app_sdf")

                st.session_state.base = st.selectbox(
                    "Base", st.session_state.raw_headers)

                #st.markdown("**ID Column will be formatted as:** "
                #f"`{ID_INSTRUCTION}`")

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
                    cfg.use_col = c3.checkbox("Use",
                                              value=True,
                                              key=f"use_{cfg.col_num}")

                new_match = st.button("New Column Set",
                                      use_container_width=True)
                if new_match:
                    st.session_state.num_f += 1

            with st.container(border=True):
                st.header("Metrics")
                corr = st.checkbox("Correlation", value=True)
                gini = st.checkbox("Gini Coefficient", value=True)
                ks = st.checkbox("KS Test", value=True)

            save_and_run = st.button("Save Mapping & Run Analysis V3",
                                     key="ver3_save")
            if save_and_run:
                try:
                    if corr:
                        tests_run["correlation"] = True
                    else:
                        tests_run["correlation"] = False

                    if gini:
                        tests_run["gini"] = True
                    else:
                        tests_run["gini"] = False

                    if ks:
                        tests_run["ks"] = True
                    else:
                        tests_run["ks"] = False

                    #Build mapped DataFrame
                    raw_df = pd.DataFrame(st.session_state.raw_data,
                                          columns=st.session_state.raw_headers)
                    keep_cfgs = [
                        cfg for cfg in st.session_state.configs if cfg.use_col
                    ]
                    origs = [cfg.original for cfg in keep_cfgs]

                    matches = [
                        keep_cfgs[i].new_header
                        for i in range(st.session_state.num_f)
                    ]

                    #st.write(matches)

                    from aiFairnessPipeline.src.ParsePredictionsByDem import iterateOverData4, labelBins, create_bin_function, _loadApproachColumn2, calcMetricOnFullData

                    #mapped_df = raw_df[origs].rename(
                    #columns=dict(zip(origs, new_names)))

                    #st.write(st.session_state.sdf)
                    #WORKING DF
                    #st.dataframe(raw_df)

                    results_df = iterateOverData4(raw_df, matches,
                                                  st.session_state.sdf,
                                                  st.session_state.base,
                                                  tests_run)

                    st.subheader("Bias Scores")
                    #st.write("WORKING RESULTS!! ")
                    st.dataframe(results_df)

                    #is this gini or gini p?
                    if tests_run["gini"]:
                        for index, row in results_df.iterrows():
                            st.metric(label="Gini Coefficient",
                                      value=round(row["giniCoefficient"], 3))

                    if tests_run["ks"]:
                        for index, row in results_df.iterrows():
                            st.metric(label="KS Test",
                                      value=round(row["ksStat"], 3))

                    if tests_run["correlation"]:
                        for index, row in results_df.iterrows():
                            st.metric(label="Pearson Correlation",
                                      value=round(row["pearson"], 3))

                    original_dir = os.getcwd()
                    os.chdir('aiFairnessPipeline')
                    sys.path.append('.')
                    os.chdir(original_dir)

                    #st.dataframe(mapped_df)

                    #PUT IMAGE AND TABLE CODE BACK IN FROM VERSION 4

                except Exception as e:
                    st.error(f"Error saving/running V3 analysis: {e}")
                    st.code(str(e), language='text')

                st.subheader("Graphs")
                from PIL import Image
                pic_file = "aiFairnessPipeline/ConcentrationCurve.png"
                image = Image.open(pic_file)
                st.image(image, caption="Uploaded PNG", use_container_width=True)
