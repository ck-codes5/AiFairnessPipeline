from dlatk.featureGetter import FeatureGetter
from dlatk.regressionPredictor import RegressionPredictor, ClassifyPredictor
from dlatk.outcomeGetter import OutcomeGetter
import dlatk.dlaConstants as dlac
import csv
from abc import ABC, abstractmethod


#Default vars
CSV = "DS4UDIsBlackResults2.csv"
MODEL_NAME = 'ridgecv'
DATABASE = 'ds4ud_adapt'
TABLES = ['ema_nagg_v9_txt', 'fb_text_v8']
TABLE = 'fb_text_v8'
CORREL_FIELD = 'user_id'
OUTCOME_TABLE = "bl_wave_v9"
OUTCOME_FIELDS = ["phq9_sum"]
OUTCOME_CONTROLS = ["individual_income"]
GROUP_FREQ_THRESH = 0
FEATURE_TABLES = [['feat$dr_pca_ema_nagg_v9_txt_reduced100$ema_nagg_v9_txt$user_id'], ['feat$dr_pca_fb_text_v8_reduced100$fb_text_v8$user_id']]#[['feat$cat_LIWC2022$ema_nagg_v9_txt$user_id$1gra'], ['feat$cat_LIWC2022$fb_text_v8$user_id$1gra']]#[['feat$roberta_ba_meL11con$ema_nagg_v9_txt$user_id'], ['feat$roberta_ba_meL11con$fb_text_v8$user_id']]
FEATURE_SELECTION_STRING = 'magic_sauce'


class Test(ABC):
    @abstractmethod
    def __init__(self):
        self.scores = {}
    @abstractmethod
    def run(self):
        pass



class ClassificationTest(Test):

    name = "Classification Test"

    def __init__(self, csv=CSV, db=DATABASE, table=TABLE, outcomeTable=OUTCOME_TABLE, correlField = CORREL_FIELD, featTables = FEATURE_TABLES, outcomeFields = OUTCOME_FIELDS, outcomeControls = OUTCOME_CONTROLS, groupFreqThresh = GROUP_FREQ_THRESH, featureSelectionString = FEATURE_SELECTION_STRING):
        self.csv = csv
        og = OutcomeGetter(corpdb = db, corptable = table, correl_field=correlField, outcome_table=outcomeTable, outcome_value_fields=outcomeFields, outcome_controls=outcomeControls, outcome_categories = [], multiclass_outcome = [], featureMappingTable='', featureMappingLex='', wordTable = None, fold_column = None, group_freq_thresh = groupFreqThresh)
        fgs = [FeatureGetter(corpdb = db, corptable = table, correl_field=correlField, featureTable=featTable, featNames="", wordTable = None) for featTable in featTables]
        RegressionPredictor.featureSelectionString = dlac.DEF_RP_FEATURE_SELECTION_MAPPING[featureSelectionString]

        self.rp = ClassifyPredictor(og, fgs, 'lr', None, None)
        self.result = {
            "name": self.name,
            "tables": {
                "table": table,
                "feat": featTables,
                "outcome": outcomeTable,
                "outcomeFields": outcomeFields,
                "outcomeControls": outcomeControls
            },
            "scores": {}
        }

    def run(self):
        scoresRaw = self.rp.testControlCombos(savePredictions=True, comboSizes = [], nFolds = 10, allControlsOnly=True)
        self._saveResults(scoresRaw)

    def _saveResults(self, scoresRaw):
        outputStream = open(self.csv, 'a')
        csv_writer = csv.writer(outputStream)
        csv_writer.writerow([self.name, self.result["tables"]["outcomeFields"], self.result["tables"]["outcomeControls"], self.result["tables"]["feat"]])
        ClassifyPredictor.printComboControlScoresToCSV(scoresRaw, outputStream, delimiter=",")
        outputStream.close()
        outputStream = open("CLASS_" + self.csv, 'a')
        csv_writer = csv.writer(outputStream)
        csv_writer.writerow([self.name, self.result["tables"]["outcomeFields"], self.result["tables"]["outcomeControls"], self.result["tables"]["feat"]])
        ClassifyPredictor.printComboControlPredictionsToCSV(scoresRaw, outputStream, delimiter=",")
        outputStream.close()
        
        #for score in scoresRaw:
        #    self.result["scores"][score] = scoresRaw[score][tuple()][1]



class FactorAdaptationClassificationTest(ClassificationTest):
    name = "Factor Adaptation Classification Test"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, factorTable, adaptationFactors = None):
        scoresRaw = self.rp.testControlCombos(nFolds = 10, savePredictions=True, adaptTables=factorTable, adaptColumns = adaptationFactors, allControlsOnly=True)
        self._saveResults(scoresRaw)



class RegressionTest(Test):

    name = "Regression Test"

    def __init__(self, csv=CSV, db=DATABASE, table=TABLE, outcomeTable=OUTCOME_TABLE, correlField = CORREL_FIELD, featTables = FEATURE_TABLES, outcomeFields = OUTCOME_FIELDS, outcomeControls = OUTCOME_CONTROLS, groupFreqThresh = GROUP_FREQ_THRESH, featureSelectionString = FEATURE_SELECTION_STRING):
        self.csv = csv
        og = OutcomeGetter(corpdb = db, corptable = table, correl_field=correlField, outcome_table=outcomeTable, outcome_value_fields=outcomeFields, outcome_controls=outcomeControls, outcome_categories = [], multiclass_outcome = [], featureMappingTable='', featureMappingLex='', wordTable = None, fold_column = None, group_freq_thresh = groupFreqThresh)
        fgs = [FeatureGetter(corpdb = db, corptable = table, correl_field=correlField, featureTable=featTable, featNames="", wordTable = None) for featTable in featTables]
        RegressionPredictor.featureSelectionString = dlac.DEF_RP_FEATURE_SELECTION_MAPPING[featureSelectionString]

        self.rp = RegressionPredictor(og, fgs, 'ridgehighcv', None, None)
        self.result = {
            "name": self.name,
            "tables": {
                "table": table,
                "feat": featTables,
                "outcome": outcomeTable,
                "outcomeFields": outcomeFields,
                "outcomeControls": outcomeControls
            },
            "scores": {}
        }

    def run(self):
        scoresRaw = self.rp.testControlCombos(savePredictions=True, numOfFactors = None, comboSizes = [], nFolds = 10, allControlsOnly=True, report = False)
        #print("SCORESRAW: ", scoresRaw)
        self._saveResults(scoresRaw)

    def _saveResults(self, scoresRaw):
        '''for score in scoresRaw:
            self.result["scores"][score] = scoresRaw[score][tuple()][1]

        rh.SaveResult(self.result, "PoststratFactoradaptResults.json")'''
        outputStream = open(self.csv, 'a')
        csv_writer = csv.writer(outputStream)
        csv_writer.writerow([self.name, self.result["tables"]["outcomeFields"], self.result["tables"]["outcomeControls"], self.result["tables"]["feat"]])
        #csv_writer.writerow([str(scoresRaw)])

        RegressionPredictor.printComboControlScoresToCSV(scoresRaw, outputStream)
        outputStream.close()
        outputStream.close()
        outputStream = open("REG_" + self.csv, 'a')
        csv_writer = csv.writer(outputStream)
        csv_writer.writerow([self.name, self.result["tables"]["outcomeFields"], self.result["tables"]["outcomeControls"], self.result["tables"]["feat"]])
        RegressionPredictor.printComboControlPredictionsToCSV(scoresRaw, outputStream)
        outputStream.close()



class ResidualControlRegressionTest(RegressionTest):

    name = "Residualized Controls Regression Test"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        scoresRaw = self.rp.testControlCombos(savePredictions=True, numOfFactors = None, nFolds = 10, residualizedControls =True, allControlsOnly=True, report = False)
        self._saveResults(scoresRaw)



class FactorAdaptationRegressionTest(RegressionTest):
    name = "Factor Adaptation Regression Test"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, adaptationFactors):
        self.result["adaptationFactors"] = adaptationFactors
        scoresRaw = self.rp.testControlCombos(savePredictions=True, numOfFactors = None, nFolds = 10, adaptationFactorsName = adaptationFactors, integrationMethod="fa", allControlsOnly=True, report = False)
        self._saveResults(scoresRaw)



class FactorAdditionRegressionTest(RegressionTest):
    name = "Factor Addition Regression Test"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, adaptationFactors):
        self.result["adaptationFactors"] = adaptationFactors
        scoresRaw = self.rp.testControlCombos(savePredictions=True, numOfFactors = None, nFolds = 10, adaptationFactorsName = adaptationFactors, integrationMethod="plus", allControlsOnly=True, report = False)
        self._saveResults(scoresRaw)



class ResidualFactorAdaptationRegressionTest(RegressionTest):
    name = "Residualized Factor Adaptation Regression Test"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, adaptationFactors):
        scoresRaw = self.rp.testControlCombos(savePredictions=True, numOfFactors = None, nFolds = 10, residualizedControls=True, adaptationFactorsName = adaptationFactors, integrationMethod="rfa", allControlsOnly=True, report = False)
        self._saveResults(scoresRaw)

        