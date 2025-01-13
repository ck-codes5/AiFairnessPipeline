from dlatk.featureGetter import FeatureGetter
from dlatk.database.dataEngine import DataEngine
from dlatk.regressionPredictor import RegressionPredictor, ClassifyPredictor
from dlatk.outcomeGetter import OutcomeGetter
import dlatk.dlaConstants as dlac
import RegAndClassTests
import csv
import subprocess
import multiprocessing
from abc import ABC, abstractmethod



JSON = "DS4UDIsBlackResults2.json"
MODEL_NAME = 'ridgecv'
DATABASE = 'ds4ud_adapt'
TABLES = ['ema_nagg_v9_txt', 'fb_text_v8']
TABLE = 'fb_text_v8'
CORREL_FIELD = 'user_id'
OUTCOME_TABLE = "bl_wave_v9"
OUTCOME_FIELDS = ["phq9_sum"]
OUTCOME_CONTROLS = ["education", "is_male", "individual_income", "age"]
GROUP_FREQ_THRESH = 0
FEATURE_TABLES = [['feat$dr_pca_ema_nagg_v9_txt_reduced100$ema_nagg_v9_txt$user_id'], ['feat$dr_pca_fb_text_v8_reduced100$fb_text_v8$user_id']]



def main():

    ds4udTests()

    args = {
        "json" : "PostStratFactorAdaptTests.json",
        "db" : 'ai_fairness',
        "table" : 'msgs_100u',
        "correlField" : 'cnty',
        "outcomeTable" : "combined_county_outcomes",
        "outcomeFields" : ["heart_disease", "suicide", "life_satisfaction", "perc_fair_poor_health"],
        "outcomeControls" : ["total_pop10", "femalePOP165210D$10", "hispanicPOP405210D$10", "blackPOP255210D$10", "forgnbornHC03_VC134ACS3yr$10", "bachdegHC03_VC94ACS3yr$10", "marriedaveHC03_AC3yr$10", "logincomeHC01_VC85ACS3yr$10", "unemployAve_BLSLAUS$0910", "perc_less_than_18_chr14_2012", "perc_65_and_over_chr14_2012"],
        "groupFreqThresh" : 0,
        "featTables" : ["feat$1gram$msgs_100u$cnty$16to16"]
    }

    ctlbTests()



def ds4udTests():
    outcome = ["substance_past_any", "depress_past_any", "anxiety_past_any"]
    args = {
        "json" : "DS4UD_Outcomes_SingleControlClass.json",
        "db" : 'ds4ud_adapt',
        "table" : 'msgs_ema_words_day_v9',
        "correlField" : 'user_id',
        "outcomeTable" : "survey_outcomes_waves_agg_v9_mod_v2",
        "outcomeFields" : outcome,
        "outcomeControls" : [],
        "groupFreqThresh" : 0,
        "featTables" : ['feat$roberta_la_meL23con$msgs_ema_words_day_v9$user_id']
    }

    adaptationFactors = [["individual_income"], ["age"], ["is_female"]]

    for facs in adaptationFactors:
        
        args["outcomeFields"] = outcome
        args["outcomeControls"] = facs
        #DLATKTests.RegressionTest(**args).run()
        RegAndClassTests.ClassificationTest(**args).run()
        
        #DLATKTests.ResidualControlRegressionTest(**args).run()
        #args["outcomeFields"] = outcome + facs
        #args["outcomeControls"] = []
        #DLATKTests.FactorAdaptationRegressionTest(**args).run(facs)

        RegAndClassTests.FactorAdaptationClassificationTest(**args).run([0])#22])

        #args["outcomeControls"] = facs
        #args["outcomeControls"] = []
        #DLATKTests.ResidualFactorAdaptationRegressionTest(**args).run(facs)



def splitOnDemographic():
    outcome = ["depression_past_any"]
    args = {
        "json" : "DS4UD_Tests8Class.json",
        "db" : 'ds4ud_adapt',
        "table" : 'msgs_ema_words_day_v9',
        "correlField" : 'user_id',
        "outcomeTable" : "survey_outcomes_waves_agg_v9_more4_Black_Dep_BinAge",
        "outcomeFields" : outcome,
        "outcomeControls" : [],
        "groupFreqThresh" : 0,
        "featTables" : ['feat$roberta_la_meL23con$msgs_ema_words_day_v9$user_id']
    }

    adaptationFactors = [["age_binarized", "is_female", "is_black"]]

    for facs in adaptationFactors:
        args["outcomeControls"] = facs
        RegAndClassTests.FactorAdaptationClassificationTest(**args).run([0])




def ctlbTests():

    outcome = ["heart_disease", "suicide", "life_satisfaction", "perc_fair_poor_health"]
    args = {
        "json" : "CTLB_1grams_SingleControls.json",
        "db" : 'ai_fairness',
        "table" : 'msgs_100u',
        "correlField" : 'cnty',
        "outcomeTable" : "combined_county_outcomes",
        "outcomeFields" : outcome,
        "outcomeControls" : [],
        "groupFreqThresh" : 0,
        "featTables" : ['feat$1gram$msgs_100u$cnty$16to16$0_9']##['feat$1gram$msgs_100u$cnty$16to16']#
    }
    adaptationFactors = [["logincomeHC01_VC85ACS3yr$10"], ["hsgradHC03_VC93ACS3yr$10"], ["forgnbornHC03_VC134ACS3yr$10"]]#[["logincomeHC01_VC85ACS3yr$10", "hsgradHC03_VC93ACS3yr$10", "forgnbornHC03_VC134ACS3yr$10"]]#, ["logincomeHC01_VC85ACS3yr$10"], ["hsgradHC03_VC93ACS3yr$10"], ["forgnbornHC03_VC134ACS3yr$10"]]

    for facs in adaptationFactors:
        args["outcomeControls"] = facs
        args["outcomeFields"] = outcome
        DLATKTests.RegressionTest(**args).run()
        
        DLATKTests.ResidualControlRegressionTest(**args).run()
        args["outcomeFields"] = outcome + facs
        DLATKTests.FactorAdaptationRegressionTest(**args).run(facs)
        DLATKTests.ResidualFactorAdaptationRegressionTest(**args).run(facs)

    #DLATKTests.ResidualFactorAdaptationRegressionTest(outcomeControls = list(set(test.OUTCOME_CONTROLS) - set(adaptationFactorsToRemove)), outcomeFields=test.OUTCOME_FIELDS + adaptationFactors).run(adaptationFactors=adaptationFactors)
    
    
    #adaptationFactors = ["logincomeHC01_VC85ACS3yr$10", "hsgradHC03_VC93ACS3yr$10"]




if __name__ == "__main__":
    main()
