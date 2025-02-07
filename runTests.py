import src.RegAndClassTests as RegAndClassTests
import json
import argparse



def main():
    # Parse command-line arguments
    args = parse_arguments()

    if args.function:
        # Dynamically call the function by name if it exists in the global scope
        if args.function in globals() and callable(globals()[args.function]):
            globals()[args.function]()  # Call the function
            
        else:
            print("Function " + args.function + " not recognized.")

    else:
        # Load arguments from JSON and run the default logic
        with open("testArgs.json", "r") as f:
            args = json.load(f)
        print("Using arguments from JSON:", args)
        RegAndClassTests.RegressionTest(**args).run()



def parse_arguments():
    parser = argparse.ArgumentParser(description="Run specified tests or default function using JSON config.")
    parser.add_argument(
        "function", 
        nargs="?", 
        default=None, 
        help="Specify the function to run (e.g., 'ds4udTests' or 'ctlbTests')."
    )
    return parser.parse_args()



def ds4udTests():
    outcome = ["substance_past_any", "depress_past_any", "anxiety_past_any"]
    args = {
        "csv" : "DS4UD_Outcomes_SingleControlClass.csv",
        "db" : 'ds4ud_adapt',
        "table" : 'msgs_ema_words_day_v9',
        "correlField" : 'user_id',
        "outcomeTable" : "survey_outcomes_waves_agg_v9_mod_v2",
        "outcomeFields" : outcome,
        "outcomeControls" : [],
        "groupFreqThresh" : 0,
        "featTables" : ['feat$roberta_la_meL23con$msgs_ema_words_day_v9$user_id'],
        "featureSelectionString" : "magic_sauce"
    }

    adaptationFactors = [["individual_income"], ["age"], ["is_female"]]

    for facs in adaptationFactors:
        
        args["outcomeFields"] = outcome
        args["outcomeControls"] = facs
        RegAndClassTests.RegressionTest(**args).run()
        #RegAndClassTests.ClassificationTest(**args).run()
        
        #RegAndClassTests.ResidualControlRegressionTest(**args).run()
        #args["outcomeFields"] = outcome + facs
        #args["outcomeControls"] = []
        #RegAndClassTests.FactorAdaptationRegressionTest(**args).run(facs)

        RegAndClassTests.FactorAdaptationClassificationTest(**args).run([0])#22])

        #args["outcomeControls"] = facs
        #args["outcomeControls"] = []
        #RegAndClassTests.ResidualFactorAdaptationRegressionTest(**args).run(facs)



def ctlbTests():

    outcome = ["heart_disease", "suicide", "life_satisfaction", "perc_fair_poor_health"]
    args = {
        "csv" : "CTLB_1grams_SingleControls.csv",
        "db" : 'ai_fairness',
        "table" : 'msgs_100u',
        "correlField" : 'cnty',
        "outcomeTable" : "combined_county_outcomes",
        "outcomeFields" : outcome,
        "outcomeControls" : [],
        "groupFreqThresh" : 0,
        "featTables" : ['feat$1gram$msgs_100u$cnty$16to16$0_9'],
        "featureSelectionString" : "topic_ngram_ms"
    }
    adaptationFactors = [["logincomeHC01_VC85ACS3yr$10"], ["hsgradHC03_VC93ACS3yr$10"], ["forgnbornHC03_VC134ACS3yr$10"]]#[["logincomeHC01_VC85ACS3yr$10", "hsgradHC03_VC93ACS3yr$10", "forgnbornHC03_VC134ACS3yr$10"]]#, ["logincomeHC01_VC85ACS3yr$10"], ["hsgradHC03_VC93ACS3yr$10"], ["forgnbornHC03_VC134ACS3yr$10"]]


    for facs in adaptationFactors:
        args["outcomeControls"] = facs
        args["outcomeFields"] = outcome
        RegAndClassTests.RegressionTest(**args).run()
        
        RegAndClassTests.ResidualControlRegressionTest(**args).run()
        args["outcomeFields"] = outcome + facs
        RegAndClassTests.FactorAdaptationRegressionTest(**args).run(facs)
        RegAndClassTests.ResidualFactorAdaptationRegressionTest(**args).run(facs)


def ctlbTests2():

    outcome = ["heart_disease", "suicide", "life_satisfaction", "perc_fair_poor_health"]
    args = {
        "csv" : "CTLB_1grams_SingleControls_age.csv",
        "db" : 'ai_fairness',
        "table" : 'msgs_100u',
        "correlField" : 'cnty',
        "outcomeTable" : "combined_county_outcomes",
        "outcomeFields" : outcome,
        "outcomeControls" : [],
        "groupFreqThresh" : 0,
        "featTables" : ['feat$1gram$msgs_100u$cnty$16to16$0_9'],
        "featureSelectionString" : "topic_ngram_ms"
    }
    adaptationFactors = [["perc_65_and_over_chr14_2012"]]#[["logincomeHC01_VC85ACS3yr$10", "hsgradHC03_VC93ACS3yr$10", "forgnbornHC03_VC134ACS3yr$10"]]#, ["logincomeHC01_VC85ACS3yr$10"], ["hsgradHC03_VC93ACS3yr$10"], ["forgnbornHC03_VC134ACS3yr$10"]]


    for facs in adaptationFactors:
        args["outcomeControls"] = facs
        args["outcomeFields"] = outcome
        RegAndClassTests.RegressionTest(**args).run()
        
        RegAndClassTests.ResidualControlRegressionTest(**args).run()
        args["outcomeFields"] = outcome + facs
        RegAndClassTests.FactorAdaptationRegressionTest(**args).run(facs)
        RegAndClassTests.ResidualFactorAdaptationRegressionTest(**args).run(facs)



if __name__ == "__main__":
    main()
