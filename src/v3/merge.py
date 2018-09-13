import pandas as pd

prev = 'z_test_xgb_'

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

df = pd.DataFrame()

for target in class_names:
    filename = "../../output/v3/" + prev + target + '.csv'
    df[prev + target] = pd.read_csv(filename)

