
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.feature_selection import RFE


def feature_selection_by_mdi(train, act, numFeat):
    clf = RandomForestRegressor(n_estimators=100, random_state=42)
    clf.fit(train.values,act.values)
    feature_importances = clf.feature_importances_
    selected_feature_indices = feature_importances.argsort()[-numFeat:][::-1]
    return pd.DataFrame(act).join(train[train.columns[selected_feature_indices]])

def feature_selection_by_wrapper(train, act,numFeat):
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=numFeat)

    selector.fit(train, act)
    selected_feature_indices = selector.support_
    return pd.DataFrame(act).join(train[train.columns[selected_feature_indices]])


if __name__ == '__main__':

    name = 'ACE_coop_ejec1.csv_2100.csv'
    data = pd.read_csv(name)

    mdNames = data.columns[1:]
    actName = data.columns[0]

    feature_selection_by_mdi(data[mdNames],data[actName],8).to_csv(name+"_best_MDI.csv")
    feature_selection_by_wrapper(data[mdNames], data[actName], 8).to_csv(name + "_best_wrapper.csv")



