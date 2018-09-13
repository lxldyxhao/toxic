import gc
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import time
from scipy.sparse import load_npz, save_npz

time_begin = time.time()
import getopt, sys

# try:
#     opts, args = getopt.getopt(sys.argv[1:], "t:")
# except getopt.GetoptError:
#     print('parameters: xxxxx.py -t <target>')
#     sys.exit(2)
#
# class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
#
# train_data = pd.read_csv('../../output/v1/train_data.csv').fillna(' ')
# test_feats = pd.read_csv('../../output/v1/test_feats.csv').fillna(' ')
#
# z_train = pd.DataFrame(train_data['id'][:])
# z_test = pd.DataFrame(test_feats['id'][:])
# for m in ["xgb", "lgb", "lr"]:
#     for n in class_names:
#         col_name = m + '_' + n
#         train_file = "../../output/v3/z_train_" + col_name + '.csv'
#         test_file = "../../output/v3/z_test_" + col_name + '.csv'
#         z_train[col_name] = pd.read_csv(train_file).values
#         z_test[col_name] = pd.read_csv(test_file).values
#
# lstm_kf = KFold(n_splits=5, shuffle=True, random_state=36)
# lstm_ztrain_df = pd.read_csv("../../output/v3/lstm_cnn_stacking_train_0.csv")
# lstm_ztest_df = pd.read_csv("../../output/v3/lstm_cnn_stacking_test_0.csv").drop("id", axis=1)
# for i, (train_index, test_index) in enumerate(lstm_kf.split(train_data)):
#     if i != 0:
#         data_part = pd.read_csv("../../output/v3/lstm_cnn_stacking_train_"+str(i)+".csv")
#         lstm_ztrain_df.iloc[test_index] = data_part.iloc[test_index]
#         lstm_ztest_df = lstm_ztest_df + pd.read_csv("../../output/v3/lstm_cnn_stacking_test_"+str(i)+".csv").drop("id", axis=1)
# lstm_ztest_df = lstm_ztest_df * 0.2
#
# gru_kf = KFold(n_splits=5, shuffle=True, random_state=64)
# gru_ztrain_df = pd.read_csv("../../output/v3/gru_stacking_train_0.csv")
# gru_ztest_df = pd.read_csv("../../output/v3/gru_stacking_test_0.csv").drop("id", axis=1)
# for i, (train_index, test_index) in enumerate(gru_kf.split(train_data)):
#     if i != 0:
#         data_part = pd.read_csv("../../output/v3/gru_stacking_train_"+str(i)+".csv")
#         gru_ztrain_df.iloc[test_index] = data_part.iloc[test_index]
#         gru_ztest_df = gru_ztest_df + pd.read_csv("../../output/v3/gru_stacking_test_"+str(i)+".csv").drop("id", axis=1)
# gru_ztest_df = gru_ztest_df * 0.2
#
#
#
# for target in class_names:
#     z_train['lstm_' + target] = lstm_ztrain_df["lstm_stacking_train_"+str(target)]
# for target in class_names:
#     z_train['gru_' + target] = gru_ztrain_df["gru_stacking_train_"+str(target)]
# for target in class_names:
#     z_test['lstm_' + target] = lstm_ztest_df["lstm_stacking_test_"+str(target)]
# for target in class_names:
#     z_test['gru_' + target] = gru_ztest_df["gru_stacking_test_" + str(target)]
#
#
# z_train.to_csv("../../output/v3/all_stacking_train.csv", index=False)
# z_test.to_csv("../../output/v3/all_stacking_test.csv", index=False)

z_train = pd.read_csv("../../output/v3/all_stacking_train.csv")
z_test = pd.read_csv("../../output/v3/all_stacking_test.csv")
train_data = pd.read_csv('../../output/v1/train_data.csv').fillna(' ')

print('Loaded')

submission = pd.DataFrame.from_dict({'id': z_test['id']})


def get_tuned_lgb(target):
    return LGBMClassifier(
            learning_rate=0.1,
            # n_estimators=442,
            # num_leaves=100,
            # max_depth=-1,
            # min_split_gain=0.,
            # min_child_weight=1e-3,
            # min_child_samples=20,
            # subsample=1.,
            # subsample_freq=1,
            # colsample_bytree=1.,
            # reg_alpha=0.,
            # reg_lambda=0.,
            random_state=6,
            n_jobs=-1,
            silent=False, )



    # if target == 'toxic':
    #     return LGBMClassifier(
    #         learning_rate=0.2,
    #         n_estimators=442,
    #         num_leaves=100,
    #         max_depth=-1,
    #         min_split_gain=0.,
    #         min_child_weight=1e-3,
    #         min_child_samples=20,
    #         subsample=1.,
    #         subsample_freq=1,
    #         colsample_bytree=1.,
    #         reg_alpha=0.,
    #         reg_lambda=0.,
    #         random_state=36,
    #         n_jobs=4,
    #         silent=False, )
    # elif target == 'severe_toxic':
    #     return LGBMClassifier(
    #         learning_rate=0.2,
    #         n_estimators=83,
    #         num_leaves=100,
    #         max_depth=-1,
    #         min_split_gain=0.,
    #         min_child_weight=1e-3,
    #         min_child_samples=20,
    #         subsample=1.,
    #         subsample_freq=1,
    #         colsample_bytree=1.,
    #         reg_alpha=0.,
    #         reg_lambda=0.,
    #         random_state=36,
    #         n_jobs=4,
    #         silent=False, )
    # elif target == 'obscene':
    #     return LGBMClassifier(
    #         learning_rate=0.2,
    #         n_estimators=135,
    #         num_leaves=63,
    #         max_depth=-1,
    #         min_split_gain=0.,
    #         min_child_weight=1e-3,
    #         min_child_samples=20,
    #         subsample=1.,
    #         subsample_freq=1,
    #         colsample_bytree=1.,
    #         reg_alpha=0.,
    #         reg_lambda=0.,
    #         random_state=36,
    #         n_jobs=4,
    #         silent=False, )
    # elif target == 'threat':
    #     return LGBMClassifier(
    #         learning_rate=0.2,
    #         n_estimators=126,
    #         num_leaves=31,
    #         max_depth=-1,
    #         min_split_gain=0.,
    #         min_child_weight=1e-3,
    #         min_child_samples=20,
    #         subsample=1.,
    #         subsample_freq=1,
    #         colsample_bytree=1.,
    #         reg_alpha=0.,
    #         reg_lambda=0.,
    #         random_state=36,
    #         n_jobs=4,
    #         silent=False, )
    # elif target == 'insult':
    #     return LGBMClassifier(
    #         learning_rate=0.2,
    #         n_estimators=367,
    #         num_leaves=100,
    #         max_depth=-1,
    #         min_split_gain=0.,
    #         min_child_weight=1e-3,
    #         min_child_samples=20,
    #         subsample=1.,
    #         subsample_freq=1,
    #         colsample_bytree=1.,
    #         reg_alpha=0.,
    #         reg_lambda=0.,
    #         random_state=36,
    #         n_jobs=4,
    #         silent=False, )
    # elif target == 'identity_hate':
    #     return LGBMClassifier(
    #         learning_rate=0.2,
    #         n_estimators=119,
    #         num_leaves=100,
    #         max_depth=-1,
    #         min_split_gain=0.,
    #         min_child_weight=1e-3,
    #         min_child_samples=20,
    #         subsample=1.,
    #         subsample_freq=1,
    #         colsample_bytree=1.,
    #         reg_alpha=0.,
    #         reg_lambda=0.,
    #         random_state=36,
    #         n_jobs=4,
    #         silent=False, )


def modelfit(model, X_train, y_train, useTrainCV=True, cv_fold=3, early_stopping_rounds=50):
    if useTrainCV:
        lgb_param = model.get_params()
        lgtrain = lgb.Dataset(X_train, label=y_train)

        cv_params = {'learning_rate': lgb_param['learning_rate'],
                     'num_iterations': lgb_param['n_estimators'],
                     'max_depth': lgb_param['max_depth'],
                     'num_leaves': lgb_param['num_leaves'],
                     'min_child_samples': lgb_param['min_child_samples'],
                     'min_child_weight': lgb_param['min_child_weight'],
                     'min_split_gain': lgb_param['min_split_gain'],
                     'subsample': lgb_param['subsample'],
                     'subsample_freq': lgb_param['subsample_freq'],
                     'colsample_bytree': lgb_param['colsample_bytree'],
                     'lambda_l1': lgb_param['reg_alpha'],
                     'lambda_l2': lgb_param['reg_lambda'],
                     'nthread': lgb_param['n_jobs'],
                     'application': 'binary',
                     'data_random_seed': 36,
                     'metric': 'auc',
                     'verbosity': 1}

        print('交叉验证自动设定迭代次数...')
        cvresult = lgb.cv(cv_params, lgtrain,
                          nfold=cv_fold,
                          metrics='auc',
                          early_stopping_rounds=early_stopping_rounds,
                          seed=36,
                          callbacks=[lgb.print_evaluation(period=10, show_stdv=True)])
        print('cv完成 n_estimators：', len(cvresult['auc-mean']))
        model.set_params(n_estimators=len(cvresult['auc-mean'])
                         )
    # fit
    print('模型拟合中...')
    model.fit(X_train,
              y_train,
              eval_metric='auc')


feat_importaces = pd.DataFrame()
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for target in class_names:
    gc.collect()
    print(target)

    lgb_model = get_tuned_lgb(target)
    # =============== single model ===============

    print("Start training: ", target)
    modelfit(lgb_model,
             X_train=z_train.drop('id', axis=1),
             y_train=train_data[target],
             useTrainCV=True)

    submission[target] = lgb_model.predict_proba(z_test.drop('id', axis=1))[:, 1]
    feat_importaces[target] = pd.Series(lgb_model.feature_importances_)
submission.to_csv('../../output/v4/lgb_stacking_submission.csv', index=False)
feat_importaces.to_csv('../../output/v4/stacking_feat_imp.csv', index=False)

# # ==================== Grid Search ====================
# print(target)
#
# train_sparse_matrix = load_npz("../../output/v2/xgb_train_less_" + target + ".npz")
# test_sparse_matrix = load_npz("../../output/v2/xgb_test_less_" + target + ".npz")
#
# # # small data for test
# # train_sparse_matrix = train_sparse_matrix[:100, :]
# # test_sparse_matrix = train_sparse_matrix[:100, :]
# # train_data = train_data[:100]
#
# lgb_model = get_tuned_lgb(target)
# print("tuning: ", target)
#
# #  格点搜索参数
# param_test = {}
# # if target == 'threat':
# #     param_test = {
# #         'max_depth': [5, 8, 11]  # -1
# #     }
# # else :
# #     param_test = {
# #         'max_depth': [10, 15, 20]  # -1
# #     }
#
#
# gsearch = GridSearchCV(
#     estimator=lgb_model,
#     param_grid=param_test,
#     scoring='roc_auc',
#     iid=False,
#     cv=3,
#     n_jobs=3,
# )
#
# # print('正在搜索...')
# gsearch.fit(train_sparse_matrix, train_data[target])
#
# # 输出搜索结果
# print('\n当前格点搜索的参数：\n', param_test)
# print('\ngsearch.best_params_:', gsearch.best_params_, )
# print('\ngsearch.best_score_:', gsearch.best_score_, )

time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
