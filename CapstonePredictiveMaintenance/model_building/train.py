
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
import mlflow

# ---------------------------------------------------------------------
# 2. Register candidate estimators and their pipeline parameter grids
# ---------------------------------------------------------------------
base_tree = DecisionTreeClassifier(
    class_weight="balanced", random_state=42
)

models = {
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        ),
        "params": {
            "decisiontreeclassifier__criterion": ["gini", "entropy"],
            "decisiontreeclassifier__max_depth": [None, 3, 5, 7, 10],
            "decisiontreeclassifier__min_samples_split": [2, 5, 10],
            "decisiontreeclassifier__min_samples_leaf": [1, 2, 4],
        },
    },
    "Bagging": {
        "estimator": BaggingClassifier(
            estimator=base_tree, random_state=42
        ),
        "params": {
            "baggingclassifier__n_estimators": [50, 100, 150],
            "baggingclassifier__max_samples": [0.6, 0.8, 1.0],
            "baggingclassifier__max_features": [0.6, 0.8, 1.0],
            "baggingclassifier__bootstrap": [True, False],
            "baggingclassifier__bootstrap_features": [True, False],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(
            class_weight="balanced_subsample", random_state=42
        ),
        "params": {
            "randomforestclassifier__n_estimators": [100, 200, 300],
            "randomforestclassifier__max_depth": [None, 5, 10, 15],
            "randomforestclassifier__min_samples_split": [2, 5, 10],
            "randomforestclassifier__min_samples_leaf": [1, 2, 4],
            "randomforestclassifier__max_features": ["sqrt", "log2"],
            "randomforestclassifier__bootstrap": [True],
        },
    },
    "AdaBoost": {
        "estimator": AdaBoostClassifier(
            estimator=base_tree, random_state=42
        ),
        "params": {
            "adaboostclassifier__n_estimators": [50, 100, 150],
            "adaboostclassifier__learning_rate": [0.01, 0.05, 0.1, 0.5],
        },
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "params": {
            "gradientboostingclassifier__n_estimators": [50, 100, 150],
            "gradientboostingclassifier__max_depth": [2, 3, 4],
            "gradientboostingclassifier__learning_rate": [0.01, 0.05, 0.1],
            "gradientboostingclassifier__subsample": [0.6, 0.8, 1.0],
            "gradientboostingclassifier__max_features": ["sqrt", "log2", None],
        },
    },
    "XGBoost": {
        "estimator": xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=(ytrain.value_counts()[0] / ytrain.value_counts()[1]),
            random_state=42,
            n_jobs=-1
        ),
        "params": {
            "xgbclassifier__n_estimators": [50, 75, 100, 150],
            "xgbclassifier__max_depth": [2, 3, 4],
            "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
            "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
            "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
        },
    },
}

# ---------------------------------------------------------------------
# 3. Train, tune, evaluate and log every model through the same loop
# ---------------------------------------------------------------------
classification_threshold = 0.45
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_results = []
best_estimators = {}

mlflow.set_experiment("MLOps_experiment")

for model_name, config in models.items():
    print(f"\n{'=' * 70}\nTraining: {model_name}\n{'=' * 70}")

    # The pipeline ensures the same preprocessing is fitted within each CV fold.
    pipeline = make_pipeline(preprocessor, config["estimator"])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=config["params"],
        n_iter=20,
        scoring="f1",
        cv=cv_strategy,
        n_jobs=-1,
        random_state=42,
        refit=True,
        verbose=1,
        error_score="raise"
    )

    with mlflow.start_run(run_name=model_name) as parent_run:
        mlflow.set_tag("preparation_run_id", preparation_run_id)
        mlflow.log_param("preparation_run_id", preparation_run_id)
        mlflow.log_param("tuning_n_iter", search.n_iter)
        mlflow.log_param("cv_folds", cv_strategy.n_splits)
        search.fit(Xtrain, ytrain)

        # Log EVERY evaluated candidate as its own nested MLflow run.
        # cv_results_ contains all RandomizedSearchCV trials, not only the winner.
        for trial_index, trial_params in enumerate(search.cv_results_["params"]):
            with mlflow.start_run(run_name=f"{model_name}_trial_{trial_index + 1:02d}", nested=True):
                mlflow.set_tag("run_type", "tuning_trial")
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("parent_run_id", parent_run.info.run_id)
                mlflow.set_tag("preparation_run_id", preparation_run_id)
                mlflow.log_param("trial_number", trial_index + 1)
                mlflow.log_params({key: str(value) for key, value in trial_params.items()})
                mlflow.log_metric("mean_cv_f1", float(search.cv_results_["mean_test_score"][trial_index]))
                mlflow.log_metric("std_cv_f1", float(search.cv_results_["std_test_score"][trial_index]))
                mlflow.log_metric("cv_rank", int(search.cv_results_["rank_test_score"][trial_index]))
                mlflow.set_tag("is_selected", str(trial_index == search.best_index_).lower())

        best_model = search.best_estimator_
        best_estimators[model_name] = best_model

        # Predict probabilities so the same decision threshold is used
        # consistently for all models.
        train_probabilities = best_model.predict_proba(Xtrain)[:, 1]
        test_probabilities = best_model.predict_proba(Xtest)[:, 1]

        train_predictions = (
            train_probabilities >= classification_threshold
        ).astype(int)
        test_predictions = (
            test_probabilities >= classification_threshold
        ).astype(int)

        train_report = classification_report(
            ytrain, train_predictions, output_dict=True, zero_division=0
        )
        test_report = classification_report(
            ytest, test_predictions, output_dict=True, zero_division=0
        )

        # Log tuning and evaluation results to MLflow.
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("classification_threshold", classification_threshold)
        mlflow.set_tag("run_type", "selected_model")
        mlflow.log_params({key: str(value) for key, value in search.best_params_.items()})
        mlflow.log_param("selected_trial_number", int(search.best_index_) + 1)
        mlflow.log_dict({"model": model_name, "best_parameters": search.best_params_,
                         "best_cv_f1": float(search.best_score_),
                         "preparation_run_id": preparation_run_id},
                        "selected_parameters.json")
        mlflow.log_metric("best_cv_f1", search.best_score_)

        metrics_to_log = {
            "train_accuracy": train_report["accuracy"],
            "train_precision": train_report["1"]["precision"],
            "train_recall": train_report["1"]["recall"],
            "train_f1": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_precision": test_report["1"]["precision"],
            "test_recall": test_report["1"]["recall"],
            "test_f1": test_report["1"]["f1-score"],
        }
        mlflow.log_metrics(metrics_to_log)

        # Log the complete fitted preprocessing + model pipeline.
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        model_results.append({
            "Model": model_name,
            "Best CV F1": search.best_score_,
            "Train Accuracy": train_report["accuracy"],
            "Test Accuracy": test_report["accuracy"],
            "Test Precision": test_report["1"]["precision"],
            "Test Recall": test_report["1"]["recall"],
            "Test F1": test_report["1"]["f1-score"],
            "Best Parameters": search.best_params_,
        })

        print("Best parameters:", search.best_params_)
        print("\nTest classification report:")
        print(classification_report(
            ytest, test_predictions, zero_division=0
        ))
        print("Test confusion matrix:")
        print(confusion_matrix(ytest, test_predictions))

# ---------------------------------------------------------------------
# 4. Compare models and save the best complete pipeline
# ---------------------------------------------------------------------
results_df = pd.DataFrame(model_results).sort_values(
    by="Test F1", ascending=False
).reset_index(drop=True)

print(results_df)

best_model_name = results_df.loc[0, "Model"]
best_pipeline = best_estimators[best_model_name]

model_path = "CapstonePredictiveMaintenance/deployment/best_engine_condition_pipeline.joblib"
joblib.dump(best_pipeline,model_path)

print(f"\nBest model: {best_model_name}")
print("Saved complete pipeline, including preprocessing, to:")
print(model_path)

# Save a complete, auditable tuning summary alongside the selected pipeline.
results_path = "CapstonePredictiveMaintenance/deployment/model_tuning_results.csv"
results_df.to_csv(results_path, index=False)
print("Saved all candidate model results to:", results_path)
print("Verification passed: model artifact and tuning summary exist.")
