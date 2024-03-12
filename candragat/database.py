import sqlite3
import pandas as pd
from datetime import datetime
import json
import os

class DatabaseConnection(object):
    def __init__(self, db_path) -> None:
        self.db_path = db_path
        self._conn = None
        
    def __enter__(self):
        self._conn = sqlite3.connect(self.db_path, timeout=10, isolation_level="EXCLUSIVE")
        return self._conn.cursor()
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._conn.commit()
        self._conn.close()
        
class Experiment(object):
    def __init__(self, storage, run_folder, run_id, safe_mode=True, name=None):
        self.name = name
        self._run_folder = run_folder
        self._run_id = run_id
        self._storage = storage
        self._safe_mode = safe_mode
        self.connection = storage.connection
        self._abs_path = os.path.join(self._storage.dirname,
                "records",
                self._run_folder)
        self.status = "RUNNING"
    
    def _assert_safe_mode(self):
        if self.safe_mode:
            raise ValueError("This operation is not allowed in safe mode")
        
    def report_test_score(self, fold, metric, value):
        self._logger.info(f"Reporting test score for fold {fold} — {metric}: {value}")
        self._assert_safe_mode()
        self._storage.report_test_score(self._run_id, fold, metric, value)
    
    def test_score_dataframe(self):
        return self._storage.test_score_dataframe(self._run_id)
    
    def set_safe_mode(self, value):
        self.safe_mode = value
    
    @property
    def hyperparameters(self):
        return json.read(
            open(os.path.join(
                self._abs_path, 
                "hyperparameters.json"
                ),'r')
            )
        
    def add_hyperparameters(self, hyperparameters: dict):
        json.dump(hyperparameters,open(os.path.join(
                self._abs_path, 
                "hyperparameters.json"
                ),'w'), indent=2)
        
    def set_name(self, name: str):
        self.name = name
        with self.connection as cursor:
           cursor.execute('''UPDATE experiments
                          SET name = ?
                          WHERE run_id = ?
                          ''', [name, self._run_id])
        
    def add_experiment_details(self, details: dict):
        self._assert_safe_mode()
        for k, v in details.items():
            self._logger.info(f"Add experiment details: {k}: {v}")
        with self.connection as cursor:
            cursor.execute('''INSERT INTO experiment_details (
                        run_id, model, task, enable_drug, 
                        enable_mutation, 
                        enable_cnv, 
                        enable_gene_expression, 
                        enable_methylation, 
                        data_study
                        )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', 
                    [self._run_id] + [details[k] for k in 
                                      ["model", 
                                       "task", 
                                       "enable_drug", 
                                       "enable_mutation", 
                                       "enable_cnv", 
                                       "enable_gene_expression", 
                                       "enable_methylation", 
                                       "data_study"]])

    def complete(self):
        self._assert_safe_mode()
        with self.connection as cursor:
            start_time = datetime.strptime(cursor.execute(
                                    "SELECT start_time FROM experiments " +
                                    "WHERE run_id = ?", [self._run_id]).fetchone()[0], "%Y-%m-%d %H:%M:%S")
            end_time = datetime.now()
            elapsed_time = end_time - start_time
            cursor.execute(
                    "UPDATE experiments " +
                    "SET run_status = 'COMPLETE', end_time = ?, elapsed_time = ? " +
                    "WHERE run_id = ?", [end_time, str(elapsed_time).split(".")[0], self._run_id])
        self.status = "COMPLETE"
        self.set_safe_mode(True)
    
    def error_handling(self):
        if self.status != "COMPLETE":
            with self.connection as cursor:
                cursor.execute('''UPDATE experiments 
                        SET run_status = "FAIL"
                        WHERE run_id = ?
                        ''', [self._run_id])
            self.status = "ERROR"
            self.set_safe_mode(True)
            self._logger.error(f"Experiment {self.name} failed")
    
    def set_logger(self, logger):
        self._logger = logger
        
    @property
    def run_folder(self):
        return self._abs_path

class Storage:
    def __init__(self, db_path = "results/storage.db", logger=None):
        logger.info("Initializing database connection at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        dirname = os.path.dirname(db_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        self.connection = DatabaseConnection(db_path)
        with open("schema.sql", 'r') as f:
            with self.connection as cursor:
                for command in f.read().split(";"):
                    cursor.execute(command)
        self._dirname = dirname
        self._logger = logger

    def create_experiment(self):
        with self.connection as cursor:
            cursor.execute('BEGIN EXCLUSIVE')
            res = cursor.execute(
                "SELECT strftime('%Y-%m-%d', start_time) as start_date, COUNT(run_id_by_date) as count FROM experiments " + 
                "WHERE start_date = strftime('%Y-%m-%d', date('now','localtime'))").fetchone()
            run_id_by_date = res[1] + 1
            start_time = datetime.now()
            run_folder = f"{start_time.strftime('%Y_%m_%d')}-{run_id_by_date:03}"
            cursor.execute("INSERT INTO experiments (run_folder, run_id_by_date, start_time, run_status) VALUES (?, ?, ?, ?)", [
                        run_folder, 
                        run_id_by_date, 
                        start_time.strftime("%Y-%m-%d %H:%M:%S"), 
                        "RUNNING"])
            
            run_id = cursor.execute('''SELECT run_id FROM experiments 
                                WHERE run_folder = ?
                                ''', [run_folder]).fetchone()[0]
            
        self._logger.info(f"Creating experiment ")
        abs_run_folder = os.path.join(self._dirname, "records", run_folder)
        experiment = Experiment(self, run_folder, run_id)
        os.makedirs(abs_run_folder)
        return experiment
    
    def load_experiment_by_record(self, name):
        with self.connection as cursor:
            res = cursor.execute("SELECT run_id FROM experiments WHERE run_folder = ?", [name]).fetchone()
            if res is None:
                raise ValueError(f"Experiment '{name}' not found")
            run_id = res[0]
            experiment = Experiment(self, name, run_id)
            return experiment
    
    def report_test_score(self, run_id, fold, metric, value):
        with self.connection as cursor:
            cursor.execute("INSERT INTO test_scores (run_id, fold, metric, value) " +
                    "VALUES (?, ?, ?, ?) ", [run_id, fold, metric, value])
        
    def test_score_dataframe(self, run_id):
        with self.connection as cursor:
            res = cursor.execute(
                '''SELECT fold, metric, value FROM test_scores
                WHERE run_id = ?
                ''', [run_id]
            ).fetchall()
        
        df = pd.DataFrame(res, columns=["fold", "metric", "value"])
        return df.pivot(index="fold", columns="metric", values="value")
    
    @property
    def dirname(self):
        return self._dirname        
    

