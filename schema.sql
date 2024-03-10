CREATE TABLE IF NOT EXISTS experiments(
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_folder VARCHAR(100) NOT NULL,
    name VARCHAR(100),
    run_id_by_date INTEGER,
    run_status VARCHAR(20),
    start_time TIME NOT NULL,
    end_time TIME NULL,
    elapsed_time VARCHAR(100) NULL
);

CREATE TABLE IF NOT EXISTS experiment_details(
    run_id INTEGER PRIMARY KEY,
    model VARCHAR(50) NOT NULL,
    task VARCHAR(20) NOT NULL,
    enable_drug BOOLEAN NOT NULL,
    enable_mutation BOOLEAN NOT NULL,
    enable_cnv BOOLEAN NOT NULL,
    enable_gene_expression BOOLEAN NOT NULL,
    enable_methylation BOOLEAN NOT NULL,
    data_study VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS test_scores(
    run_id INT,
    fold INT,
    metric VARCHAR(50),
    value FLOAT,
    PRIMARY KEY(run_id, fold, metric)
);