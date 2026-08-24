DROP VIEW IF EXISTS v4_request_log;
DROP VIEW IF EXISTS v4_route_candidate_log;
DROP VIEW IF EXISTS v4_candidate_decision_log;
DROP VIEW IF EXISTS v4_event_log;
DROP VIEW IF EXISTS v4_mature_label_log;
DROP VIEW IF EXISTS v4_training_example_log;
DROP VIEW IF EXISTS v4_checkpoint_log;

CREATE VIEW v4_request_log AS
SELECT * FROM file('v4/v4_request_log.parquet', Parquet);

CREATE VIEW v4_route_candidate_log AS
SELECT * FROM file('v4/v4_route_candidate_log.parquet', Parquet);

CREATE VIEW v4_candidate_decision_log AS
SELECT * FROM file('v4/v4_candidate_decision_log.parquet', Parquet);

CREATE VIEW v4_event_log AS
SELECT * FROM file('v4/v4_event_log.parquet', Parquet);

CREATE VIEW v4_mature_label_log AS
SELECT * FROM file('v4/v4_mature_label_log.parquet', Parquet);

CREATE VIEW v4_training_example_log AS
SELECT * FROM file('v4/v4_training_example_log.parquet', Parquet);

CREATE VIEW v4_checkpoint_log AS
SELECT * FROM file('v4/v4_checkpoint_log.parquet', Parquet);
