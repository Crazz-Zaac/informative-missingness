-----------------------------------------
-- Load data into the MIMIC-IV schemas --
-----------------------------------------

-- This script is configured to run inside the Docker container
-- The data is mounted at /mimiciv in the container
-- No need to specify mimic_data_dir when executing inside Docker
-- The script will be executed automatically by the init scripts

-- making sure correct encoding is defined as -utf8- 
SET CLIENT_ENCODING TO 'utf8';

-- Set path to where data is mounted in the container
\cd /mimiciv

-- hosp schema
\cd hosp

\COPY mimiciv_hosp.admissions FROM admissions.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.d_hcpcs FROM d_hcpcs.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.diagnoses_icd FROM diagnoses_icd.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.d_icd_diagnoses FROM d_icd_diagnoses.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.d_icd_procedures FROM d_icd_procedures.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.d_labitems FROM d_labitems.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.drgcodes FROM drgcodes.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.emar_detail FROM emar_detail.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.emar FROM emar.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.hcpcsevents FROM hcpcsevents.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.labevents FROM labevents.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.microbiologyevents FROM microbiologyevents.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.omr FROM omr.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.patients FROM patients.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.pharmacy FROM pharmacy.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.poe_detail FROM poe_detail.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.poe FROM poe.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.prescriptions FROM prescriptions.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.procedures_icd FROM procedures_icd.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.provider FROM provider.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.services FROM services.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_hosp.transfers FROM transfers.csv DELIMITER ',' CSV HEADER NULL '';

-- icu schema
\cd ../icu

\COPY mimiciv_icu.caregiver FROM caregiver.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_icu.chartevents FROM chartevents.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_icu.datetimeevents FROM datetimeevents.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_icu.d_items FROM d_items.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_icu.icustays FROM icustays.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_icu.ingredientevents FROM ingredientevents.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_icu.inputevents FROM inputevents.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_icu.outputevents FROM outputevents.csv DELIMITER ',' CSV HEADER NULL '';
\COPY mimiciv_icu.procedureevents FROM procedureevents.csv DELIMITER ',' CSV HEADER NULL '';