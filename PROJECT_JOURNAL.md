## P10 - Informative Missingness Project Journal

### 2025-05-09
- [x] Attended a general meeting (10 AM)
    - Key discussions:
        - Google Scholar / Scopus for finding relevant papers
        - FAU VPN 
        - creating virtual environments
        - LaTeX 
        - HPC training
        - Documenting and presenting work
- [x] Initialized the project structure and repository.
- [x] Set up `.ssh` keys for GitHub access.
- [x] Decided to log progress in the `PROJECT_JOURNAL.md`.
- [x] Created HPC account and accepted the invitation
- [x] Initially I had downloaded incomplete data, so had to download it again

---

### 2025-05-10
- [x] Decided to use the docker version of `postgres` 
- [x] Configured the `docker-compose.yml` file with:
    - initial mount of `postgres/create.sql` and `postgres/load.sql`
- [x] The `create.sql` worked smoothly upon `docker-compose.yml` 
- [x] For loading the data I had to do it manually using the scripts below:
    - Copy `postgres/load.sql` to `load_mimic.sql`
        - `docker cp postgres/load.sql mimiciv_postgres:load_mimic.sql`
    - Then docker execute the `load_mimic.sql`
        - `docker exec -it mimiciv_postgres psql -U postgres -d mimiciv -f /load_mimic.sql`
    - After waiting for sometime, had to test it. So the following query is used to login to postgres and displaying the data from the `mimiciv_hosp.admissions` table.
        - `docker exec mimiciv_postgres psql -U postgres -d mimiciv -c "SELECT * FROM mimiciv_hosp.admissions;"`
    - Loading the `MIMIC-IV` data into PostgreSQL failed and terminated  because of millions of data. I had to increase the memory size in the [docker-compose.yml](./docker-compose.yml) file.
    ```bash
    psql:/load_mimic.sql:51: server closed the connection unexpectedly
            This probably means the server terminated abnormally
            before or while processing the request.
    FATAL:  terminating connection due to administrator command
    CONTEXT:  COPY inputevents, line 2912337: "12634755,25987676,36738588,..."
    psql:/load_mimic.sql:51: invalid socket
    psql:/load_mimic.sql:51: error: connection to server was lost

    ```
    - I had to increase the memory size and increase the number of CPUs to handle the overloading.
    ```bash
    deploy:
        resources:
            limits:
                cpus: '12'
                memory: 4G
            reservations:
                memory: 4G
    ```
    
### 2025-05-11
- [x] Created `utils/config.py` to store database settings
- [x] Created `utils/database.py` to:
    - connect to database
    - show list of all tables in schema
    - read a table table into a dataframe with limit
- [x] Made it possible to import into notebook using the command `pip install -e .`
- [x] Created a `notebooks/exp_2025.ipynb` for teseting the package