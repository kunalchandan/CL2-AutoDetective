# Logging information

## General Architecture
Python logging provides support for Logging by handler.
We can pick a logger by using
```python
logger = logging.getLogger(__name__)
```


## Installation
Install PostgreSQL server.

Since there is no good reason to use an older version we wil use the latest major version (15).
Follow instructions here https://www.postgresql.org/download/linux/ubuntu/
```bash
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install postgresql
```

## Usage
Ensure postgres is running with:
```bash
sudo service postgresql start
# Or
sudo service postgresql restart
```

Create the database with:
```bash
createdb -U postgres autodetective
```

Connect on the command line with.
```bash
psql -U postgres
# Or
psql -U detective -d autodetective -W
```
You will be prompted to enter the password which is located in `db_connection.py` the password is `auto`

Create a user with:
```sql
CREATE USER detective WITH PASSWORD 'auto';
GRANT ALL PRIVILEGES ON DATABASE autodetective TO detective;
```

Disconnect on the command line with.
```bash
\q
```

# Moving to the secondary drive
# THIS IS TODO
# THIS FAILS BECUASE OF CONFIG ISSUES

https://www.digitalocean.com/community/tutorials/how-to-move-a-postgresql-data-directory-to-a-new-location-on-ubuntu-22-04

Edit the postgres configuration file at:
```bash
sudo service postgresql stop
sudo rsync -az /var/lib/postgresql /media/e5_5044/OSDisk/auto_detective_logs/postgresql/15/main
sudo nano /etc/postgresql/15/main/postgresql.conf

```
Alter the data directory:

```
data_directory = '/media/e5_5044/OSDisk/auto_detective_logs/main/'
```
Restart the Postgres server

```bash
sudo service postgresql start
```

```bash
sudo lsof -i -P -n | grep postgres
```