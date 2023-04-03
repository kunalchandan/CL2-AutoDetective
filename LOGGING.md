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
```

Create a user with:
```sql
CREATE USER detective WITH PASSWORD 'auto';
GRANT ALL PRIVILEGES ON DATABASE autodetective TO detective;
```

Disconnect on the command line with.
```bash
\q
```
