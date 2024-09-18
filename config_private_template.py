"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-28
Python Version: 3.11
"""

# example of settings for Base DB service
DB_USER = "your username"
DB_PWD = "your pwd"
DB_HOST_IP = "your ip"
DB_SERVICE = "your service name"

# example of settings for ADB (23AI +)
# remember to set ADB = True in config.py
DB_USER = "your username"
DB_PWD = "your pwd"
# must be the connection string in tnsnames.ora
DSN = "your dsn"

# needed to connect to ADB
# if not ADB set to "" (empty string), it is not used but imported
TNS_ADMIN = "path of wallet dir"
WALLET_PWD = "your wallet pwd"

# OCI (9/05) moved to new tenant
COMPARTMENT_ID = "ocid of your compartment"
