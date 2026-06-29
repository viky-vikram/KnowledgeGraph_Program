import logging
import os
import traceback

from dotenv import load_dotenv

import truststore

truststore.inject_into_ssl()

from neo4j import GraphDatabase


def main() -> None:
    load_dotenv(override=True)

    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE")

    print("Connection settings:")
    print(f"URI      : {uri}")
    print(f"Username : {username}")
    print(f"Database : {database}")
    print("Password : [hidden]")
    print()

    logging.basicConfig(level=logging.INFO)

    try:
        with GraphDatabase.driver(
            uri,
            auth=(username, password),
        ) as driver:

            driver.verify_connectivity()

            print("Neo4j AuraDB connected successfully!")

            records, _, _ = driver.execute_query(
                "RETURN 1 AS result"
            )

            print(f"Query result: {records[0]['result']}")

    except Exception as error:
        print("\nConnection failed")
        print(f"Error type   : {type(error).__name__}")
        print(f"Error message: {error}")
        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()