import argparse

import yaml
import requests


def build_defaults(url, projects, **kwargs):
    presets = {}

    for project in projects:
        config = {"constraints": {"project": project,}}
        params = {
            "format": "application/solr+json",
            "limit": 1,
            "project": project,
            "type": "File",
        }
        response = requests.get(url, params=params, **kwargs)

        data = response.json()

        config["fields"] = list(data["response"]["docs"][0])

        presets[project] = config

    return yaml.dump({"presets": presets})


def main():  # pragma: no cover
    desc = """
Queries ESGF index node to generate list of facets for a one or more projects.
"""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--url",
        help="Full URL of the search endpoint",
        default="https://esgf-node.llnl.gov/esg-search/search",
    )
    parser.add_argument(
        "-p",
        "--project",
        action="extend",
        nargs="+",
        type=str,
        help="Name of projects to query",
    )

    args = parser.parse_args()

    output = build_defaults(args.url, args.project)

    print(output)
