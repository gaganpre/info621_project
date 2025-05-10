#!/usr/bin/env python3

import requests
import logging
import simplejson as json
from langchain_community.utilities import ArxivAPIWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAlexAPI:
    def __init__(self, query):
        
        self.base_url = "https://api.openalex.org/works"
        self.headers = {
            'Accept': 'application/json',
        }
        self.query = query
        self.query_alex_repsone = None
        self.cites = None
        self.citation_url = None

    def get_openalex_id(self, page=1):
        """
        Fetch OpenAlex IDs based on a query.
        """
        params = {
            'search': self.query,
            'page': page,
        }
        response = requests.get(self.base_url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            # just return the first response
            if not data['results']:
                logger.warning(f"No results found for query: {self.query}")
                self.query_alex_repsone = None
            self.query_alex_repsone = data['results'][0]
        else:
            logger.error(f"Error fetching OpenAlex IDs: {response.status_code}")
            self.query_alex_repsone = None
            raise Exception(f"Error fetching OpenAlex IDs: {response.status_code}")
    
    def get_citation_url(self):
        """
        Get the citation URL for a given OpenAlex ID.
        """
        if self.query_alex_repsone:
            self.citation_url = self.query_alex_repsone.get('cited_by_api_url', None)

        else:
            logger.warning(f"No OpenAlex ID found for query: {self.query}")
            self.citation_url = None
            raise Exception(f"No OpenAlex ID found for query: {self.query}")

    def query_citation_url(self):
        """
        Fetch citation data from OpenAlex using a citation URL.
        """
        response = requests.get(self.citation_url, headers=self.headers)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                self.cites = data.get('results', [])
        else:
            logger.error(f"Error fetching citation data: {response.status_code}")
            self.cites = None
            raise  Exception(f"Error fetching citation data: {response.status_code}")

    def get_citations(self):
        """
        Fetch citations, related works and references for each one of them for a given OpenAlex ID.
        """
        self.get_openalex_id()
        self.get_citation_url()
        self.query_citation_url()
        
        if self.cites:
            citations = {}
            for cite in self.cites:
                citation_data = {
                    'title': cite.get('title', None),
                    # 'doi': cite.get('doi', None),
                    'openalex_id': cite.get('id', None),
                    'cited_by_count': cite.get('cited_by_count', 0),
                    # 'abstract': cite.get('abstract', None),
                    'publication_year': cite.get('publication_year', None),
                    'related_works': cite.get('related_works', [])[:5],
                    'references': cite.get('referenced_works', [])[:5],
                }
                citations.update({cite.get('id', None): citation_data})
            return { self.query_alex_repsone.get('id', "root"): citations }
            
        else:
            logger.warning(f"No citations found for OpenAlex ID")
            return []


if __name__ == "__main__":
    query = "Attention is all you need"
    openalex_api = OpenAlexAPI(query)
    _citations = openalex_api.get_citations()

    with open("citations.json", "w") as _file:
        json.dump(_citations, _file, indent=4)