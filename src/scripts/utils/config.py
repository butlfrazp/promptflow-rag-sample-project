import dotenv


class Config(object):
    def __init__(self):
        config = dotenv.dotenv_values()

        self._cognitive_search_endpoint = config.get("COGNITIVE_SEARCH_ENDPOINT")
        self._cognitive_search_api_key = config.get("COGNITIVE_SEARCH_API_KEY")
        self._azure_openai_api_key = config.get("AZURE_OPENAI_API_KEY")
        self._azure_openai_endpoint = config.get("AZURE_OPENAI_ENDPOINT")

    @property
    def cognitive_search_endpoint(self):
        return self._cognitive_search_endpoint

    @property
    def cognitive_search_api_key(self):
        return self._cognitive_search_api_key

    @property
    def azure_openai_api_key(self):
        return self._azure_openai_api_key

    @property
    def azure_openai_endpoint(self):
        return self._azure_openai_endpoint


config = Config()
