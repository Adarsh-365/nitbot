from openai import OpenAI


class NVIDIAEmbeddings:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model_name: str = "nvidia/nv-embed-v1",
        max_batch_size: int = 512,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.max_batch_size = max_batch_size

    def _log(self, message: str):
        print(f"[embeddings] {message}", flush=True)

    def _embed(self, texts, input_type: str):
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "NONE"},
        )
        return [item.embedding for item in response.data]

    def embed_documents(self, texts):
        if not texts:
            return []

        embeddings = []
        total = len(texts)
        total_batches = (total + self.max_batch_size - 1) // self.max_batch_size

        for batch_index, start in enumerate(range(0, total, self.max_batch_size), start=1):
            
            batch = texts[start:start + self.max_batch_size]
            end = start + len(batch)
            self._log(
                f"batch {batch_index}/{total_batches}: embedding items {start + 1}-{end} of {total}"
            )
            embeddings.extend(self._embed(batch, "passage"))
        return embeddings

    def embed_query(self, text):
        return self._embed([text], "query")[0]
