"""Run this on a machine with internet access (huggingface.co reachable) and
the real dependencies installed: `pip install -r requirements.txt`.

Produces `appsretrieval_results.json` — upload this to your GitHub release
per the submission guidelines.
"""
import json

import mteb
from sentence_transformers import SentenceTransformer

from src.encoder import PrePostPipelineEncoder

# Swap this for whichever CPU-friendly code embedding model you settle on.
MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"


def main() -> None:
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    encoder = PrePostPipelineEncoder(embed_fn=model.encode)

    task = mteb.get_task("AppsRetrieval")  # the required task
    result = mteb.evaluate(
        encoder,
        [task],
        encode_kwargs={"batch_size": 64},
    )

    task_result = list(result.task_results)[0]
    with open("appsretrieval_results.json", "w") as f:
        json.dump(task_result.to_dict(), f, indent=2)

    print("Wrote appsretrieval_results.json")
    print(task_result)


if __name__ == "__main__":
    main()
