import os
import json
import configparser
import argparse
from typing import List, Optional
from dataclasses import dataclass
from huggingface_hub import login, upload_file


@dataclass
class ModelConfig:
    dataset_name: str
    model_name: str
    file_dir: str
    repo_name: str


@dataclass
class DatasetConfig:
    dataset_name: str
    file_dir: str
    repo_name: str


class HuggingFaceUploader:
    def __init__(
        self,
        org_name: str,
        stored_tokens_path: str,
        model_files: Optional[List[str]] = None,
        gen_dataset_files: Optional[List[str]] = None,
    ):
        self.org_name = org_name
        self.stored_tokens_path = stored_tokens_path
        self.token = self._get_token()
        login(self.token)

        if model_files:
            self.model_files = model_files
        else:
            self.model_files = [
                "config.json",
                "generation_config.json",
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
                "model.safetensors.index.json",
                "special_tokens_map.json",
                "spiece.model",
                "tokenizer.json",
                "tokenizer_config.json",
            ]
        if gen_dataset_files:
            self.gen_dataset_files = gen_dataset_files
        else:
            self.gen_dataset_files = [
                "queries_Llama-3.1-8B_Zero-shot.jsonl",
                "queries_Llama-3.1-8B_Zero-shot_filtered_all.jsonl",
                "queries_Llama-3.1-8B_Zero-shot_triplets.tsv",
                "queries_Llama-3.1-8B_CoT.jsonl",
                "queries_Llama-3.1-8B_CoT_filtered_all.jsonl",
                "queries_Llama-3.1-8B_CoT_triplets.tsv",
            ]

    def _get_token(self) -> str:
        """Retrieve the token for the given organization."""
        if os.path.exists(self.stored_tokens_path):
            config = configparser.ConfigParser()
            config.read(self.stored_tokens_path)

            if self.org_name in config:
                token = config[self.org_name].get("hf_token", None)
                if token:
                    print(
                        f"Successfully retrieved token for '{self.org_name}' at: {self.stored_tokens_path}"
                    )
                    return token

        return input(
            f"Token for {self.org_name} not found. Please paste the token: "
        ).strip()

    def upload_model_files(self, model: ModelConfig) -> None:
        """Upload model files to Hugging Face Hub."""
        print(
            f"Uploading model files for dataset: {model.dataset_name}, model: {model.model_name}"
        )

        for file in self.model_files:
            try:
                upload_file(
                    path_or_fileobj=f"{model.file_dir}/{file}",
                    path_in_repo=file,
                    repo_id=f"{self.org_name}/{model.repo_name}",
                    repo_type="model",
                )
                print(f"Uploaded: {file}")
            except Exception as e:
                print(f"Failed to upload {file}: {e}")

    def upload_dataset_files(self, dataset: DatasetConfig) -> None:
        """Upload dataset files to Hugging Face Hub."""
        print(f"Uploading dataset files for: {dataset.dataset_name}")

        for file in self.gen_dataset_files:
            try:
                upload_file(
                    path_or_fileobj=f"{dataset.file_dir}/{file}",
                    path_in_repo=f"{dataset.dataset_name.lower()}/{file}",
                    repo_id=f"{self.org_name}/{dataset.repo_name}",
                    repo_type="dataset",
                )
                print(f"Uploaded: {file}")
            except Exception as e:
                print(f"Failed to upload {file}: {e}")

    def process_models(self, models: List[ModelConfig]) -> None:
        """Process model uploads only."""
        print("\n=== Processing Model Uploads ===")
        for model in models:
            self.upload_model_files(model)

    def process_datasets(self, datasets: List[DatasetConfig]) -> None:
        """Process dataset uploads only."""
        print("\n=== Processing Dataset Uploads ===")
        for dataset in datasets:
            self.upload_dataset_files(dataset)


def load_config(config_path: str) -> tuple[List[ModelConfig], List[DatasetConfig], str]:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        config_data = json.load(f)

    models_base_dir = config_data["config"]["models_base_dir"]
    gen_data_base_dir = config_data["config"]["gen_data_base_dir"]
    models = []
    datasets = []

    for dataset_name, dataset_info in config_data["datasets"].items():
        for model_name, model_info in dataset_info["models"].items():
            models.append(
                ModelConfig(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    file_dir=os.path.join(models_base_dir, model_info["file_dir"]),
                    repo_name=model_info["repo_name"],
                )
            )

        dataset_file_info = dataset_info["dataset_files"]
        datasets.append(
            DatasetConfig(
                dataset_name=dataset_name,
                file_dir=os.path.join(gen_data_base_dir, dataset_file_info["file_dir"]),
                repo_name=dataset_file_info["repo_name"],
            )
        )

    return models, datasets


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload models and datasets to Hugging Face Hub"
    )
    parser.add_argument(
        "--org-name",
        default="inpars-plus",
        help="Organization name on Hugging Face",
    )
    parser.add_argument(
        "--stored-tokens-path",
        default="/scratch-shared/scur2880/.cache/huggingface/stored_tokens",
        help="Path to the stored tokens file",
    )
    parser.add_argument(
        "--config",
        default="config/hf_uploads_config.json",
        help="Path to the upload configuration JSON file",
    )
    parser.add_argument(
        "--model-files",
        nargs="+",
        help="Model files to upload to Hugging Face Hub. If not specified, will use default files.",
    )
    parser.add_argument(
        "--gen-dataset-files",
        nargs="+",
        help="Dataset files to upload to Hugging Face Hub. If not specified, will use default files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    models, datasets = load_config(args.config)

    uploader = HuggingFaceUploader(
        org_name=args.org_name,
        stored_tokens_path=args.stored_tokens_path,
        model_files=args.model_files,
        gen_dataset_files=args.gen_dataset_files
    )

    uploader.process_models(models)
    uploader.process_datasets(datasets)


if __name__ == "__main__":
    main()
