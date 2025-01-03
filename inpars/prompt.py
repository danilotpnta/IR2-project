import os
import ftfy
import yaml
import random
import pandas as pd
from typing import List, Optional

with open(f"{os.path.dirname(__file__)}/prompts/templates.yaml") as f:
    templates = yaml.safe_load(f)


class Prompt:
    def __init__(
        self,
        template=None,
        examples=None,
        n_generated_queries=1,
        tokenizer=None,
        max_doc_length=None,
        max_query_length=None,
        max_prompt_length=None,
        max_new_token=16,
        deterministic=False,
    ):
        self.template = template
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.max_prompt_length = max_prompt_length
        self.max_prompt_length_words = int(max_prompt_length) * 2 / 3  # magic number
        self.n_generated_queries = n_generated_queries
        self.max_new_token = max_new_token
        self.deterministic = deterministic

    @classmethod
    def load(cls, name: str, dataset: str = None, *args, **kwargs):
        if "inparsplus" in name:
            if dataset is None:
                raise ValueError("Dataset must be provided to load inparsplus prompt")
            name = f"inparsplus-{dataset.split('/')[0].lower()}"
        if name in templates:
            template = templates[name]
            prompt_class = {
                "dynamic": DynamicPrompt,
                "static": StaticPrompt,
                "dynamic_v2": DynamicPromptV2,
            }[template["mode"]]
            return prompt_class(template=template["template"], *args, **kwargs)
        else:
            if not os.path.exists(name):
                raise FileNotFoundError(f"Prompt file {name} was not found!")

            with open(name) as f:
                return StaticPrompt(template=f.read(), *args, **kwargs)

    def _truncate_max_doc_length(self, document):
        if self.max_doc_length:
            document = self.tokenizer.decode(
                self.tokenizer(
                    document, truncation=True, max_length=self.max_doc_length
                )["input_ids"],
                skip_special_tokens=True,
            )
        return document
 
    def _truncate_max_query_length(self, query):
        if self.max_query_length:
            query = self.tokenizer.decode(
                self.tokenizer(
                    query, truncation=True, max_length=self.max_query_length
                )["input_ids"],
                skip_special_tokens=True,
            )
        return query

    def check_max_prompt_length(self, prompt):
        if self.max_prompt_length:
            prompt_length = len(self.tokenizer.tokenize(prompt))
            if prompt_length + self.max_new_token > self.max_prompt_length:
                raise Exception(
                    f"Overflowing prompt (prompt length: {prompt_length} + {self.max_new_token}, \
                     max length: {self.max_prompt_length})"
                )


class StaticPrompt(Prompt):
    def build(self, document, *args, **kwargs):
        document = self._truncate_max_doc_length(document)

        prompt = self.template.format(document=document, query="").rstrip()

        self.check_max_prompt_length(prompt)
        return prompt


class DynamicPrompt(Prompt):
    def build(self, document, n_examples=3):
        if self.deterministic:
            random_examples = random.sample(self.examples, n_examples)
        else:
            random_examples = self.examples[:n_examples]

        prompt = ""
        for i in range(n_examples):
            _, _, query, doc = random_examples[i]
            query = self._truncate_max_query_length(query)
            doc = self._truncate_max_doc_length(doc)

            prompt += self.template.format(document=doc, query=query)

        document = ftfy.fix_text(document)
        document = self._truncate_max_doc_length(document)

        prompt += self.template.format(document=document, query="").rstrip()

        self.check_max_prompt_length(prompt)
        return prompt


class DynamicPromptV2(Prompt):
    def build(self, document, n_examples=3, **dataset_stats) -> str:
        if self.deterministic:
            random_examples = self.examples[:n_examples]
        else:
            random_examples = random.sample(self.examples, n_examples)

        # Dataset_Stats is a dictionary with the following schema:
        # {
        #     "Dataset": str,
        #     "{} Words": int,
        #     "{} Tokens": int,
        # }
        # where {} is all of "Documents", "Queries" and "Total"
        qlen = dataset_stats.get("Queries Words", 30)  # more magic numbers
        dlen = dataset_stats.get("Documents Words", 300)

        template = self.template
        if not template:
            template = "Document: {document}\nRelevant Query: {query}"
        template = template.strip()

        if self.n_generated_queries > 1:
            # Assuming the last line is the query line and it contains a colon.
            context, _, query_str = template.rpartition("\n")
            _query_str = query_str.replace(":", " {i}:")
            _query_str = "\n".join(
                _query_str.replace("{query}", "{query_{i}}").replace("{i}", str(i))
                for i in range(1, self.n_generated_queries + 1)
            )
            template = f"{context}\n{_query_str}"

        prompt = ""
        for i in range(n_examples):
            _, _, query, doc = random_examples[i]
            query = self._truncate_max_query_length(query)
            doc = self._truncate_max_doc_length(doc)

            if (
                len(prompt.split())
                + len(doc.split())
                + self.n_generated_queries * len(query.split())
                + dlen
                + self.n_generated_queries * qlen
                - self.max_prompt_length_words
                < 0  # needs to be <0 instead, in case we don't provide a check for max_prompt_length
            ):
                break

            if self.n_generated_queries > 1:
                query_text_split = query.split()
                queries = {"query_1": query} | {
                    f"query_{i}": " ".join(
                        random.sample(query_text_split, len(query_text_split))
                    )
                    for i in range(2, self.n_generated_queries + 1)
                }
                prompt += template.format(document=doc.strip(), **queries) + "\n\n"
            else:
                prompt += template.format(document=doc.strip(), query=query) + "\n\n"

        document = ftfy.fix_text(document)
        document = self._truncate_max_doc_length(document)

        if self.n_generated_queries > 1:
            prompt += (
                f"{context.format(document=document.strip())}\n{query_str}".rstrip()
            )
        else:
            prompt += template.format(document=document.strip(), query="").rstrip()

        self.check_max_prompt_length(prompt)
        return prompt
