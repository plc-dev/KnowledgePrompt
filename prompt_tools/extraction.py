from typing import Optional, Callable, List, Union, NamedTuple
import re
from enum import Enum

class Predicate(Enum):
    isA = 0
    consistsOf = 1
    hasA = 2
    can = 3

class ExtractionGroup(NamedTuple):
    subject: Optional[int]
    predicate: Optional[int]
    object: Optional[int]

class GroupProcessing(NamedTuple):
    subject: Callable[[str], str]
    predicate: Callable[[str], Union[Predicate, str]]
    object: Callable[[str], List[str]]

class ExtractionSpecification(NamedTuple):
    extraction_regex: re.Pattern
    expected_groups: int
    extraction_ordering: ExtractionGroup
    processing_functions: GroupProcessing

class TripleExtraction:
    def __init__(self, extraction_specification: ExtractionSpecification):
        self.extraction_specification = extraction_specification

    def extract_triples(self, output: str, prompt: str) -> List[List[str]]:
        """
        return structured triples from extracted patterns of the model output in the shape of [ [<s: str>, <p: str>, <o: str>], ... ]
        """
        clean_output = self.__clean_output(output, prompt)
        extraction_regex = self.extraction_specification["extraction_regex"]

        triples = []
        for match in re.finditer(extraction_regex, clean_output, re.DOTALL):
            triples.extend(self.__process_match(match))

        return triples

    def __clean_output(self, output: str, prompt: str) -> str:
        """
        remove original prompt from the output
        """
        return output.replace(prompt, "").strip()

    def __process_match(self, match: re.Match) -> List[str]:
        """
        process a regex match in the output into structured triples according to the extraction_specification
        """
        triples = []
        if len(match.groups()) == self.extraction_specification["expected_groups"]:
            subject = self.__process_group(match, "subject")
            predicate = self.__process_group(match, "predicate")
            objects = self.__process_group(match, "object")
            
            for object in objects:
                triple = {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "source_id": 1
                }
                triples.append(triple)
        return triples

    def __process_group(self, match: re.Match, key: str) -> None:
        """
        apply post processing to transform the extracted sequence to the atomic triple element(s)
        """
        group_index = self.extraction_specification["extraction_groups"][key]
        process_function = self.extraction_specification["processing_functions"][key]

        group = match.group(group_index)
        return process_function(group)
