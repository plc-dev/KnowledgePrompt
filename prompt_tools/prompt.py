import random as rng
from typing import List, Dict, Any, Callable, NamedTuple
import re
from enum import Enum
import string

class PromptInstructionLength(Enum):
    one_line = 1
    two_lines = 2
    three_lines = 3

class EnumerationRule(Enum):
    only_first = 1
    all = 2
    none = 3

class Instruction(NamedTuple):
    sample_size: int
    enumeration_rule: EnumerationRule
    problem_formulation: str
    repeat_problem: bool
    descriptors: List[str]
    example_delimeter: str
    example_templates: List[str]
    instruction_length: PromptInstructionLength
    global_delimeter: str


class PromptPattern:
    """
    class for defining a prompt pattern \n
    param:
        - templates:    list of templates to form the prompt
        - examples:     callable that returns a list of variables, required to fill the templates
    """
    __prompt_base_template = """$PROBLEM\n\n$INSTRUCTION"""

    __prompt_instruction_templates = {
        PromptInstructionLength.one_line: """$DESCRIPTOR1$ENUM1$DELIM $EXAMPLE1""",
        PromptInstructionLength.two_lines: """$DESCRIPTOR1$ENUM1$DELIM $EXAMPLE1\n$DESCRIPTOR2$ENUM2$DELIM $EXAMPLE2""",
        PromptInstructionLength.three_lines: """$DESCRIPTOR1$ENUM1$DELIM $EXAMPLE1\n$DESCRIPTOR2$ENUM2$DELIM $EXAMPLE2\n$DESCRIPTOR3$ENUM3$DELIM $EXAMPLE3"""
    }

    def __init__(
        self, 
        instruction: Dict,
        get_examples: Callable[[], List[Dict[str, str]]],
        relation: str
    ) -> None:
        self.instruction = instruction
        self.get_examples = get_examples
        self.relation = relation

    def generate_prompt(self):
        """
        generate random prompt according to the given instructions
        """
        problem_formulation = "" if self.instruction["repeat_formulation"] else self.instruction["problem_formulation"]
        prompt_template_variables = {
            "PROBLEM": self.instruction["problem_formulation"],
            "INSTRUCTION": self.__fill_instruction_template()
        }
        prompt = self.__fill_template(
            self.__prompt_base_template,
            prompt_template_variables
        )

        return prompt

    def __remove_all_whitespace(self, s: str) -> str:
        return s.translate({
            ord(c): None 
            for c in string.whitespace
        })

    def __fill_template(self, template: str, variables: Dict[str, str]) -> str:
        """
        template: string of shape "This is a template with two variables $VAR1 and $VAR2."
        variables: dict of shape { "VAR1": "some text", "VAR2": "some more text" }
        """
        for key, value in variables.items():
            pattern = re.compile(f"\${key}")
            template = re.sub(pattern, value, template)
        return template

    def __sample_examples(self, population: List[Any], sample_size: int) -> List[Any]:
        """
        return a sample of size <sample_size> without returning from a given population
        """
        sample = rng.sample(population, k=sample_size)
        return sample

    def __fill_instruction_template(self) -> str:
        """
        fill instructions with sampled examples according to the passed instruction parameters
        """
        instruction_length = self.instruction["instruction_length"]
        instruction_template = self.__prompt_instruction_templates[instruction_length]
        example_delimeter = self.instruction["example_delimeter"]
        global_delimeter = self.instruction["global_delimeter"] + \
            self.instruction["problem_formulation"] + \
            self.instruction["global_delimeter"] if self.instruction["repeat_formulation"] else self.instruction["global_delimeter"]
        enumeration_rule = self.instruction["enumeration_rule"]
        descriptors = self.instruction["descriptors"]
        instruction_template_variables = {
            "DELIM": example_delimeter
        }

        sample_size = self.instruction["sample_size"]
        samples = self.__sample_examples(self.get_examples(), sample_size)

        instructions = []
        for i, sample in enumerate(samples):
            example_dict = {
                f"EXAMPLE{j+1}": self.__fill_template(example_template, sample)
                for j, example_template in enumerate(self.instruction["example_templates"])
            }
            descriptor_dict = {
                f"DESCRIPTOR{j+1}": descriptor
                for j, descriptor in enumerate(descriptors)
            }

            enums = {
                "ENUM1": f" {i+1}.)" if (enumeration_rule == EnumerationRule.only_first) or enumeration_rule == EnumerationRule.all else "",
                "ENUM2": f" {i+1}.)" if enumeration_rule == EnumerationRule.all else "",
                "ENUM3": f" {i+1}.)" if enumeration_rule == EnumerationRule.all else ""
            }

            instruction_template_variables = {
                **instruction_template_variables,
                **example_dict,
                **descriptor_dict,
                **enums
            }

            instructions.append(self.__fill_template(
                instruction_template,
                instruction_template_variables
            ))

        return global_delimeter.join(instructions)
