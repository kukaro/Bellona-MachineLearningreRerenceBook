import numpy as np
from reference_book_scratch.a_perceptron.nand_gate import nand_gate
from reference_book_scratch.a_perceptron.and_gate import and_gate
from reference_book_scratch.a_perceptron.or_gate import or_gate


def xor_gate(x1, x2):
    s1 = nand_gate(x1, x2)
    s2 = or_gate(x1, x2)
    return and_gate(s1, s2)