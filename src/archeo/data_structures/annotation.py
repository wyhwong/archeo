from typing import TypeAlias, Union

from archeo.data_structures.distribution import Normal, PiecewiseUniform, Uniform
from archeo.data_structures.physics.mahapatra import MahapatraMassFunction


Interface: TypeAlias = object
Distribution: TypeAlias = Union[Uniform, Normal, PiecewiseUniform, MahapatraMassFunction]
