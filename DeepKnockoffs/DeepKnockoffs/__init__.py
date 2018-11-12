from pkg_resources import get_distribution
__version__ = get_distribution('DeepKnockoffs').version

from DeepKnockoffs.machine import KnockoffMachine
from DeepKnockoffs.gaussian import GaussianKnockoffs
